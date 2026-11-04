//! WP-C: persisting allocations to the alloc ConfigMap.
//!
//! Contract:
//!   - In-memory state is authoritative between writes; the ConfigMap exists so
//!     allocations survive a restart.
//!   - Updates carry the observed `resourceVersion`; on 409 re-read and retry.
//!   - `AllocTable::generation` is bumped on every gateway-originated write so
//!     the watcher can ignore its own echo instead of reloading in a loop.
//!   - Writes are coalesced (idle flush, with a floor on write frequency) so a
//!     burst of resizes does not hammer the API server.
//!   - The gateway needs get/list/watch/update on this ConfigMap only; the key
//!     ConfigMap stays read-only so a compromised gateway cannot mint keys.

use std::{
    sync::{
        atomic::{AtomicU64, Ordering},
        Arc, OnceLock,
    },
    time::Duration,
};

use async_trait::async_trait;
use futures::TryStreamExt;
use k8s_openapi::api::core::v1::ConfigMap;
use kube::{
    api::{Api, PostParams},
    runtime::{
        watcher::{watcher, Config as WatcherConfig},
        WatchStreamExt,
    },
    Client,
};
use tokio::{
    sync::mpsc::{self, UnboundedSender},
    time::Instant,
};
use tracing::{debug, error, info, warn};

use super::{manager::TenantManager, types::AllocTable};

/// Key holding the serialized `AllocTable` inside the ConfigMap.
pub const ALLOC_DATA_KEY: &str = "alloc.yaml";

/// Signals the flusher that in-memory allocations moved. A process runs one
/// gateway, so one global sender is enough; it stays unset in unit tests and
/// `mutate_alloc` then only mutates memory.
static FLUSH_SIGNAL: OnceLock<UnboundedSender<()>> = OnceLock::new();

impl TenantManager {
    pub fn install_alloc(&self, table: AllocTable) {
        self.alloc.store(Arc::new(table));
    }

    /// Applies a mutation under the write lock and schedules a write-back.
    pub async fn mutate_alloc<F>(&self, mutate: F) -> Result<(), String>
    where
        F: FnOnce(&mut AllocTable),
    {
        let _guard = self.alloc_write_lock.lock().await;
        let mut next = (**self.alloc.load()).clone();
        mutate(&mut next);
        next.generation = next.generation.saturating_add(1);
        self.alloc.store(Arc::new(next));
        if let Some(tx) = FLUSH_SIGNAL.get() {
            let _ = tx.send(());
        }
        Ok(())
    }
}

/// How hard the flusher is allowed to hit the API server.
#[derive(Debug, Clone, Copy)]
pub struct WritePolicy {
    /// Quiet period after the last mutation before a write goes out.
    pub idle: Duration,
    /// Lower bound on the gap between two writes.
    pub min_interval: Duration,
    /// Attempts for one flush, counting the first; each 409 costs one.
    pub max_attempts: u32,
}

impl Default for WritePolicy {
    fn default() -> Self {
        Self {
            idle: Duration::from_millis(500),
            min_interval: Duration::from_secs(1),
            max_attempts: 5,
        }
    }
}

#[derive(Debug)]
pub enum BackendError {
    /// The stored object moved since it was read; re-read and retry.
    Conflict,
    Other(String),
}

impl std::fmt::Display for BackendError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            BackendError::Conflict => write!(f, "conflict"),
            BackendError::Other(e) => write!(f, "{}", e),
        }
    }
}

/// The persistence surface, kept behind a trait so the retry and coalescing
/// logic is testable without a cluster.
#[async_trait]
pub trait AllocBackend: Send + Sync {
    /// Returns the stored table and the resourceVersion it was read at. A
    /// ConfigMap that does not exist yet reads as an empty table.
    async fn read(&self) -> Result<(AllocTable, String), BackendError>;

    /// Writes `table` at `resource_version`, returning the new version.
    async fn write(
        &self,
        table: &AllocTable,
        resource_version: &str,
    ) -> Result<String, BackendError>;
}

pub struct KubeAllocBackend {
    api: Api<ConfigMap>,
    name: String,
}

impl KubeAllocBackend {
    pub fn new(client: Client, namespace: &str, name: &str) -> Self {
        Self {
            api: Api::namespaced(client, namespace),
            name: name.to_string(),
        }
    }
}

fn classify(err: kube::Error) -> BackendError {
    match err {
        kube::Error::Api(ref resp) if resp.code == 409 => BackendError::Conflict,
        other => BackendError::Other(other.to_string()),
    }
}

#[async_trait]
impl AllocBackend for KubeAllocBackend {
    async fn read(&self) -> Result<(AllocTable, String), BackendError> {
        match self.api.get_opt(&self.name).await.map_err(classify)? {
            Some(cm) => {
                let version = cm.metadata.resource_version.clone().unwrap_or_default();
                Ok((parse_alloc_configmap(&cm), version))
            }
            None => Ok((AllocTable::default(), String::new())),
        }
    }

    async fn write(
        &self,
        table: &AllocTable,
        resource_version: &str,
    ) -> Result<String, BackendError> {
        let rendered = render_alloc(table).map_err(BackendError::Other)?;
        let mut cm = ConfigMap {
            metadata: kube::core::ObjectMeta {
                name: Some(self.name.clone()),
                ..Default::default()
            },
            data: Some(
                [(ALLOC_DATA_KEY.to_string(), rendered)]
                    .into_iter()
                    .collect(),
            ),
            ..Default::default()
        };
        // An empty version means the object was absent at read time: create it,
        // and let a racing creator surface as a conflict for the retry loop.
        let written = if resource_version.is_empty() {
            self.api
                .create(&PostParams::default(), &cm)
                .await
                .map_err(|e| match e {
                    kube::Error::Api(ref resp) if resp.code == 409 => BackendError::Conflict,
                    other => classify(other),
                })?
        } else {
            cm.metadata.resource_version = Some(resource_version.to_string());
            self.api
                .replace(&self.name, &PostParams::default(), &cm)
                .await
                .map_err(classify)?
        };
        Ok(written.metadata.resource_version.unwrap_or_default())
    }
}

pub fn render_alloc(table: &AllocTable) -> Result<String, String> {
    serde_yaml::to_string(table).map_err(|e| e.to_string())
}

pub fn parse_alloc(yaml: &str) -> Result<AllocTable, String> {
    serde_yaml::from_str(yaml).map_err(|e| e.to_string())
}

/// Missing or unparsable data yields an empty table rather than an error: the
/// ConfigMap is created empty on first write and a corrupt one must not stop
/// the gateway from serving.
fn parse_alloc_configmap(cm: &ConfigMap) -> AllocTable {
    cm.data
        .as_ref()
        .and_then(|d| d.get(ALLOC_DATA_KEY))
        .and_then(|raw| match parse_alloc(raw) {
            Ok(table) => Some(table),
            Err(e) => {
                warn!(
                    "alloc ConfigMap does not parse, ignoring its content: {}",
                    e
                );
                None
            }
        })
        .unwrap_or_default()
}

pub struct AllocStore {
    manager: Arc<TenantManager>,
    backend: Arc<dyn AllocBackend>,
    policy: WritePolicy,
    /// Generation of the last write this gateway made, so the watch can tell
    /// its own echo from an external edit.
    last_written_generation: AtomicU64,
}

impl AllocStore {
    pub fn new(
        manager: Arc<TenantManager>,
        backend: Arc<dyn AllocBackend>,
        policy: WritePolicy,
    ) -> Arc<Self> {
        Arc::new(Self {
            manager,
            backend,
            policy,
            last_written_generation: AtomicU64::new(0),
        })
    }

    pub fn manager(&self) -> &Arc<TenantManager> {
        &self.manager
    }

    /// Mutates allocations in memory and schedules persistence.
    pub async fn mutate<F>(&self, mutate: F) -> Result<(), String>
    where
        F: FnOnce(&mut AllocTable),
    {
        self.manager.mutate_alloc(mutate).await
    }

    /// True when an observed table is this gateway's own write coming back.
    pub fn is_self_echo(&self, observed: &AllocTable) -> bool {
        observed.generation == self.last_written_generation.load(Ordering::Relaxed)
            && observed.generation != 0
    }

    /// Loads the persisted table at startup, before any request is served.
    pub async fn load_initial(&self) -> Result<(), String> {
        let (table, _version) = self.backend.read().await.map_err(|e| e.to_string())?;
        info!(
            "loaded {} persisted tenant allocations (generation {})",
            table.allocated.len(),
            table.generation
        );
        self.manager.install_alloc(table);
        Ok(())
    }

    /// Writes the current in-memory table, re-reading on conflict. The in-memory
    /// table wins: a conflicting external edit is overwritten, since allocation
    /// is the gateway's own state.
    pub async fn flush_once(&self) -> Result<(), String> {
        let mut attempt = 0;
        loop {
            attempt += 1;
            let (_, version) = self.backend.read().await.map_err(|e| e.to_string())?;
            let table = (**self.manager.alloc.load()).clone();
            match self.backend.write(&table, &version).await {
                Ok(_) => {
                    self.last_written_generation
                        .store(table.generation, Ordering::Relaxed);
                    debug!(
                        "persisted {} allocations at generation {}",
                        table.allocated.len(),
                        table.generation
                    );
                    return Ok(());
                }
                Err(BackendError::Conflict) if attempt < self.policy.max_attempts => {
                    debug!("alloc ConfigMap changed under us, retry {}", attempt);
                }
                Err(e) => return Err(e.to_string()),
            }
        }
    }

    /// Spawns the flusher and the watch. Registers the global flush signal, so
    /// `mutate_alloc` called from anywhere (including the pool rescale) lands
    /// here.
    pub fn start(self: &Arc<Self>, client: Option<Client>) {
        let (tx, mut rx) = mpsc::unbounded_channel::<()>();
        if FLUSH_SIGNAL.set(tx).is_err() {
            warn!("alloc flush signal already registered; this store will not persist");
            return;
        }

        let store = Arc::clone(self);
        tokio::spawn(async move {
            let mut last_write: Option<Instant> = None;
            while rx.recv().await.is_some() {
                // Coalesce: absorb further mutations until the stream goes quiet.
                loop {
                    match tokio::time::timeout(store.policy.idle, rx.recv()).await {
                        Ok(Some(())) => continue,
                        Ok(None) => return,
                        Err(_) => break,
                    }
                }
                if let Some(prev) = last_write {
                    let since = prev.elapsed();
                    if since < store.policy.min_interval {
                        tokio::time::sleep(store.policy.min_interval - since).await;
                    }
                }
                if let Err(e) = store.flush_once().await {
                    error!("failed to persist tenant allocations: {}", e);
                }
                last_write = Some(Instant::now());
            }
        });

        let Some(client) = client else {
            return;
        };
        let store = Arc::clone(self);
        tokio::spawn(async move {
            store.watch(client).await;
        });
    }

    async fn watch(self: Arc<Self>, client: Client) {
        let cfg = self.manager.config();
        let api: Api<ConfigMap> = Api::namespaced(client, &cfg.namespace);
        let name = cfg.alloc_configmap.clone();
        let resync = Duration::from_secs(cfg.resync_secs);
        let mut backoff = Duration::from_secs(1);
        const MAX_BACKOFF: Duration = Duration::from_secs(300);

        loop {
            let wc = WatcherConfig::default().fields(&format!("metadata.name={}", name));
            let stream = watcher(api.clone(), wc).applied_objects();
            let store = Arc::clone(&self);
            let result = stream
                .try_for_each(move |cm| {
                    let store = Arc::clone(&store);
                    async move {
                        let observed = parse_alloc_configmap(&cm);
                        if store.is_self_echo(&observed) {
                            return Ok(());
                        }
                        info!(
                            "alloc ConfigMap changed externally (generation {}), adopting it",
                            observed.generation
                        );
                        store.manager.install_alloc(observed);
                        Ok(())
                    }
                })
                .await;

            match result {
                Ok(()) => backoff = Duration::from_secs(1),
                Err(e) => {
                    error!("alloc ConfigMap watch failed: {}", e);
                    tokio::time::sleep(backoff).await;
                    backoff = (backoff * 2).min(MAX_BACKOFF);
                    continue;
                }
            }
            // A watch that ends cleanly is the resync boundary.
            tokio::time::sleep(resync).await;
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::Mutex;

    use super::*;
    use crate::tenancy::TenancyConfig;

    struct MockBackend {
        table: Mutex<AllocTable>,
        version: Mutex<u64>,
        /// Conflicts to serve before the next write succeeds.
        conflicts: Mutex<u32>,
        writes: Mutex<u32>,
    }

    impl MockBackend {
        fn new(conflicts: u32) -> Arc<Self> {
            Arc::new(Self {
                table: Mutex::new(AllocTable::default()),
                version: Mutex::new(1),
                conflicts: Mutex::new(conflicts),
                writes: Mutex::new(0),
            })
        }
    }

    #[async_trait]
    impl AllocBackend for MockBackend {
        async fn read(&self) -> Result<(AllocTable, String), BackendError> {
            Ok((
                self.table.lock().unwrap().clone(),
                self.version.lock().unwrap().to_string(),
            ))
        }

        async fn write(&self, table: &AllocTable, _rv: &str) -> Result<String, BackendError> {
            let mut remaining = self.conflicts.lock().unwrap();
            if *remaining > 0 {
                *remaining -= 1;
                let mut v = self.version.lock().unwrap();
                *v += 1;
                return Err(BackendError::Conflict);
            }
            *self.writes.lock().unwrap() += 1;
            *self.table.lock().unwrap() = table.clone();
            let mut v = self.version.lock().unwrap();
            *v += 1;
            Ok(v.to_string())
        }
    }

    fn store_with(backend: Arc<MockBackend>) -> (Arc<AllocStore>, Arc<TenantManager>) {
        let manager = TenantManager::new(TenancyConfig::default());
        let store = AllocStore::new(Arc::clone(&manager), backend, WritePolicy::default());
        (store, manager)
    }

    #[tokio::test]
    async fn mutate_bumps_generation_and_persists() {
        let backend = MockBackend::new(0);
        let (store, manager) = store_with(Arc::clone(&backend));

        store
            .mutate(|t| {
                t.allocated.insert("team-a".to_string(), 8);
            })
            .await
            .unwrap();
        assert_eq!(manager.alloc().generation, 1);

        store.flush_once().await.unwrap();
        assert_eq!(*backend.writes.lock().unwrap(), 1);
        assert_eq!(
            backend.table.lock().unwrap().allocated.get("team-a"),
            Some(&8)
        );
    }

    #[tokio::test]
    async fn flush_retries_on_conflict() {
        let backend = MockBackend::new(2);
        let (store, _manager) = store_with(Arc::clone(&backend));

        store
            .mutate(|t| {
                t.allocated.insert("team-a".to_string(), 4);
            })
            .await
            .unwrap();
        store.flush_once().await.unwrap();

        assert_eq!(*backend.writes.lock().unwrap(), 1);
        assert_eq!(*backend.conflicts.lock().unwrap(), 0);
    }

    #[tokio::test]
    async fn flush_gives_up_after_max_attempts() {
        let backend = MockBackend::new(99);
        let manager = TenantManager::new(TenancyConfig::default());
        let store = AllocStore::new(
            manager,
            Arc::clone(&backend) as Arc<dyn AllocBackend>,
            WritePolicy {
                max_attempts: 3,
                ..Default::default()
            },
        );

        assert!(store.flush_once().await.is_err());
        assert_eq!(*backend.conflicts.lock().unwrap(), 96);
    }

    #[tokio::test]
    async fn self_echo_is_suppressed_but_external_edit_is_adopted() {
        let backend = MockBackend::new(0);
        let (store, _manager) = store_with(backend);

        store
            .mutate(|t| {
                t.allocated.insert("team-a".to_string(), 2);
            })
            .await
            .unwrap();
        store.flush_once().await.unwrap();

        let echo = AllocTable {
            allocated: [("team-a".to_string(), 2)].into_iter().collect(),
            generation: 1,
        };
        assert!(store.is_self_echo(&echo));

        let external = AllocTable {
            allocated: [("team-a".to_string(), 99)].into_iter().collect(),
            generation: 7,
        };
        assert!(!store.is_self_echo(&external));
    }

    #[test]
    fn generation_zero_is_never_an_echo() {
        let backend = MockBackend::new(0);
        let manager = TenantManager::new(TenancyConfig::default());
        let store = AllocStore::new(manager, backend, WritePolicy::default());
        assert!(!store.is_self_echo(&AllocTable::default()));
    }

    #[test]
    fn alloc_table_round_trips() {
        let table = AllocTable {
            allocated: [("team-a".to_string(), 16), ("team-b".to_string(), 4)]
                .into_iter()
                .collect(),
            generation: 3,
        };
        let parsed = parse_alloc(&render_alloc(&table).unwrap()).unwrap();
        assert_eq!(parsed.generation, 3);
        assert_eq!(parsed.allocated.get("team-a"), Some(&16));
    }

    #[test]
    fn corrupt_configmap_reads_as_empty() {
        let cm = ConfigMap {
            data: Some(
                [(ALLOC_DATA_KEY.to_string(), "}{not yaml".to_string())]
                    .into_iter()
                    .collect(),
            ),
            ..Default::default()
        };
        assert!(parse_alloc_configmap(&cm).allocated.is_empty());
    }
}
