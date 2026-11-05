//! WP-A: loading the key table from its ConfigMap.
//!
//! Contract:
//!   - `KeySource::start` spawns a watch that calls `TenantManager::install_table`
//!     on every change, backed by a full resync every `config.resync_secs`.
//!   - A ConfigMap that fails to parse leaves the previous table in force; it
//!     must never clear the table, or every tenant loses access at once.
//!   - The ConfigMap carries plaintext keys under `data["tenants.yaml"]`; they
//!     are hashed into `TenantKey` at load and the plaintext is dropped.

use std::{collections::HashMap, sync::Arc, time::Duration};

use futures::TryStreamExt;
use k8s_openapi::api::core::v1::ConfigMap;
use kube::{
    api::Api,
    runtime::{
        watcher::{watcher, Config},
        WatchStreamExt,
    },
    Client,
};
use metrics::{counter, gauge};
use serde::Deserialize;
use tokio::{task, time};
use tracing::{debug, error, info, warn};

use super::{
    manager::TenantManager,
    types::{KeyRole, TenantKey, TenantSpec, TenantTable},
};

/// Key of the document inside the ConfigMap's `data` map.
pub const KEY_DOCUMENT: &str = "tenants.yaml";

const MAX_RETRY_DELAY: Duration = Duration::from_secs(300);

#[async_trait::async_trait]
pub trait KeySource: Send + Sync {
    async fn start(&self, manager: Arc<TenantManager>) -> Result<(), String>;
}

impl TenantManager {
    /// Installs a freshly loaded key table and re-applies this model's quota
    /// rules to it.
    pub fn install_table(&self, table: TenantTable) {
        let _guard = self.merge_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.keys_raw.store(Arc::new(table));
        self.remerge_locked();
    }
}

// ===== Document schema =====
//
// Unknown fields are tolerated rather than rejected: the ConfigMap is edited by
// hand and often by an older gateway than the one that wrote it, so a field
// added by a newer release must not lock every tenant out.

#[derive(Debug, Deserialize)]
struct KeyDocument {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    tenants: Vec<TenantDocument>,
}

#[derive(Debug, Deserialize)]
struct TenantDocument {
    id: String,
    #[serde(default)]
    name: String,
    #[serde(default)]
    keys: Vec<KeyDocumentEntry>,
    /// Ignored: quota is per model and lives in each model's own ConfigMap
    /// (see quota.rs). Still accepted so an older key table keeps loading.
    quota: Option<serde_yaml::Value>,
}

#[derive(Debug, Deserialize)]
struct KeyDocumentEntry {
    id: String,
    secret: String,
    #[serde(default)]
    role: KeyRole,
    #[serde(default)]
    disabled: bool,
}

/// Parses the `tenants.yaml` document into a table.
pub fn parse_key_table(yaml: &str, resource_version: String) -> Result<TenantTable, String> {
    let doc: KeyDocument =
        serde_yaml::from_str(yaml).map_err(|e| format!("invalid tenants.yaml: {}", e))?;
    if doc.version != 0 && doc.version != 1 {
        return Err(format!(
            "unsupported tenants.yaml version {}; this gateway understands 1",
            doc.version
        ));
    }

    let mut tenants: HashMap<_, TenantSpec> = HashMap::new();
    let mut by_hash: HashMap<[u8; 32], TenantKey> = HashMap::new();

    for tenant in doc.tenants {
        if tenant.id.is_empty() {
            return Err("tenant with an empty id".to_string());
        }
        let tenant_id: Arc<str> = Arc::from(tenant.id.as_str());
        if tenants.contains_key(&tenant_id) {
            return Err(format!("duplicate tenant id {:?}", tenant.id));
        }

        if tenant.quota.is_some() {
            warn!(
                "tenant {:?}: `quota` in the key table is ignored; set it in each \
                 model's quota ConfigMap",
                tenant.id
            );
        }

        let mut seen_key_ids = HashMap::new();
        for entry in tenant.keys {
            if entry.id.is_empty() {
                return Err(format!("tenant {:?}: key with an empty id", tenant.id));
            }
            if entry.secret.is_empty() {
                return Err(format!(
                    "tenant {:?}: key {:?} has an empty secret",
                    tenant.id, entry.id
                ));
            }
            if seen_key_ids.insert(entry.id.clone(), ()).is_some() {
                return Err(format!(
                    "tenant {:?}: duplicate key id {:?}",
                    tenant.id, entry.id
                ));
            }
            let key = TenantKey::new(
                Arc::from(entry.id.as_str()),
                Arc::clone(&tenant_id),
                entry.role,
                entry.disabled,
                &entry.secret,
            );
            // A shared secret would make the tenant of a request ambiguous.
            if let Some(existing) = by_hash.get(key.hash()) {
                return Err(format!(
                    "key {:?} of tenant {:?} reuses the secret of key {:?} of tenant {:?}",
                    entry.id, tenant.id, existing.id, existing.tenant
                ));
            }
            by_hash.insert(*key.hash(), key);
        }

        tenants.insert(
            Arc::clone(&tenant_id),
            TenantSpec {
                id: tenant_id,
                name: tenant.name,
                quota: None,
            },
        );
    }

    Ok(TenantTable {
        tenants,
        by_hash,
        resource_version,
    })
}

/// Installs a parsed document, or keeps the table in force and reports why.
fn install_document(manager: &TenantManager, yaml: &str, resource_version: String) {
    match parse_key_table(yaml, resource_version) {
        Ok(table) => {
            let (tenants, keys) = (table.tenants.len(), table.by_hash.len());
            let version = table.resource_version.clone();
            manager.install_table(table);
            counter!("smg_tenancy_key_reload_total", "result" => "ok").increment(1);
            gauge!("smg_tenancy_tenants").set(tenants as f64);
            gauge!("smg_tenancy_keys").set(keys as f64);
            info!(
                "Tenancy key table loaded: {} tenants, {} keys (resourceVersion {})",
                tenants, keys, version
            );
        }
        Err(e) => {
            counter!("smg_tenancy_key_reload_total", "result" => "parse_error").increment(1);
            error!(
                "Tenancy key ConfigMap rejected, keeping previous table: {}",
                e
            );
        }
    }
}

fn document_of(cm: &ConfigMap) -> Option<(String, String)> {
    let yaml = cm.data.as_ref()?.get(KEY_DOCUMENT)?.clone();
    let version = cm
        .metadata
        .resource_version
        .clone()
        .unwrap_or_else(|| "unknown".to_string());
    Some((yaml, version))
}

/// Watches one ConfigMap in one namespace.
pub struct ConfigMapKeySource {
    client: Option<Client>,
}

impl ConfigMapKeySource {
    pub fn new() -> Self {
        Self { client: None }
    }

    /// Injects a client instead of building one from the ambient kubeconfig.
    pub fn with_client(client: Client) -> Self {
        Self {
            client: Some(client),
        }
    }
}

impl Default for ConfigMapKeySource {
    fn default() -> Self {
        Self::new()
    }
}

#[async_trait::async_trait]
impl KeySource for ConfigMapKeySource {
    async fn start(&self, manager: Arc<TenantManager>) -> Result<(), String> {
        let client = match &self.client {
            Some(client) => client.clone(),
            None => Client::try_default()
                .await
                .map_err(|e| format!("failed to create Kubernetes client: {}", e))?,
        };

        let config = manager.config().clone();
        let api: Api<ConfigMap> = Api::namespaced(client, &config.namespace);

        // An initial get means the table is populated before the first request,
        // rather than whenever the watch happens to deliver its first event.
        match api.get(&config.keys_configmap).await {
            Ok(cm) => match document_of(&cm) {
                Some((yaml, version)) => install_document(&manager, &yaml, version),
                None => warn!(
                    "Tenancy key ConfigMap {}/{} has no {:?} entry",
                    config.namespace, config.keys_configmap, KEY_DOCUMENT
                ),
            },
            Err(e) => warn!(
                "Tenancy key ConfigMap {}/{} not readable at startup: {}",
                config.namespace, config.keys_configmap, e
            ),
        }

        let watch_api = api.clone();
        let watch_manager = Arc::clone(&manager);
        let watch_name = config.keys_configmap.clone();
        task::spawn(async move {
            let mut retry_delay = Duration::from_secs(1);
            loop {
                let watcher_config =
                    Config::default().fields(&format!("metadata.name={}", watch_name));
                let stream = watcher(watch_api.clone(), watcher_config).applied_objects();

                let inner = Arc::clone(&watch_manager);
                let result = stream
                    .try_for_each(move |cm| {
                        let manager = Arc::clone(&inner);
                        async move {
                            match document_of(&cm) {
                                Some((yaml, version)) => install_document(&manager, &yaml, version),
                                None => {
                                    warn!("Tenancy key ConfigMap has no {:?} entry", KEY_DOCUMENT)
                                }
                            }
                            Ok(())
                        }
                    })
                    .await;

                match result {
                    Ok(_) => retry_delay = Duration::from_secs(1),
                    Err(e) => {
                        error!("Tenancy key watcher error: {}", e);
                        counter!("smg_tenancy_key_reload_total", "result" => "watch_error")
                            .increment(1);
                        time::sleep(retry_delay).await;
                        retry_delay = std::cmp::min(retry_delay * 2, MAX_RETRY_DELAY);
                    }
                }
            }
        });

        if config.resync_secs > 0 {
            let resync_api = api;
            let resync_manager = manager;
            let resync_name = config.keys_configmap.clone();
            let period = Duration::from_secs(config.resync_secs);
            task::spawn(async move {
                let mut ticker = time::interval(period);
                ticker.tick().await;
                loop {
                    ticker.tick().await;
                    match resync_api.get(&resync_name).await {
                        Ok(cm) => {
                            if let Some((yaml, version)) = document_of(&cm) {
                                if version != resync_manager.table().resource_version {
                                    debug!("Tenancy key resync found a missed update");
                                    install_document(&resync_manager, &yaml, version);
                                }
                            }
                        }
                        Err(e) => warn!("Tenancy key resync failed: {}", e),
                    }
                }
            });
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tenancy::TenancyConfig;

    const DOC: &str = r#"
version: 1
tenants:
  - id: team-a
    name: "Team A"
    keys:
      - {id: k1, secret: "secret-a1", role: user}
      - {id: k2, secret: "secret-a2", role: admin, disabled: true}
    quota: {initial: 32, floor: 0, ceiling: 64}
  - id: team-b
    name: "Team B"
    keys:
      - {id: k1, secret: "secret-b1"}
    quota: {initial: 8, floor: 4, ceiling: 16}
"#;

    #[test]
    fn parses_tenants_and_keys() {
        let table = parse_key_table(DOC, "42".to_string()).unwrap();
        assert_eq!(table.tenants.len(), 2);
        assert_eq!(table.by_hash.len(), 3);
        assert_eq!(table.resource_version, "42");

        let spec = table.tenants.get("team-a").unwrap();
        assert_eq!(spec.name, "Team A");
        // A quota block in the key table is accepted but no longer applied.
        assert_eq!(spec.quota, None);

        let key = table
            .by_hash
            .get(&super::super::types::hash_key("secret-a2"))
            .unwrap();
        assert_eq!(key.role, KeyRole::Admin);
        assert!(key.disabled);
        assert_eq!(key.tenant.as_ref(), "team-a");
    }

    #[test]
    fn a_tenant_with_no_quota_block_is_unlimited() {
        let doc = r#"
version: 1
tenants:
  - id: team-a
    name: "Team A"
    keys:
      - {id: k1, secret: "secret-a1"}
"#;
        let table = parse_key_table(doc, "1".to_string()).unwrap();
        let spec = table.tenants.get("team-a").unwrap();
        assert_eq!(spec.quota, None);
    }

    #[test]
    fn tolerates_unknown_fields() {
        let doc = r#"
version: 1
tenants:
  - id: team-a
    unknown_tenant_field: 1
    keys: [{id: k1, secret: s1, future_flag: true}]
    quota: {initial: 1, ceiling: 2, unknown: 9}
"#;
        let table = parse_key_table(doc, String::new()).unwrap();
        assert_eq!(table.by_hash.len(), 1);
    }

    #[test]
    fn rejects_duplicate_secret_across_tenants() {
        let doc = r#"
tenants:
  - id: team-a
    keys: [{id: k1, secret: same}]
  - id: team-b
    keys: [{id: k1, secret: same}]
"#;
        let err = parse_key_table(doc, String::new()).unwrap_err();
        assert!(err.contains("reuses the secret"), "{}", err);
    }

    #[test]
    fn rejects_duplicate_tenant_and_key_ids() {
        let dup_tenant = "tenants:\n  - id: a\n    keys: []\n  - id: a\n    keys: []\n";
        assert!(parse_key_table(dup_tenant, String::new())
            .unwrap_err()
            .contains("duplicate tenant id"));

        let dup_key = "tenants:\n  - id: a\n    keys: [{id: k, secret: s1}, {id: k, secret: s2}]\n";
        assert!(parse_key_table(dup_key, String::new())
            .unwrap_err()
            .contains("duplicate key id"));
    }

    #[test]
    fn malformed_document_keeps_the_previous_table() {
        let manager = TenantManager::new(TenancyConfig::default());
        install_document(&manager, DOC, "1".to_string());
        assert_eq!(manager.table().by_hash.len(), 3);

        install_document(&manager, "tenants: [this is not a tenant]", "2".to_string());
        assert_eq!(manager.table().by_hash.len(), 3);
        assert_eq!(manager.table().resource_version, "1");
    }
}
