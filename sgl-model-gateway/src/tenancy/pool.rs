//! WP-B: sizing the shared concurrency pool from worker capacity.
//!
//! `/server_info.max_running_requests` is the server-wide cap across attention-DP
//! ranks, not a per-rank one: `build_server_info` (python/sglang/srt/utils/common.py)
//! merges `server_args.resolved_dict()` with the scheduler handshake, and that
//! handshake carries no `max_running_requests`, so the resolved CLI value is what
//! is published; sglang derives the per-rank cap by dividing it by `attn_dp_size`
//! (python/sglang/srt/model_executor/pool_configurator.py). Under `--dp-aware` one
//! server appears as many `url@rank` workers, so capacity is summed over distinct
//! `base_url()` values -- summing the expanded list would multiply by the DP degree.
//!
//! Capacity is sampled rather than pushed: `max_running_requests` is parsed into
//! `ServerInfo` but never stored on the worker, and polling also covers
//! registration, removal and health transitions without a hook in shared code.

use std::{
    collections::{HashMap, HashSet},
    sync::{atomic::Ordering, Arc},
    time::Duration,
};

use tracing::{debug, warn};

use super::{manager::TenantManager, metrics, types::TenantId};
use crate::core::WorkerRegistry;

/// Only the one field is deserialized here: `discover_metadata::ServerInfo` is
/// private to the worker steps, and widening it is not worth a shared edit.
#[derive(serde::Deserialize)]
struct ServerCapacity {
    max_running_requests: Option<usize>,
}

/// Arbitrary; fast enough to follow a worker roll, slow enough to be invisible.
const SAMPLE_INTERVAL_SECS: u64 = 15;
/// Arbitrary; a worker too slow to answer this is not serving either.
const SAMPLE_TIMEOUT_SECS: u64 = 5;

/// New allocations to persist after a pool shrink, keyed by tenant id.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RescalePlan {
    pub allocations: HashMap<String, u32>,
}

impl TenantManager {
    /// Stores the pool derived from `worker_capacity` and reports the rescale
    /// needed when the tenants' allocations no longer fit. Allocations are left
    /// alone while they fit, so a pool that merely grew changes nothing.
    pub fn recompute_pool(&self, worker_capacity: u32) -> Option<RescalePlan> {
        let total = match self.config.pool_size_override {
            Some(fixed) => fixed,
            None => (worker_capacity as f32 * self.config.pool_headroom) as u32,
        };
        self.pool_total.store(total, Ordering::Relaxed);

        let table = self.table();
        // Unlimited tenants have nothing to rescale and nothing to report.
        let current: HashMap<String, u32> = table
            .tenants
            .keys()
            .filter_map(|t| Some((t.to_string(), self.allocated(t)?)))
            .collect();
        let floors: HashMap<String, u32> = table
            .tenants
            .iter()
            .filter_map(|(id, spec)| Some((id.to_string(), spec.quota?.floor)))
            .collect();
        for (tenant, allocated) in &current {
            metrics::set_tenant_allocated(tenant, *allocated);
        }

        let snapshot = self.pool_snapshot();
        metrics::set_pool(&snapshot);
        if !snapshot.oversubscribed {
            return None;
        }
        Some(RescalePlan {
            allocations: rescale_allocations(&current, &floors, total),
        })
    }

    /// Persists a shrink. Separate from `recompute_pool` so the arithmetic stays
    /// testable without the ConfigMap write path.
    pub async fn apply_rescale(&self, plan: RescalePlan) -> Result<(), String> {
        self.mutate_alloc(move |alloc| {
            for (tenant, value) in plan.allocations {
                alloc.allocated.insert(tenant, value);
            }
        })
        .await
    }
}

/// Scales every tenant by `total / allocated`, floored at its `Quota::floor`.
/// Honouring the floors can leave the sum above `total`; that is deliberate, a
/// floor is a promise the pool does not get to break.
pub fn rescale_allocations(
    current: &HashMap<String, u32>,
    floors: &HashMap<String, u32>,
    total: u32,
) -> HashMap<String, u32> {
    let allocated: u64 = current.values().map(|v| *v as u64).sum();
    if allocated == 0 {
        return current.clone();
    }
    current
        .iter()
        .map(|(tenant, value)| {
            let scaled = (*value as u64 * total as u64 / allocated) as u32;
            let floor = floors.get(tenant).copied().unwrap_or(0);
            (tenant.clone(), scaled.max(floor).min(*value))
        })
        .collect()
}

/// Sums `max_running_requests` over the distinct healthy servers behind the
/// registry. `None` when no server reported one, which leaves the pool as it is
/// rather than collapsing it to zero.
pub async fn sample_worker_capacity(
    client: &reqwest::Client,
    registry: &WorkerRegistry,
) -> Option<u32> {
    let mut seen: HashSet<String> = HashSet::new();
    let mut bases: Vec<String> = Vec::new();
    for worker in registry.get_all() {
        if !worker.is_healthy() {
            continue;
        }
        let base = worker.base_url().to_string();
        if seen.insert(base.clone()) {
            bases.push(base);
        }
    }

    let mut capacity: u32 = 0;
    let mut reported = false;
    for base in bases {
        match fetch_max_running_requests(client, &base).await {
            Some(value) => {
                capacity = capacity.saturating_add(value);
                reported = true;
            }
            None => debug!("tenancy: {} reported no max_running_requests", base),
        }
    }
    reported.then_some(capacity)
}

async fn fetch_max_running_requests(client: &reqwest::Client, base: &str) -> Option<u32> {
    let response = client
        .get(format!("{}/server_info", base))
        .timeout(Duration::from_secs(SAMPLE_TIMEOUT_SECS))
        .send()
        .await
        .ok()?;
    let info: ServerCapacity = response.json().await.ok()?;
    info.max_running_requests.map(|v| v as u32)
}

/// Keeps the pool following the fleet. Runs forever; cancel by dropping the task.
pub fn spawn_pool_sampler(
    manager: Arc<TenantManager>,
    client: reqwest::Client,
    registry: Arc<WorkerRegistry>,
) -> tokio::task::JoinHandle<()> {
    tokio::spawn(async move {
        let mut ticker = tokio::time::interval(Duration::from_secs(SAMPLE_INTERVAL_SECS));
        loop {
            ticker.tick().await;
            let Some(capacity) = sample_worker_capacity(&client, &registry).await else {
                continue;
            };
            metrics::set_worker_capacity(capacity);
            if let Some(plan) = manager.recompute_pool(capacity) {
                warn!(
                    "tenancy: pool shrank below the sum of allocations, rescaling {} tenants",
                    plan.allocations.len()
                );
                if let Err(err) = manager.apply_rescale(plan).await {
                    warn!("tenancy: failed to persist rescaled allocations: {}", err);
                }
            }
        }
    })
}

/// Current in-flight count, for the snapshot the API serves.
pub fn inflight_of(manager: &TenantManager, tenant: &TenantId) -> u32 {
    manager.inflight_counter(tenant).load(Ordering::Relaxed)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tenancy::{types::Quota, TenancyConfig};

    fn manager_with(tenants: &[(&str, Quota, u32)], config: TenancyConfig) -> Arc<TenantManager> {
        use crate::tenancy::types::{TenantSpec, TenantTable};
        let manager = TenantManager::new(config);
        let mut table = TenantTable::default();
        let mut alloc = crate::tenancy::types::AllocTable::default();
        for (id, quota, allocated) in tenants {
            let id: TenantId = Arc::from(*id);
            table.tenants.insert(
                id.clone(),
                TenantSpec {
                    id: id.clone(),
                    name: id.to_string(),
                    quota: Some(*quota),
                },
            );
            alloc.allocated.insert(id.to_string(), *allocated);
        }
        manager.install_table(table);
        manager.install_alloc(alloc);
        manager
    }

    fn quota(initial: u32, floor: u32, ceiling: u32) -> Quota {
        Quota {
            initial,
            floor,
            ceiling,
        }
    }

    #[test]
    fn override_ignores_worker_capacity() {
        let config = TenancyConfig {
            pool_size_override: Some(100),
            ..Default::default()
        };
        let manager = manager_with(&[("a", quota(10, 0, 50), 10)], config);
        assert!(manager.recompute_pool(9999).is_none());
        assert_eq!(manager.pool_snapshot().total, 100);
    }

    #[test]
    fn derived_pool_applies_headroom() {
        let config = TenancyConfig {
            pool_headroom: 0.8,
            ..Default::default()
        };
        let manager = manager_with(&[("a", quota(10, 0, 50), 10)], config);
        manager.recompute_pool(1280);
        assert_eq!(manager.pool_snapshot().total, 1024);
    }

    #[test]
    fn pool_above_allocations_leaves_them_alone() {
        let config = TenancyConfig {
            pool_size_override: Some(64),
            ..Default::default()
        };
        let manager = manager_with(
            &[("a", quota(16, 0, 64), 16), ("b", quota(16, 0, 64), 16)],
            config,
        );
        assert!(manager.recompute_pool(0).is_none());
        let snapshot = manager.pool_snapshot();
        assert_eq!((snapshot.allocated, snapshot.oversubscribed), (32, false));
    }

    #[test]
    fn oversubscribed_pool_rescales_proportionally() {
        let config = TenancyConfig {
            pool_size_override: Some(30),
            ..Default::default()
        };
        let manager = manager_with(
            &[("a", quota(40, 0, 80), 40), ("b", quota(20, 0, 80), 20)],
            config,
        );
        let plan = manager.recompute_pool(0).expect("expected a rescale");
        assert_eq!(plan.allocations.get("a"), Some(&20));
        assert_eq!(plan.allocations.get("b"), Some(&10));
    }

    #[test]
    fn rescale_respects_the_floor() {
        let current = HashMap::from([("a".to_string(), 40), ("b".to_string(), 20)]);
        let floors = HashMap::from([("a".to_string(), 0), ("b".to_string(), 16)]);
        let scaled = rescale_allocations(&current, &floors, 30);
        assert_eq!(scaled.get("a"), Some(&20));
        assert_eq!(scaled.get("b"), Some(&16));
    }

    #[test]
    fn rescale_never_grows_an_allocation() {
        let current = HashMap::from([("a".to_string(), 10)]);
        let floors = HashMap::from([("a".to_string(), 64)]);
        assert_eq!(
            rescale_allocations(&current, &floors, 5).get("a"),
            Some(&10)
        );
    }

    #[test]
    fn allocation_falls_back_to_the_spec_initial() {
        use crate::tenancy::types::{TenantSpec, TenantTable};
        let manager = TenantManager::new(TenancyConfig::default());
        let mut table = TenantTable::default();
        let id: TenantId = Arc::from("a");
        table.tenants.insert(
            id.clone(),
            TenantSpec {
                id: id.clone(),
                name: "a".to_string(),
                quota: Some(quota(12, 0, 64)),
            },
        );
        manager.install_table(table);
        assert_eq!(manager.allocated(&id), Some(12));
    }
}
