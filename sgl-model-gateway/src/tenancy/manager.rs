//! Shared state for the tenancy module. Fields are `pub(crate)` on purpose:
//! each work package adds its own `impl TenantManager` block in its own file
//! rather than editing this one.

use std::sync::{
    atomic::{AtomicU32, AtomicU64, Ordering},
    Arc,
};

use arc_swap::ArcSwap;
use dashmap::DashMap;

use super::{
    types::{AllocTable, PoolSnapshot, TenantId, TenantTable},
    TenancyConfig,
};

pub struct TenantManager {
    pub(crate) config: TenancyConfig,
    /// Key table merged with this model's quota rules; what every reader sees.
    pub(crate) table: ArcSwap<TenantTable>,
    /// The key table as loaded, quotas unset. Shared by every model's router.
    pub(crate) keys_raw: ArcSwap<TenantTable>,
    /// This model's quota rules, keyed by tenant id.
    pub(crate) quota_rules: ArcSwap<super::quota::QuotaRules>,
    /// Serializes rebuilding `table` from `keys_raw` and `quota_rules`.
    pub(crate) merge_lock: std::sync::Mutex<()>,
    /// Replaced wholesale by the alloc ConfigMap watcher and by write-back.
    pub(crate) alloc: ArcSwap<AllocTable>,
    /// Serializes read-modify-write cycles against the alloc ConfigMap.
    pub(crate) alloc_write_lock: tokio::sync::Mutex<()>,
    /// Derived from healthy worker capacity, or the configured override.
    pub(crate) pool_total: AtomicU32,
    pub(crate) inflight: DashMap<TenantId, Arc<AtomicU32>>,
    pub(crate) over_quota: DashMap<TenantId, Arc<AtomicU64>>,
}

impl TenantManager {
    pub fn new(config: TenancyConfig) -> Arc<Self> {
        let pool_total = AtomicU32::new(config.pool_size_override.unwrap_or(0));
        Arc::new(Self {
            config,
            table: ArcSwap::from_pointee(TenantTable::default()),
            keys_raw: ArcSwap::from_pointee(TenantTable::default()),
            quota_rules: ArcSwap::from_pointee(super::quota::QuotaRules::new()),
            merge_lock: std::sync::Mutex::new(()),
            alloc: ArcSwap::from_pointee(AllocTable::default()),
            alloc_write_lock: tokio::sync::Mutex::new(()),
            pool_total,
            inflight: DashMap::new(),
            over_quota: DashMap::new(),
        })
    }

    pub fn config(&self) -> &TenancyConfig {
        &self.config
    }

    pub fn table(&self) -> arc_swap::Guard<Arc<TenantTable>> {
        self.table.load()
    }

    pub fn alloc(&self) -> arc_swap::Guard<Arc<AllocTable>> {
        self.alloc.load()
    }

    pub fn inflight_counter(&self, tenant: &TenantId) -> Arc<AtomicU32> {
        self.inflight
            .entry(tenant.clone())
            .or_insert_with(|| Arc::new(AtomicU32::new(0)))
            .clone()
    }

    pub fn over_quota_counter(&self, tenant: &TenantId) -> Arc<AtomicU64> {
        self.over_quota
            .entry(tenant.clone())
            .or_insert_with(|| Arc::new(AtomicU64::new(0)))
            .clone()
    }

    /// Concurrency cap in force, or `None` when the tenant is unlimited.
    /// A persisted allocation wins over the spec's `initial`; an unknown or
    /// quota-less tenant is unlimited.
    pub fn allocated(&self, tenant: &TenantId) -> Option<u32> {
        let quota = self.table().tenants.get(tenant)?.quota?;
        Some(
            self.alloc()
                .allocated
                .get(tenant.as_ref())
                .copied()
                .unwrap_or(quota.initial),
        )
    }

    pub fn pool_snapshot(&self) -> PoolSnapshot {
        let total = self.pool_total.load(Ordering::Relaxed);
        // Unlimited tenants draw nothing from the pool: there is no number to draw.
        let allocated: u32 = self
            .table()
            .tenants
            .keys()
            .filter_map(|t| self.allocated(t))
            .sum();
        PoolSnapshot {
            total,
            allocated,
            oversubscribed: allocated > total,
        }
    }
}
