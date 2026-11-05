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
    /// Replaced wholesale by the key ConfigMap watcher; shared by every model.
    pub(crate) table: ArcSwap<TenantTable>,
    /// This model's quota rules, keyed by tenant id; the only source of quota.
    pub(crate) quota_rules: ArcSwap<super::quota::QuotaRules>,
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
            quota_rules: ArcSwap::from_pointee(super::quota::QuotaRules::new()),
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

    /// This model's quota for a known tenant; `None` means unlimited here.
    pub fn quota_of(&self, tenant: &TenantId) -> Option<super::types::Quota> {
        if !self.table().tenants.contains_key(tenant) {
            return None;
        }
        self.quota_rules.load().get(tenant.as_ref()).copied()
    }

    /// Concurrency cap in force, or `None` when the tenant is unlimited on this
    /// model. A persisted allocation wins over the rule's `initial`, clamped to
    /// the rule's current bounds: the rule is edited by hand and may since have
    /// moved below an allocation written under an older one.
    pub fn allocated(&self, tenant: &TenantId) -> Option<u32> {
        let quota = self.quota_of(tenant)?;
        let persisted = self.alloc().allocated.get(tenant.as_ref()).copied();
        Some(
            persisted
                .unwrap_or(quota.initial)
                .clamp(quota.floor, quota.ceiling),
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
