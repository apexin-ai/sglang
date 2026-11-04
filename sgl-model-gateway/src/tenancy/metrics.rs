//! Per-tenant metrics. Label cardinality is bounded by the tenant count, which
//! is operator-controlled and small.

use metrics::{counter, gauge};

use super::types::PoolSnapshot;

pub fn set_tenant_inflight(tenant: &str, inflight: u32) {
    gauge!("smg_tenant_inflight", "tenant" => tenant.to_string()).set(inflight as f64);
}

pub fn set_tenant_allocated(tenant: &str, allocated: u32) {
    gauge!("smg_tenant_concurrency_allocated", "tenant" => tenant.to_string())
        .set(allocated as f64);
}

/// A request admitted while the tenant was already at or above its allocation.
/// Observe-only: the request is still served.
pub fn record_tenant_over_quota(tenant: &str) {
    counter!("smg_tenant_over_quota_total", "tenant" => tenant.to_string()).increment(1);
}

pub fn record_tenant_request(tenant: &str) {
    counter!("smg_tenant_requests_total", "tenant" => tenant.to_string()).increment(1);
}

pub fn set_pool(snapshot: &PoolSnapshot) {
    gauge!("smg_tenant_pool_total").set(snapshot.total as f64);
    gauge!("smg_tenant_pool_allocated").set(snapshot.allocated as f64);
    gauge!("smg_tenant_pool_oversubscribed").set(u8::from(snapshot.oversubscribed) as f64);
}

/// Capacity reported by the workers before `pool_headroom` is applied; 0 when
/// no worker reported one, which is what makes a derived pool fall back.
pub fn set_worker_capacity(capacity: u32) {
    gauge!("smg_tenant_worker_capacity").set(capacity as f64);
}
