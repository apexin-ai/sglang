//! WP-B: sizing the shared concurrency pool from worker capacity.
//!
//! Contract:
//!   - `pool = sum(healthy workers' max_running_requests) * pool_headroom`,
//!     unless `pool_size_override` is set, in which case workers are ignored.
//!   - Under `--dp-aware` one server appears as many `url@rank` workers; summing
//!     the expanded list multiplies the pool by the DP degree. Whether sglang's
//!     `max_running_requests` is per server or per attention-DP rank is NOT yet
//!     established -- settle that before trusting the derived value, and keep
//!     the override as the safe default until then.
//!   - Recompute on worker registration, removal and health transitions.
//!   - Shrink policy: allocations are untouched while `allocated <= total`.
//!     Once `allocated > total`, every tenant is scaled by `total / allocated`,
//!     floored at `Quota::floor`.

use super::manager::TenantManager;

impl TenantManager {
    /// Recomputes the pool and, only when oversubscribed, rescales allocations.
    pub fn recompute_pool(&self, _worker_capacity: u32) {
        // TODO(WP-B): set pool_total, detect oversubscription, scale proportionally,
        // persist the rescaled allocations through store.rs, update metrics.
        unimplemented!("WP-B")
    }
}
