//! WP-B: per-tenant in-flight accounting.
//!
//! Contract:
//!   - Observe-only for now: a request over its tenant's allocation is still
//!     served, it only bumps `over_quota_total` and the matching metric.
//!   - The decrement must happen when the response body ends, not when the
//!     handler returns, or every streamed request leaks a count. Mirror
//!     `TokenGuardBody` (src/middleware.rs) and release on `Drop` so an aborted
//!     stream is also released.

use std::sync::{
    atomic::{AtomicU32, AtomicU64, Ordering},
    Arc,
};

/// Holds one unit of a tenant's in-flight count until dropped.
pub struct InflightGuard {
    counter: Arc<AtomicU32>,
}

impl InflightGuard {
    pub fn new(counter: Arc<AtomicU32>, allocated: u32, over_quota: &Arc<AtomicU64>) -> Self {
        let now = counter.fetch_add(1, Ordering::Relaxed) + 1;
        if now > allocated {
            over_quota.fetch_add(1, Ordering::Relaxed);
        }
        Self { counter }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        self.counter.fetch_sub(1, Ordering::Relaxed);
    }
}

// TODO(WP-B): body wrapper holding an InflightGuard until the stream ends, the
// middleware that installs it, and the per-tenant gauges.
