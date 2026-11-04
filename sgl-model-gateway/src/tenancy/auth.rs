//! WP-A: resolving a request's bearer token to a tenant.
//!
//! Contract:
//!   - Runs only after the community `--api-key` / `--control-plane-api-keys`
//!     checks have already admitted or rejected the request, so those paths
//!     keep working unchanged and stay usable as an emergency credential.
//!   - On a hit, insert `TenantCtx` into the request extensions; `gate.rs` and
//!     `api.rs` both read it from there.
//!   - Control-plane routes additionally require `TenantCtx::role` to be admin.

use super::{
    manager::TenantManager,
    types::{hash_key, TenantCtx},
};

impl TenantManager {
    /// Constant-time lookup of a presented bearer token.
    pub fn resolve_key(&self, presented: &str) -> Option<TenantCtx> {
        let table = self.table();
        let key = table.by_hash.get(&hash_key(presented))?;
        if key.disabled {
            return None;
        }
        Some(TenantCtx {
            tenant: key.tenant.clone(),
            key_id: key.id.clone(),
            role: key.role,
        })
    }
}

// TODO(WP-A): axum middleware that extracts the bearer token, calls resolve_key,
// inserts TenantCtx, and records an auth-outcome metric.
