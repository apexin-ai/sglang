//! WP-C: the self-service quota API.
//!
//! Contract:
//!   - Any key may resize its own tenant; no admin role is required.
//!   - The tenant is taken from `TenantCtx` in the request extensions and never
//!     from the request body, otherwise one tenant could resize another.
//!   - A grow is capped by both `Quota::ceiling` and the free pool, so partial
//!     grants are normal and reported in `QuotaDecision`.
//!   - Listing every tenant stays admin-only; a plain key sees only itself.

use std::sync::Arc;

use axum::Router;

use super::manager::TenantManager;

/// Routes mounted under `/tenants`.
pub fn router(_manager: Arc<TenantManager>) -> Router {
    // TODO(WP-C): GET /tenants/self, POST /tenants/self/quota, POST /tenants/self/release,
    // GET /tenants (admin), GET /tenants/pool.
    Router::new()
}
