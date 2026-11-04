//! WP-A: resolving a request's bearer token to a tenant.
//!
//! Contract:
//!   - Runs only after the community `--api-key` / `--control-plane-api-keys`
//!     checks have already admitted or rejected the request, so those paths
//!     keep working unchanged and stay usable as an emergency credential.
//!   - On a hit, insert `TenantCtx` into the request extensions; `gate.rs` and
//!     `api.rs` both read it from there.
//!   - Control-plane routes additionally require `TenantCtx::role` to be admin.

use std::sync::Arc;

use axum::{
    body::Body,
    extract::{Request, State},
    http::{header, StatusCode},
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use metrics::counter;
use serde_json::json;

use super::{
    manager::TenantManager,
    types::{hash_key, KeyRole, TenantCtx},
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

#[derive(Clone)]
pub struct TenancyAuthState {
    pub manager: Arc<TenantManager>,
    /// Reject a request whose token is not in the table. Required when no
    /// community `--api-key` is set, because that layer then admits everything
    /// and an unidentified request would bypass accounting entirely.
    pub enforce: bool,
}

fn bearer(request: &Request) -> Option<&str> {
    let value = request
        .headers()
        .get(header::AUTHORIZATION)?
        .to_str()
        .ok()?;
    value
        .strip_prefix("Bearer ")
        .or_else(|| value.strip_prefix("bearer "))
}

fn unauthorized(message: &str) -> Response {
    (
        StatusCode::UNAUTHORIZED,
        Json(json!({"error": {"message": message, "type": "invalid_request_error"}})),
    )
        .into_response()
}

/// Identifies the tenant behind a data-plane request.
pub async fn tenancy_auth_middleware(
    State(state): State<TenancyAuthState>,
    mut request: Request<Body>,
    next: Next,
) -> Response {
    match bearer(&request).and_then(|token| state.manager.resolve_key(token)) {
        Some(ctx) => {
            counter!("smg_tenancy_auth_total", "result" => "identified").increment(1);
            request.extensions_mut().insert(ctx);
            next.run(request).await
        }
        None if state.enforce => {
            counter!("smg_tenancy_auth_total", "result" => "rejected").increment(1);
            unauthorized("invalid or missing API key")
        }
        None => {
            // Admitted by the community key path; served without a tenant, so
            // it is counted against no allocation.
            counter!("smg_tenancy_auth_total", "result" => "anonymous").increment(1);
            next.run(request).await
        }
    }
}

/// Gates control-plane routes on an admin-role tenancy key. Requests the
/// community control-plane auth already admitted carry no `TenantCtx` and are
/// passed through, keeping the emergency credential usable.
pub async fn tenancy_admin_middleware(
    State(state): State<TenancyAuthState>,
    request: Request<Body>,
    next: Next,
) -> Response {
    match request.extensions().get::<TenantCtx>() {
        Some(ctx) if ctx.role == KeyRole::Admin => next.run(request).await,
        Some(_) => (
            StatusCode::FORBIDDEN,
            Json(json!({"error": {"message": "admin role required", "type": "permission_error"}})),
        )
            .into_response(),
        None if state.enforce => unauthorized("invalid or missing API key"),
        None => next.run(request).await,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tenancy::{source::parse_key_table, TenancyConfig};

    const DOC: &str = r#"
tenants:
  - id: team-a
    keys:
      - {id: k1, secret: "live", role: user}
      - {id: k2, secret: "revoked", role: admin, disabled: true}
"#;

    fn manager() -> Arc<TenantManager> {
        let manager = TenantManager::new(TenancyConfig::default());
        manager.install_table(parse_key_table(DOC, "1".to_string()).unwrap());
        manager
    }

    #[test]
    fn resolves_a_live_key() {
        let ctx = manager().resolve_key("live").unwrap();
        assert_eq!(ctx.tenant.as_ref(), "team-a");
        assert_eq!(ctx.key_id.as_ref(), "k1");
        assert_eq!(ctx.role, KeyRole::User);
    }

    #[test]
    fn rejects_unknown_and_disabled_keys() {
        let manager = manager();
        assert!(manager.resolve_key("nope").is_none());
        assert!(manager.resolve_key("revoked").is_none());
    }
}
