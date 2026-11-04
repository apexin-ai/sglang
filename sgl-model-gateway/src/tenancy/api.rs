//! WP-C: the self-service quota API.
//!
//! Contract:
//!   - Any key may resize its own tenant; no admin role is required.
//!   - The tenant is taken from `TenantCtx` in the request extensions and never
//!     from the request body, otherwise one tenant could resize another.
//!   - A grow is capped by both `Quota::ceiling` and the free pool, so partial
//!     grants are normal and reported in `QuotaDecision`.
//!   - Listing every tenant stays admin-only; a plain key sees only itself.

use std::sync::{atomic::Ordering, Arc};

use axum::{
    extract::{Extension, Json, State},
    http::StatusCode,
    response::{IntoResponse, Response},
    routing::{get, post},
    Router,
};
use serde::Deserialize;
use serde_json::json;

use super::{
    manager::TenantManager,
    store::AllocStore,
    types::{Quota, QuotaDecision, TenantCtx, TenantId, TenantSnapshot},
};

/// Body of a resize request. A tenant field is deliberately absent: the caller
/// is identified by its key, so there is nothing here to spoof.
#[derive(Debug, Deserialize)]
pub struct QuotaRequest {
    pub desired: u32,
}

#[derive(Debug, Deserialize)]
pub struct ReleaseRequest {
    pub amount: u32,
}

/// Resolves a grow or shrink against the tenant bounds and the free pool.
/// `allocated_sum` includes `current`, so the free pool excludes what this
/// tenant already holds.
pub fn decide_resize(
    current: u32,
    desired: u32,
    quota: &Quota,
    pool_total: u32,
    allocated_sum: u32,
) -> QuotaDecision {
    let pool_free = pool_total.saturating_sub(allocated_sum);
    let ceiling_capped = desired.min(quota.ceiling);
    let bounded = ceiling_capped.max(quota.floor);

    if bounded <= current {
        return QuotaDecision {
            granted: bounded,
            requested: desired,
            pool_free,
            reason: if desired > quota.ceiling {
                "capped_by_ceiling"
            } else if desired < quota.floor {
                "capped_by_floor"
            } else {
                "granted"
            },
        };
    }

    let want = bounded - current;
    let delta = want.min(pool_free);
    let granted = current + delta;
    let reason = if delta < want {
        "capped_by_pool"
    } else if desired > quota.ceiling {
        "capped_by_ceiling"
    } else {
        "granted"
    };
    QuotaDecision {
        granted,
        requested: desired,
        pool_free: pool_free - delta,
        reason,
    }
}

/// Releasing never fails: it is a shrink, floored at the tenant minimum.
pub fn decide_release(current: u32, amount: u32, quota: &Quota) -> QuotaDecision {
    let granted = current.saturating_sub(amount).max(quota.floor);
    QuotaDecision {
        granted,
        requested: amount,
        pool_free: 0,
        reason: if granted == current.saturating_sub(amount) {
            "granted"
        } else {
            "capped_by_floor"
        },
    }
}

fn snapshot_of(manager: &TenantManager, tenant: &TenantId) -> Option<TenantSnapshot> {
    let table = manager.table();
    let spec = table.tenants.get(tenant)?;
    Some(TenantSnapshot {
        tenant: tenant.to_string(),
        allocated: manager.allocated(tenant),
        inflight: manager.inflight_counter(tenant).load(Ordering::Relaxed),
        quota: spec.quota,
        over_quota_total: manager.over_quota_counter(tenant).load(Ordering::Relaxed),
    })
}

fn unknown_tenant(tenant: &TenantId) -> Response {
    (
        StatusCode::NOT_FOUND,
        Json(json!({ "error": format!("unknown tenant {}", tenant) })),
    )
        .into_response()
}

async fn get_self(
    State(store): State<Arc<AllocStore>>,
    Extension(ctx): Extension<TenantCtx>,
) -> Response {
    match snapshot_of(store.manager(), &ctx.tenant) {
        Some(snap) => (StatusCode::OK, Json(snap)).into_response(),
        None => unknown_tenant(&ctx.tenant),
    }
}

async fn get_pool(State(store): State<Arc<AllocStore>>) -> Response {
    (StatusCode::OK, Json(store.manager().pool_snapshot())).into_response()
}

async fn list_tenants(
    State(store): State<Arc<AllocStore>>,
    Extension(ctx): Extension<TenantCtx>,
) -> Response {
    if !ctx.role.is_admin() {
        return (
            StatusCode::FORBIDDEN,
            Json(json!({ "error": "listing all tenants requires an admin key" })),
        )
            .into_response();
    }
    let manager = store.manager();
    let tenants: Vec<TenantSnapshot> = manager
        .table()
        .tenants
        .keys()
        .filter_map(|t| snapshot_of(manager, t))
        .collect();
    (StatusCode::OK, Json(tenants)).into_response()
}

/// Applies a decided allocation and persists it.
async fn commit(store: &Arc<AllocStore>, tenant: &TenantId, decision: QuotaDecision) -> Response {
    let key = tenant.to_string();
    let granted = decision.granted;
    if let Err(e) = store
        .mutate(move |table| {
            table.allocated.insert(key, granted);
        })
        .await
    {
        return (
            StatusCode::INTERNAL_SERVER_ERROR,
            Json(json!({ "error": e })),
        )
            .into_response();
    }
    (StatusCode::OK, Json(decision)).into_response()
}

async fn set_quota(
    State(store): State<Arc<AllocStore>>,
    Extension(ctx): Extension<TenantCtx>,
    Json(req): Json<QuotaRequest>,
) -> Response {
    let manager = store.manager();
    let quota = match manager.table().tenants.get(&ctx.tenant) {
        Some(spec) => spec.quota,
        None => return unknown_tenant(&ctx.tenant),
    };
    let pool = manager.pool_snapshot();
    let current = manager.allocated(&ctx.tenant);
    let decision = decide_resize(current, req.desired, &quota, pool.total, pool.allocated);
    commit(&store, &ctx.tenant, decision).await
}

async fn release_quota(
    State(store): State<Arc<AllocStore>>,
    Extension(ctx): Extension<TenantCtx>,
    Json(req): Json<ReleaseRequest>,
) -> Response {
    let manager = store.manager();
    let quota = match manager.table().tenants.get(&ctx.tenant) {
        Some(spec) => spec.quota,
        None => return unknown_tenant(&ctx.tenant),
    };
    let current = manager.allocated(&ctx.tenant);
    let decision = decide_release(current, req.amount, &quota);
    commit(&store, &ctx.tenant, decision).await
}

/// Routes mounted under `/tenants`. Every handler needs `TenantCtx` in the
/// extensions, which the tenancy auth layer installs.
pub fn router(store: Arc<AllocStore>) -> Router {
    Router::new()
        .route("/tenants", get(list_tenants))
        .route("/tenants/pool", get(get_pool))
        .route("/tenants/self", get(get_self))
        .route("/tenants/self/quota", post(set_quota))
        .route("/tenants/self/release", post(release_quota))
        .with_state(store)
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use axum::{body::Body, http::Request};
    use http_body_util::BodyExt;
    use tower::ServiceExt;

    use super::*;
    use crate::tenancy::{
        store::{AllocBackend, BackendError, WritePolicy},
        types::{AllocTable, KeyRole, TenantSpec, TenantTable},
        TenancyConfig,
    };

    struct NullBackend;

    #[async_trait::async_trait]
    impl AllocBackend for NullBackend {
        async fn read(&self) -> Result<(AllocTable, String), BackendError> {
            Ok((AllocTable::default(), "1".to_string()))
        }
        async fn write(&self, _t: &AllocTable, _rv: &str) -> Result<String, BackendError> {
            Ok("2".to_string())
        }
    }

    fn quota(initial: u32, floor: u32, ceiling: u32) -> Quota {
        Quota {
            initial,
            floor,
            ceiling,
        }
    }

    fn store_with_tenants(specs: &[(&str, Quota)], pool: u32) -> Arc<AllocStore> {
        let manager = TenantManager::new(TenancyConfig::default());
        let mut tenants = HashMap::new();
        for (id, q) in specs {
            let tid: TenantId = Arc::from(*id);
            tenants.insert(
                tid.clone(),
                TenantSpec {
                    id: tid,
                    name: id.to_string(),
                    quota: *q,
                },
            );
        }
        manager.install_table(TenantTable {
            tenants,
            by_hash: HashMap::new(),
            resource_version: "1".to_string(),
        });
        manager.pool_total.store(pool, Ordering::Relaxed);
        AllocStore::new(manager, Arc::new(NullBackend), WritePolicy::default())
    }

    fn ctx(tenant: &str, role: KeyRole) -> TenantCtx {
        TenantCtx {
            tenant: Arc::from(tenant),
            key_id: Arc::from("k1"),
            role,
        }
    }

    async fn body_json(resp: Response) -> serde_json::Value {
        let bytes = resp.into_body().collect().await.unwrap().to_bytes();
        serde_json::from_slice(&bytes).unwrap()
    }

    #[test]
    fn grow_is_capped_by_the_free_pool() {
        // 100 total, 94 already handed out: only 6 of the 20 asked for are free.
        let d = decide_resize(10, 30, &quota(10, 0, 64), 100, 94);
        assert_eq!(d.granted, 16);
        assert_eq!(d.requested, 30);
        assert_eq!(d.reason, "capped_by_pool");
    }

    #[test]
    fn grow_is_capped_by_the_ceiling() {
        let d = decide_resize(10, 999, &quota(10, 0, 32), 1000, 10);
        assert_eq!(d.granted, 32);
        assert_eq!(d.reason, "capped_by_ceiling");
    }

    #[test]
    fn full_grant_reports_granted() {
        let d = decide_resize(10, 20, &quota(10, 0, 64), 100, 10);
        assert_eq!(d.granted, 20);
        assert_eq!(d.reason, "granted");
    }

    #[test]
    fn shrink_through_set_quota_needs_no_pool_room() {
        let d = decide_resize(40, 8, &quota(10, 4, 64), 40, 40);
        assert_eq!(d.granted, 8);
        assert_eq!(d.reason, "granted");
    }

    #[test]
    fn release_is_floored() {
        let d = decide_release(10, 9, &quota(10, 4, 64));
        assert_eq!(d.granted, 4);
        assert_eq!(d.reason, "capped_by_floor");

        let d = decide_release(10, 2, &quota(10, 4, 64));
        assert_eq!(d.granted, 8);
        assert_eq!(d.reason, "granted");
    }

    #[tokio::test]
    async fn quota_applies_to_the_calling_key_not_the_body() {
        let store = store_with_tenants(
            &[("team-a", quota(4, 0, 64)), ("victim", quota(4, 0, 64))],
            100,
        );
        let app = router(Arc::clone(&store));

        let req = Request::builder()
            .method("POST")
            .uri("/tenants/self/quota")
            .header("content-type", "application/json")
            .extension(ctx("team-a", KeyRole::User))
            // A tenant field in the body must have no effect.
            .body(Body::from(r#"{"desired": 20, "tenant": "victim"}"#))
            .unwrap();
        let resp = app.oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
        assert_eq!(body_json(resp).await["granted"], 20);

        let manager = store.manager();
        assert_eq!(manager.allocated(&Arc::from("team-a")), 20);
        assert_eq!(manager.allocated(&Arc::from("victim")), 4);
    }

    #[tokio::test]
    async fn self_service_resize_works_without_an_admin_key() {
        let store = store_with_tenants(&[("team-a", quota(4, 0, 64))], 100);
        let app = router(Arc::clone(&store));

        let req = Request::builder()
            .method("POST")
            .uri("/tenants/self/quota")
            .header("content-type", "application/json")
            .extension(ctx("team-a", KeyRole::User))
            .body(Body::from(r#"{"desired": 16}"#))
            .unwrap();
        assert_eq!(app.oneshot(req).await.unwrap().status(), StatusCode::OK);
        assert_eq!(store.manager().allocated(&Arc::from("team-a")), 16);
    }

    #[tokio::test]
    async fn listing_every_tenant_requires_admin() {
        let store = store_with_tenants(&[("team-a", quota(4, 0, 64))], 100);

        let req = Request::builder()
            .uri("/tenants")
            .extension(ctx("team-a", KeyRole::User))
            .body(Body::empty())
            .unwrap();
        let resp = router(Arc::clone(&store)).oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::FORBIDDEN);

        let req = Request::builder()
            .uri("/tenants")
            .extension(ctx("team-a", KeyRole::Admin))
            .body(Body::empty())
            .unwrap();
        let resp = router(store).oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::OK);
    }

    #[tokio::test]
    async fn self_view_reports_allocation_and_inflight() {
        let store = store_with_tenants(&[("team-a", quota(6, 0, 64))], 100);
        store
            .manager()
            .inflight_counter(&Arc::from("team-a"))
            .store(3, Ordering::Relaxed);

        let req = Request::builder()
            .uri("/tenants/self")
            .extension(ctx("team-a", KeyRole::User))
            .body(Body::empty())
            .unwrap();
        let resp = router(store).oneshot(req).await.unwrap();
        let body = body_json(resp).await;
        assert_eq!(body["allocated"], 6);
        assert_eq!(body["inflight"], 3);
    }

    #[tokio::test]
    async fn unknown_tenant_is_rejected() {
        let store = store_with_tenants(&[("team-a", quota(4, 0, 64))], 100);
        let req = Request::builder()
            .uri("/tenants/self")
            .extension(ctx("ghost", KeyRole::User))
            .body(Body::empty())
            .unwrap();
        let resp = router(store).oneshot(req).await.unwrap();
        assert_eq!(resp.status(), StatusCode::NOT_FOUND);
    }
}
