//! WP-B: per-tenant in-flight accounting.
//!
//! Observe-only: a request past its tenant's allocation is still served, it only
//! bumps the over-quota counter. The release has to happen when the response body
//! ends rather than when the handler returns -- a streamed completion returns from
//! the handler immediately -- so the guard rides the body and releases on `Drop`,
//! which also covers a client that disconnects mid-stream.

use std::{
    pin::Pin,
    sync::{
        atomic::{AtomicU32, AtomicU64, Ordering},
        Arc,
    },
    task::{Context, Poll},
};

use axum::{body::Body, extract::State, middleware::Next, response::Response};
use bytes::Bytes;
use http_body::Frame;

use super::{manager::TenantManager, metrics, types::TenantCtx};

/// Holds one unit of a tenant's in-flight count until dropped.
pub struct InflightGuard {
    counter: Arc<AtomicU32>,
    tenant: Arc<str>,
}

impl InflightGuard {
    pub fn new(
        tenant: Arc<str>,
        counter: Arc<AtomicU32>,
        allocated: u32,
        over_quota: &Arc<AtomicU64>,
    ) -> Self {
        let now = counter.fetch_add(1, Ordering::Relaxed) + 1;
        metrics::set_tenant_inflight(&tenant, now);
        metrics::record_tenant_request(&tenant);
        if now > allocated {
            over_quota.fetch_add(1, Ordering::Relaxed);
            metrics::record_tenant_over_quota(&tenant);
        }
        Self { counter, tenant }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        let now = self
            .counter
            .fetch_sub(1, Ordering::Relaxed)
            .saturating_sub(1);
        metrics::set_tenant_inflight(&self.tenant, now);
    }
}

/// Body wrapper that keeps the tenant's in-flight slot until the stream ends.
pub struct TenantGuardBody {
    inner: Body,
    _guard: InflightGuard,
}

impl TenantGuardBody {
    pub fn new(inner: Body, guard: InflightGuard) -> Self {
        Self {
            inner,
            _guard: guard,
        }
    }
}

impl http_body::Body for TenantGuardBody {
    type Data = Bytes;
    type Error = axum::Error;

    fn poll_frame(
        self: Pin<&mut Self>,
        cx: &mut Context<'_>,
    ) -> Poll<Option<Result<Frame<Self::Data>, Self::Error>>> {
        let this = self.get_mut();
        Pin::new(&mut this.inner).poll_frame(cx)
    }

    fn is_end_stream(&self) -> bool {
        self.inner.is_end_stream()
    }

    fn size_hint(&self) -> http_body::SizeHint {
        self.inner.size_hint()
    }
}

/// Counts the request against its tenant. A request with no `TenantCtx` is
/// passed through untouched: the community key paths admit traffic this table
/// never saw, and accounting it to nobody would be worse than not counting it.
pub async fn tenant_inflight_middleware(
    State(manager): State<Arc<TenantManager>>,
    request: axum::extract::Request,
    next: Next,
) -> Response {
    let Some(ctx) = request.extensions().get::<TenantCtx>().cloned() else {
        return next.run(request).await;
    };
    let guard = manager.inflight_guard(&ctx);
    let response = next.run(request).await;
    let (parts, body) = response.into_parts();
    Response::from_parts(parts, Body::new(TenantGuardBody::new(body, guard)))
}

impl TenantManager {
    pub fn inflight_guard(&self, ctx: &TenantCtx) -> InflightGuard {
        let allocated = self.allocated(&ctx.tenant);
        metrics::set_tenant_allocated(&ctx.tenant, allocated);
        InflightGuard::new(
            ctx.tenant.clone(),
            self.inflight_counter(&ctx.tenant),
            allocated,
            &self.over_quota_counter(&ctx.tenant),
        )
    }
}

#[cfg(test)]
mod tests {
    use http_body_util::BodyExt;

    use super::*;
    use crate::tenancy::{
        types::{KeyRole, Quota, TenantId, TenantSpec, TenantTable},
        TenancyConfig,
    };

    fn manager_with_tenant(allocated: u32) -> (Arc<TenantManager>, TenantCtx) {
        let manager = TenantManager::new(TenancyConfig::default());
        let id: TenantId = Arc::from("team-a");
        let mut table = TenantTable::default();
        table.tenants.insert(
            id.clone(),
            TenantSpec {
                id: id.clone(),
                name: "team-a".to_string(),
                quota: Quota {
                    initial: allocated,
                    floor: 0,
                    ceiling: 64,
                },
            },
        );
        manager.install_table(table);
        let ctx = TenantCtx {
            tenant: id,
            key_id: Arc::from("k1"),
            role: KeyRole::User,
        };
        (manager, ctx)
    }

    #[test]
    fn guard_counts_up_and_releases_on_drop() {
        let (manager, ctx) = manager_with_tenant(4);
        let counter = manager.inflight_counter(&ctx.tenant);
        {
            let _a = manager.inflight_guard(&ctx);
            let _b = manager.inflight_guard(&ctx);
            assert_eq!(counter.load(Ordering::Relaxed), 2);
        }
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[test]
    fn requests_past_the_allocation_are_counted_not_refused() {
        let (manager, ctx) = manager_with_tenant(1);
        let over = manager.over_quota_counter(&ctx.tenant);
        let _a = manager.inflight_guard(&ctx);
        assert_eq!(over.load(Ordering::Relaxed), 0);
        let _b = manager.inflight_guard(&ctx);
        let _c = manager.inflight_guard(&ctx);
        assert_eq!(over.load(Ordering::Relaxed), 2);
        assert_eq!(
            manager
                .inflight_counter(&ctx.tenant)
                .load(Ordering::Relaxed),
            3
        );
    }

    #[tokio::test]
    async fn streaming_body_holds_the_slot_until_it_ends() {
        let (manager, ctx) = manager_with_tenant(4);
        let counter = manager.inflight_counter(&ctx.tenant);
        let guard = manager.inflight_guard(&ctx);
        let body = Body::new(TenantGuardBody::new(Body::from("hello"), guard));
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        let collected = body.collect().await.unwrap().to_bytes();
        assert_eq!(&collected[..], b"hello");
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }

    #[tokio::test]
    async fn dropping_a_stream_early_releases_the_slot() {
        let (manager, ctx) = manager_with_tenant(4);
        let counter = manager.inflight_counter(&ctx.tenant);
        let guard = manager.inflight_guard(&ctx);
        let body = TenantGuardBody::new(Body::from("abc"), guard);
        assert_eq!(counter.load(Ordering::Relaxed), 1);
        drop(body);
        assert_eq!(counter.load(Ordering::Relaxed), 0);
    }
}
