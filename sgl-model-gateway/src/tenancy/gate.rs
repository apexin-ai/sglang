//! WP-B: per-tenant in-flight accounting.
//!
//! A tenant with no quota is unlimited: nothing is compared and nothing is
//! counted against it. For one that has a cap this stays observe-only -- a
//! request past the allocation is still served and only bumps the counter. The release has to happen when the response body
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
    time::Instant,
};

use axum::{body::Body, extract::State, middleware::Next, response::Response};
use bytes::Bytes;
use http_body::Frame;

use super::{manager::TenantManager, metrics, types::TenantCtx};

/// Holds one unit of a tenant's in-flight count until dropped, and carries what
/// the response observer needs to attribute tokens and latency.
pub struct InflightGuard {
    counter: Arc<AtomicU32>,
    tenant: Arc<str>,
    key_id: Arc<str>,
    started: Instant,
}

impl InflightGuard {
    /// `allocated` is `None` for an unlimited tenant, which is then never
    /// counted as over quota.
    pub fn new(
        tenant: Arc<str>,
        key_id: Arc<str>,
        counter: Arc<AtomicU32>,
        allocated: Option<u32>,
        over_quota: &Arc<AtomicU64>,
    ) -> Self {
        let now = counter.fetch_add(1, Ordering::Relaxed) + 1;
        metrics::set_tenant_inflight(&tenant, now);
        metrics::record_tenant_request(&tenant, &key_id);
        if allocated.is_some_and(|limit| now > limit) {
            over_quota.fetch_add(1, Ordering::Relaxed);
            metrics::record_tenant_over_quota(&tenant);
        }
        Self {
            counter,
            tenant,
            key_id,
            started: Instant::now(),
        }
    }
}

impl Drop for InflightGuard {
    fn drop(&mut self) {
        let now = self
            .counter
            .fetch_sub(1, Ordering::Relaxed)
            .saturating_sub(1);
        metrics::set_tenant_inflight(&self.tenant, now);
        metrics::record_tenant_duration(&self.tenant, self.started.elapsed().as_secs_f64());
    }
}

/// Upstream-reported token counts, pulled out of the response.
///
/// Both shapes end in a `usage` object: a non-streaming reply carries one in its
/// single JSON body, and a stream carries one in its final chunk when the client
/// set `stream_options.include_usage`. Frames are scanned for the key rather
/// than parsed, so a long stream is never buffered or deserialized.
#[derive(Default)]
struct UsageScanner {
    /// The newest frame that mentioned usage; the last one wins, since a stream
    /// sends it in its final chunk.
    candidate: Option<Vec<u8>>,
}

/// Arbitrary: a usage chunk is a few hundred bytes, and a non-streaming reply
/// holds its usage near the end. Anything larger is not worth copying.
const USAGE_SCAN_LIMIT: usize = 16 * 1024;

impl UsageScanner {
    fn observe(&mut self, frame: &[u8]) {
        if !contains(frame, USAGE_KEY) {
            return;
        }
        let tail = &frame[frame.len().saturating_sub(USAGE_SCAN_LIMIT)..];
        self.candidate = Some(tail.to_vec());
    }

    fn finish(&self) -> Option<(u64, u64)> {
        let raw = self.candidate.as_ref()?;
        // The last usage block wins: a stream's final chunk carries the
        // cumulative counts.
        let start = rfind(raw, USAGE_KEY)?;
        let block = &raw[start..];
        Some((field(block, PROMPT_KEY)?, field(block, COMPLETION_KEY)?))
    }
}

const USAGE_KEY: &[u8] = b"\"usage\"";
const PROMPT_KEY: &[u8] = b"\"prompt_tokens\"";
const COMPLETION_KEY: &[u8] = b"\"completion_tokens\"";

fn contains(haystack: &[u8], needle: &[u8]) -> bool {
    haystack.windows(needle.len()).any(|w| w == needle)
}

fn rfind(haystack: &[u8], needle: &[u8]) -> Option<usize> {
    haystack.windows(needle.len()).rposition(|w| w == needle)
}

/// Reads the integer that follows `name` in a JSON object.
fn field(block: &[u8], name: &[u8]) -> Option<u64> {
    let at = block.windows(name.len()).position(|w| w == name)? + name.len();
    let digits: Vec<u8> = block[at..]
        .iter()
        .skip_while(|b| !b.is_ascii_digit())
        .take_while(|b| b.is_ascii_digit())
        .copied()
        .collect();
    std::str::from_utf8(&digits).ok()?.parse().ok()
}

/// Body wrapper that keeps the tenant's in-flight slot until the stream ends.
pub struct TenantGuardBody {
    inner: Body,
    guard: InflightGuard,
    usage: UsageScanner,
    first_frame_seen: bool,
    reported: bool,
}

impl TenantGuardBody {
    pub fn new(inner: Body, guard: InflightGuard) -> Self {
        Self {
            inner,
            guard,
            usage: UsageScanner::default(),
            first_frame_seen: false,
            reported: false,
        }
    }

    /// Reports once, whether the stream ended or the client went away.
    fn report_usage(&mut self) {
        if self.reported {
            return;
        }
        self.reported = true;
        match self.usage.finish() {
            Some((prompt, completion)) => metrics::record_tenant_tokens(
                &self.guard.tenant,
                &self.guard.key_id,
                prompt,
                completion,
            ),
            None => metrics::record_tenant_usage_missing(&self.guard.tenant),
        }
    }
}

impl Drop for TenantGuardBody {
    fn drop(&mut self) {
        self.report_usage();
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
        let polled = Pin::new(&mut this.inner).poll_frame(cx);
        match &polled {
            Poll::Ready(Some(Ok(frame))) => {
                if let Some(data) = frame.data_ref() {
                    if !this.first_frame_seen {
                        this.first_frame_seen = true;
                        metrics::record_tenant_ttft(
                            &this.guard.tenant,
                            this.guard.started.elapsed().as_secs_f64(),
                        );
                    }
                    this.usage.observe(data);
                }
            }
            Poll::Ready(None) => this.report_usage(),
            _ => {}
        }
        polled
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
        if let Some(limit) = allocated {
            metrics::set_tenant_allocated(&ctx.tenant, limit);
        }
        InflightGuard::new(
            ctx.tenant.clone(),
            ctx.key_id.clone(),
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
                quota: Some(Quota {
                    initial: allocated,
                    floor: 0,
                    ceiling: 64,
                }),
            },
        );
        manager.install_table_with_quotas(table);
        let ctx = TenantCtx {
            tenant: id,
            key_id: Arc::from("k1"),
            role: KeyRole::User,
        };
        (manager, ctx)
    }

    #[test]
    fn scans_usage_from_a_non_streaming_reply() {
        let mut u = UsageScanner::default();
        u.observe(br#"{"id":"x","choices":[],"usage":{"prompt_tokens":12,"completion_tokens":34,"total_tokens":46}}"#);
        assert_eq!(u.finish(), Some((12, 34)));
    }

    #[test]
    fn scans_usage_from_the_final_stream_chunk() {
        let mut u = UsageScanner::default();
        u.observe(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}],\"usage\":null}\n\n");
        u.observe(b"data: {\"choices\":[],\"usage\":{\"prompt_tokens\":7,\"completion_tokens\":5}}\n\ndata: [DONE]\n\n");
        assert_eq!(u.finish(), Some((7, 5)));
    }

    #[test]
    fn reports_nothing_when_the_client_never_asked_for_usage() {
        let mut u = UsageScanner::default();
        u.observe(b"data: {\"choices\":[{\"delta\":{\"content\":\"hi\"}}]}\n\n");
        assert_eq!(u.finish(), None);
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
    fn an_unlimited_tenant_is_never_over_quota() {
        let manager = TenantManager::new(TenancyConfig::default());
        let id: TenantId = Arc::from("team-free");
        let mut table = TenantTable::default();
        table.tenants.insert(
            id.clone(),
            TenantSpec {
                id: id.clone(),
                name: "team-free".to_string(),
                quota: None,
            },
        );
        manager.install_table_with_quotas(table);
        let ctx = TenantCtx {
            tenant: id,
            key_id: Arc::from("k1"),
            role: KeyRole::User,
        };
        let guards: Vec<_> = (0..50).map(|_| manager.inflight_guard(&ctx)).collect();
        assert_eq!(manager.allocated(&ctx.tenant), None);
        assert_eq!(
            manager
                .over_quota_counter(&ctx.tenant)
                .load(Ordering::Relaxed),
            0
        );
        assert_eq!(
            manager
                .inflight_counter(&ctx.tenant)
                .load(Ordering::Relaxed),
            50
        );
        drop(guards);
        assert_eq!(
            manager
                .inflight_counter(&ctx.tenant)
                .load(Ordering::Relaxed),
            0
        );
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
