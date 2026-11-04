//! Per-tenant metrics. Label cardinality is bounded by the tenant count, which
//! is operator-controlled and small.

use metrics::{counter, gauge, histogram};

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

pub fn record_tenant_request(tenant: &str, key_id: &str) {
    counter!("smg_tenant_requests_total", "tenant" => tenant.to_string(), "key" => key_id.to_string())
        .increment(1);
}

/// Token counts as the upstream reported them. A streaming request only reports
/// them when the client asked for `stream_options.include_usage`, so these
/// counters undercount rather than estimate when it did not.
pub fn record_tenant_tokens(tenant: &str, key_id: &str, prompt: u64, completion: u64) {
    counter!("smg_tenant_prompt_tokens_total", "tenant" => tenant.to_string(), "key" => key_id.to_string())
        .increment(prompt);
    counter!("smg_tenant_completion_tokens_total", "tenant" => tenant.to_string(), "key" => key_id.to_string())
        .increment(completion);
}

/// Wall time from arrival at the gateway to the end of the response body, so a
/// streamed completion is measured over the whole stream.
pub fn record_tenant_duration(tenant: &str, seconds: f64) {
    histogram!("smg_tenant_request_duration_seconds", "tenant" => tenant.to_string())
        .record(seconds);
}

/// Time to the first body frame; for a stream that is the first token.
pub fn record_tenant_ttft(tenant: &str, seconds: f64) {
    histogram!("smg_tenant_ttft_seconds", "tenant" => tenant.to_string()).record(seconds);
}

/// Requests whose usage block never arrived, so their tokens are missing from
/// the counters above.
pub fn record_tenant_usage_missing(tenant: &str) {
    counter!("smg_tenant_usage_missing_total", "tenant" => tenant.to_string()).increment(1);
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
