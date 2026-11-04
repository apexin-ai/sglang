//! Rejects data-plane requests that carry no routing key.
//!
//! Without it the `manual` and `consistent_hashing` policies fall back to a
//! random worker, so a caller that forgets the header silently loses session
//! affinity instead of failing. Rejecting here rather than in the policy keeps
//! the error honest: a policy that selects no worker surfaces as "no available
//! workers", which reads as an outage.

use axum::{
    body::Body,
    extract::{Request, State},
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
    Json,
};
use serde_json::json;

use crate::routers::header_utils::extract_routing_key;

/// Paths exempt from the requirement; they carry no session to pin.
fn is_exempt(path: &str) -> bool {
    path == "/v1/models" || path == "/v1/tokenize" || path == "/v1/detokenize"
}

pub async fn require_routing_key_middleware(
    State(enabled): State<bool>,
    request: Request<Body>,
    next: Next,
) -> Response {
    if !enabled || is_exempt(request.uri().path()) {
        return next.run(request).await;
    }
    if extract_routing_key(Some(request.headers())).is_some() {
        return next.run(request).await;
    }
    (
        StatusCode::BAD_REQUEST,
        Json(json!({
            "error": {
                "message": "missing required header X-SMG-Routing-Key",
                "type": "invalid_request_error",
                "param": "X-SMG-Routing-Key",
                "code": "missing_routing_key",
            }
        })),
    )
        .into_response()
}

#[cfg(test)]
mod tests {
    use axum::{body::Body, http::Request, routing::post, Router};
    use tower::ServiceExt;

    use super::*;

    fn app(enabled: bool) -> Router {
        Router::new()
            .route("/v1/chat/completions", post(|| async { "ok" }))
            .route("/v1/models", axum::routing::get(|| async { "ok" }))
            .route_layer(axum::middleware::from_fn_with_state(
                enabled,
                require_routing_key_middleware,
            ))
    }

    async fn status(app: Router, path: &str, key: Option<&str>) -> StatusCode {
        let mut builder = Request::builder().uri(path);
        builder = match path {
            "/v1/models" => builder.method("GET"),
            _ => builder.method("POST"),
        };
        if let Some(key) = key {
            builder = builder.header("x-smg-routing-key", key);
        }
        app.oneshot(builder.body(Body::empty()).unwrap())
            .await
            .unwrap()
            .status()
    }

    #[tokio::test]
    async fn rejects_only_when_enabled_and_the_header_is_absent() {
        assert_eq!(
            status(app(true), "/v1/chat/completions", None).await,
            StatusCode::BAD_REQUEST
        );
        assert_eq!(
            status(app(true), "/v1/chat/completions", Some("sess-1")).await,
            StatusCode::OK
        );
        assert_eq!(
            status(app(false), "/v1/chat/completions", None).await,
            StatusCode::OK
        );
    }

    #[tokio::test]
    async fn exempt_paths_pass_without_a_key() {
        assert_eq!(status(app(true), "/v1/models", None).await, StatusCode::OK);
    }
}
