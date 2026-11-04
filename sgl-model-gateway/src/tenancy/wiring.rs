//! Startup and request-chain wiring, kept here so `server.rs` and
//! `app_context.rs` carry one line each instead of the whole feature.

use std::sync::Arc;

use axum::Router;
use kube::Client;
use tracing::{error, info, warn};

use super::{
    api,
    auth::{tenancy_auth_middleware, TenancyAuthState},
    gate::tenant_inflight_middleware,
    manager::TenantManager,
    pool::spawn_pool_sampler,
    source::{ConfigMapKeySource, KeySource},
    store::{AllocStore, KubeAllocBackend, WritePolicy},
    TenancyConfig,
};
use crate::{config::types::RouterConfig, core::WorkerRegistry};

/// Everything the request chain needs once tenancy is on.
pub struct TenancyRuntime {
    pub manager: Arc<TenantManager>,
    pub store: Arc<AllocStore>,
    pub enforce: bool,
}

/// kube's rustls client needs a process-wide `CryptoProvider`, and the crate
/// pulls in more than one backend, so it cannot pick one itself. Idempotent:
/// installing over an existing provider fails and that is fine.
/// `service_discovery` has the same requirement on its own `Client::try_default`.
fn ensure_crypto_provider() {
    let _ = rustls::crypto::ring::default_provider().install_default();
}

/// Builds the runtime when a key ConfigMap is configured, otherwise `None`.
///
/// `enforce` is derived rather than configured: with no community data-plane
/// key the upstream auth layer admits everything, so leaving unidentified
/// traffic unaccounted would defeat the feature.
pub async fn build(config: &RouterConfig) -> Option<TenancyRuntime> {
    let keys_configmap = config.tenancy_keys_configmap.clone()?;
    let namespace = config
        .tenancy_namespace
        .clone()
        .unwrap_or_else(|| "default".to_string());

    let tenancy_config = TenancyConfig {
        namespace: namespace.clone(),
        keys_configmap: keys_configmap.clone(),
        alloc_configmap: config.tenancy_alloc_configmap.clone(),
        pool_headroom: config.tenancy_pool_headroom,
        pool_size_override: config.tenancy_pool_size,
        resync_secs: config.tenancy_resync_secs,
        enforce: config.api_key.is_none(),
    };
    let enforce = tenancy_config.enforce;

    ensure_crypto_provider();
    let client = match Client::try_default().await {
        Ok(client) => client,
        Err(e) => {
            error!("tenancy: no Kubernetes client, feature stays off: {}", e);
            return None;
        }
    };

    let manager = TenantManager::new(tenancy_config);
    let backend = Arc::new(KubeAllocBackend::new(
        client.clone(),
        &namespace,
        &config.tenancy_alloc_configmap,
    ));
    let store = AllocStore::new(Arc::clone(&manager), backend, WritePolicy::default());

    // Before any request is served, so a restart does not reset allocations.
    if let Err(e) = store.load_initial().await {
        warn!("tenancy: could not load persisted allocations: {}", e);
    }
    if let Err(e) = ConfigMapKeySource::with_client(client.clone())
        .start(Arc::clone(&manager))
        .await
    {
        error!(
            "tenancy: key watch failed to start, feature stays off: {}",
            e
        );
        return None;
    }
    store.start(Some(client));

    info!(
        "Tenancy enabled: keys={}/{}, alloc={}/{}, enforce={}",
        namespace, keys_configmap, namespace, config.tenancy_alloc_configmap, enforce
    );
    Some(TenancyRuntime {
        manager,
        store,
        enforce,
    })
}

/// Follows the fleet so the pool tracks worker capacity.
pub fn start_pool_sampler(runtime: &TenancyRuntime, registry: Arc<WorkerRegistry>) {
    spawn_pool_sampler(
        Arc::clone(&runtime.manager),
        reqwest::Client::new(),
        registry,
    );
}

/// Identifies the tenant, then counts the request against it. Added in this
/// order so the inflight layer, which runs first on the way in, sees the
/// `TenantCtx` the auth layer installs.
pub fn apply_layers<S>(routes: Router<S>, runtime: Option<&TenancyRuntime>) -> Router<S>
where
    S: Clone + Send + Sync + 'static,
{
    let Some(runtime) = runtime else {
        return routes;
    };
    routes
        .route_layer(axum::middleware::from_fn_with_state(
            Arc::clone(&runtime.manager),
            tenant_inflight_middleware,
        ))
        .route_layer(axum::middleware::from_fn_with_state(
            TenancyAuthState {
                manager: Arc::clone(&runtime.manager),
                enforce: runtime.enforce,
            },
            tenancy_auth_middleware,
        ))
}

/// The `/tenants` routes, behind the same identification layer as the data
/// plane so every handler sees a `TenantCtx`.
pub fn api_routes(runtime: Option<&TenancyRuntime>) -> Router {
    let Some(runtime) = runtime else {
        return Router::new();
    };
    api::router(Arc::clone(&runtime.store)).route_layer(axum::middleware::from_fn_with_state(
        TenancyAuthState {
            manager: Arc::clone(&runtime.manager),
            enforce: true,
        },
        tenancy_auth_middleware,
    ))
}
