//! Per-tenant identity and concurrency accounting, opt-in via `--tenancy-*` flags.
//!
//! Nothing here runs unless a tenancy source is configured: the community
//! `--api-key` and `--control-plane-api-keys` paths stay untouched, and this
//! table is consulted only after they miss.
//!
//! File ownership while the feature is being built in parallel:
//!   source.rs, auth.rs  -> key table loading and request identification
//!   pool.rs, gate.rs    -> pool sizing and in-flight accounting
//!   store.rs, api.rs    -> allocation persistence and the self-service API
//! `TenantManager` fields are `pub(crate)`, so each of those files adds its own
//! `impl TenantManager` block instead of editing manager.rs.

pub mod api;
pub mod auth;
pub mod gate;
pub mod manager;
pub mod metrics;
pub mod pool;
pub mod quota;
pub mod source;
pub mod store;
pub mod types;
pub mod wiring;

pub use manager::TenantManager;
pub use types::{
    AllocTable, KeyId, KeyRole, PoolSnapshot, Quota, QuotaDecision, TenantCtx, TenantId, TenantKey,
    TenantSnapshot, TenantSpec, TenantTable,
};

/// Runtime configuration for the tenancy module; `None` anywhere disables it.
#[derive(Debug, Clone)]
pub struct TenancyConfig {
    /// Namespace holding both ConfigMaps.
    pub namespace: String,
    /// ConfigMap with the key table; the gateway only reads it.
    pub keys_configmap: String,
    /// ConfigMap with per-tenant allocations; the gateway writes this one back.
    pub alloc_configmap: String,
    /// Fraction of the workers' combined capacity offered to tenants.
    pub pool_headroom: f32,
    /// Fixed pool size; when set, worker capacity is not consulted.
    pub pool_size_override: Option<u32>,
    /// Full resync period backing up the watch stream.
    pub resync_secs: u64,
    /// Reject a request this table does not identify. Required whenever no
    /// community data-plane key is set, since that path then admits everyone
    /// and unidentified traffic would escape accounting entirely.
    pub enforce: bool,
}

impl Default for TenancyConfig {
    fn default() -> Self {
        Self {
            namespace: "default".to_string(),
            keys_configmap: "smg-tenant-keys".to_string(),
            alloc_configmap: "smg-tenant-alloc".to_string(),
            pool_headroom: 0.8,
            pool_size_override: None,
            resync_secs: 30,
            enforce: false,
        }
    }
}
