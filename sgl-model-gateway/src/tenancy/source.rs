//! WP-A: loading the key table from its ConfigMap.
//!
//! Contract:
//!   - `KeySource::start` spawns a watch that calls `TenantManager::install_table`
//!     on every change, backed by a full resync every `config.resync_secs`.
//!   - A ConfigMap that fails to parse leaves the previous table in force; it
//!     must never clear the table, or every tenant loses access at once.
//!   - The ConfigMap carries plaintext keys under `data["tenants.yaml"]`; they
//!     are hashed into `TenantKey` at load and the plaintext is dropped.

use std::sync::Arc;

use super::{manager::TenantManager, types::TenantTable};

#[async_trait::async_trait]
pub trait KeySource: Send + Sync {
    async fn start(&self, manager: Arc<TenantManager>) -> Result<(), String>;
}

impl TenantManager {
    pub fn install_table(&self, table: TenantTable) {
        self.table.store(Arc::new(table));
    }
}

/// Parses the `tenants.yaml` document into a table.
pub fn parse_key_table(_yaml: &str, _resource_version: String) -> Result<TenantTable, String> {
    // TODO(WP-A): serde_yaml -> TenantSpec + TenantKey, reject duplicate key hashes.
    Err("not implemented".to_string())
}
