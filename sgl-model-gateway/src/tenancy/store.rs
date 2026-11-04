//! WP-C: persisting allocations to the alloc ConfigMap.
//!
//! Contract:
//!   - In-memory state is authoritative between writes; the ConfigMap exists so
//!     allocations survive a restart.
//!   - Updates carry the observed `resourceVersion`; on 409 re-read and retry.
//!   - `AllocTable::generation` is bumped on every gateway-originated write so
//!     the watcher can ignore its own echo instead of reloading in a loop.
//!   - Writes are coalesced (idle flush, with a floor on write frequency) so a
//!     burst of resizes does not hammer the API server.
//!   - The gateway needs get/list/watch/update on this ConfigMap only; the key
//!     ConfigMap stays read-only so a compromised gateway cannot mint keys.

use super::{manager::TenantManager, types::AllocTable};

impl TenantManager {
    pub fn install_alloc(&self, table: AllocTable) {
        self.alloc.store(std::sync::Arc::new(table));
    }

    /// Applies a mutation under the write lock and schedules a write-back.
    pub async fn mutate_alloc<F>(&self, _mutate: F) -> Result<(), String>
    where
        F: FnOnce(&mut AllocTable),
    {
        // TODO(WP-C): take alloc_write_lock, apply, bump generation, enqueue flush.
        unimplemented!("WP-C")
    }
}
