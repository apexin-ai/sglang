//! Data model shared by every tenancy file. Treat as frozen: the parallel
//! work packages all build against these types.

use std::{collections::HashMap, sync::Arc};

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use subtle::ConstantTimeEq;

pub type TenantId = Arc<str>;
pub type KeyId = Arc<str>;

/// Role of a key. Only the control plane reads it; the data plane admits any
/// key in the table.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum KeyRole {
    #[default]
    User,
    Admin,
}

impl KeyRole {
    pub fn is_admin(&self) -> bool {
        matches!(self, KeyRole::Admin)
    }
}

pub fn hash_key(key: &str) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(key.as_bytes());
    hasher.finalize().into()
}

/// One credential. The ConfigMap carries the key in plaintext; it is hashed at
/// load time and the plaintext is dropped.
#[derive(Clone)]
pub struct TenantKey {
    pub id: KeyId,
    pub tenant: TenantId,
    pub role: KeyRole,
    pub disabled: bool,
    hash: [u8; 32],
}

impl TenantKey {
    pub fn new(id: KeyId, tenant: TenantId, role: KeyRole, disabled: bool, secret: &str) -> Self {
        Self {
            id,
            tenant,
            role,
            disabled,
            hash: hash_key(secret),
        }
    }

    pub fn hash(&self) -> &[u8; 32] {
        &self.hash
    }

    pub fn verify(&self, presented: &str) -> bool {
        !self.disabled && self.hash.ct_eq(&hash_key(presented)).into()
    }
}

impl std::fmt::Debug for TenantKey {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("TenantKey")
            .field("id", &self.id)
            .field("tenant", &self.tenant)
            .field("role", &self.role)
            .field("disabled", &self.disabled)
            .field("hash", &"[REDACTED]")
            .finish()
    }
}

/// A tenant's concurrency cap and the bounds it may move that cap within.
/// `initial` seeds the allocation when no persisted value exists. A tenant
/// with no quota at all is unlimited, which is why callers see `Option<Quota>`.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct Quota {
    pub initial: u32,
    pub floor: u32,
    pub ceiling: u32,
}

/// A tenant from the shared key table. It carries no quota: quota is per model
/// and comes from `TenantManager::quota_of`.
#[derive(Debug, Clone)]
pub struct TenantSpec {
    pub id: TenantId,
    pub name: String,
}

/// Immutable snapshot of the key table, swapped in whole on every reload.
#[derive(Debug, Default, Clone)]
pub struct TenantTable {
    pub tenants: HashMap<TenantId, TenantSpec>,
    /// Lookup by key hash; a tenant may hold several keys for rotation.
    pub by_hash: HashMap<[u8; 32], TenantKey>,
    /// Resource version of the ConfigMap this was built from.
    pub resource_version: String,
}

/// What a request carries once a key is resolved, inserted into the request
/// extensions for downstream layers and the self-service API.
#[derive(Debug, Clone)]
pub struct TenantCtx {
    pub tenant: TenantId,
    pub key_id: KeyId,
    pub role: KeyRole,
}

/// Per-tenant allocations as persisted in the alloc ConfigMap.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct AllocTable {
    pub allocated: HashMap<String, u32>,
    /// Bumped on every gateway-originated write so the watch can tell its own
    /// updates from an external edit.
    pub generation: u64,
}

#[derive(Debug, Clone, Copy, Serialize)]
pub struct PoolSnapshot {
    /// Derived from healthy worker capacity, or the configured override.
    pub total: u32,
    pub allocated: u32,
    /// Allocations are scaled down only once `allocated` would exceed `total`.
    pub oversubscribed: bool,
}

#[derive(Debug, Clone, Serialize)]
pub struct TenantSnapshot {
    pub tenant: String,
    /// Absent when the tenant is unlimited.
    pub allocated: Option<u32>,
    /// Requests in flight right now, counted until the response body ends.
    pub inflight: u32,
    pub quota: Option<Quota>,
    pub limited: bool,
    /// Requests admitted while already at or above the allocation; always 0
    /// while unlimited, since there is nothing to be over.
    pub over_quota_total: u64,
}

/// Outcome of a self-service resize. A grow is capped by both the tenant
/// ceiling and the free pool, so a partial grant is normal.
#[derive(Debug, Clone, Serialize)]
pub struct QuotaDecision {
    pub granted: u32,
    pub requested: u32,
    pub pool_free: u32,
    pub reason: &'static str,
}
