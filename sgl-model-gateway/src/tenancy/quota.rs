//! Per-model quota rules.
//!
//! Keys are global: one key table serves every model's router. Quota is not --
//! the same tenant can be capped differently on each model -- so each router
//! reads its rules from its own per-model ConfigMap, under `data["quota.yaml"]`,
//! beside the `alloc.yaml` it writes back:
//!
//!     version: 1
//!     tenants:
//!       chengsong: {initial: 16, floor: 4, ceiling: 48}
//!
//! A tenant absent from the rules is unlimited on this model. The effective
//! table the rest of the module reads is the key table with each tenant's quota
//! filled in from these rules, rebuilt whenever either side changes.

use std::{collections::HashMap, sync::Arc};

use serde::Deserialize;
use tracing::warn;

use super::{
    manager::TenantManager,
    types::{Quota, TenantTable},
};

/// Key of the rules document inside the per-model ConfigMap's `data` map.
pub const QUOTA_DATA_KEY: &str = "quota.yaml";

pub type QuotaRules = HashMap<String, Quota>;

#[derive(Debug, Deserialize)]
struct QuotaDocument {
    #[serde(default)]
    version: u32,
    #[serde(default)]
    tenants: HashMap<String, QuotaEntry>,
}

/// Each field defaults, so a partial block such as `{ceiling: 32}` loads.
#[derive(Debug, Default, Deserialize)]
struct QuotaEntry {
    #[serde(default)]
    initial: u32,
    #[serde(default)]
    floor: u32,
    #[serde(default)]
    ceiling: u32,
}

/// Parses the `quota.yaml` document. Unknown fields are tolerated for the same
/// reason as in the key table: the document is hand-edited and may be written
/// by a newer gateway than the one reading it.
pub fn parse_quota_rules(yaml: &str) -> Result<QuotaRules, String> {
    if yaml.trim().is_empty() {
        return Ok(QuotaRules::new());
    }
    let doc: QuotaDocument =
        serde_yaml::from_str(yaml).map_err(|e| format!("invalid quota.yaml: {}", e))?;
    if doc.version != 0 && doc.version != 1 {
        return Err(format!(
            "unsupported quota.yaml version {}; this gateway understands 1",
            doc.version
        ));
    }
    let mut rules = QuotaRules::new();
    for (tenant, e) in doc.tenants {
        if e.ceiling < e.floor {
            return Err(format!(
                "tenant {:?}: quota ceiling {} is below floor {}",
                tenant, e.ceiling, e.floor
            ));
        }
        if e.initial < e.floor || e.initial > e.ceiling {
            return Err(format!(
                "tenant {:?}: quota initial {} is outside [{}, {}]",
                tenant, e.initial, e.floor, e.ceiling
            ));
        }
        rules.insert(
            tenant,
            Quota {
                initial: e.initial,
                floor: e.floor,
                ceiling: e.ceiling,
            },
        );
    }
    Ok(rules)
}

/// The key table with each tenant's quota taken from `rules`. A rule for a
/// tenant the key table does not have is reported and dropped.
pub fn merge(keys: &TenantTable, rules: &QuotaRules) -> TenantTable {
    let mut merged = keys.clone();
    for (id, spec) in merged.tenants.iter_mut() {
        spec.quota = rules.get(id.as_ref()).copied();
    }
    for tenant in rules.keys() {
        if !keys.tenants.contains_key(tenant.as_str()) {
            warn!(
                "quota rule for tenant {:?} has no matching tenant in the key table; ignored",
                tenant
            );
        }
    }
    merged
}

impl TenantManager {
    /// Installs this model's rules and rebuilds the effective table.
    pub fn install_quota_rules(&self, rules: QuotaRules) {
        let _guard = self.merge_lock.lock().unwrap_or_else(|e| e.into_inner());
        self.quota_rules.store(Arc::new(rules));
        self.remerge_locked();
    }

    pub fn quota_rules(&self) -> arc_swap::Guard<Arc<QuotaRules>> {
        self.quota_rules.load()
    }

    /// Caller holds `merge_lock`, so a key reload and a rules reload cannot
    /// interleave and leave one side's update out of the published table.
    pub(crate) fn remerge_locked(&self) {
        let merged = merge(&self.keys_raw.load(), &self.quota_rules.load());
        self.table.store(Arc::new(merged));
    }
}

#[cfg(test)]
impl TenantManager {
    /// Installs a table whose specs already carry quotas, splitting it back into
    /// a key table and this model's rules. Lets fixtures state a tenant and its
    /// cap in one place.
    pub(crate) fn install_table_with_quotas(&self, table: TenantTable) {
        let rules: QuotaRules = table
            .tenants
            .iter()
            .filter_map(|(id, spec)| Some((id.to_string(), spec.quota?)))
            .collect();
        self.install_table(table);
        self.install_quota_rules(rules);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tenancy::{source::parse_key_table, TenancyConfig};

    const KEYS: &str = r#"
version: 1
tenants:
  - id: team-a
    keys: [{id: a1, secret: "sa"}]
  - id: team-b
    keys: [{id: b1, secret: "sb"}]
"#;

    #[test]
    fn parses_rules_and_defaults_missing_fields() {
        let rules = parse_quota_rules(
            "version: 1\ntenants:\n  team-a: {initial: 16, floor: 4, ceiling: 48}\n  team-b: {ceiling: 8}\n",
        )
        .unwrap();
        assert_eq!(
            rules["team-a"],
            Quota {
                initial: 16,
                floor: 4,
                ceiling: 48
            }
        );
        assert_eq!(
            rules["team-b"],
            Quota {
                initial: 0,
                floor: 0,
                ceiling: 8
            }
        );
    }

    #[test]
    fn an_empty_document_means_no_caps() {
        assert!(parse_quota_rules("").unwrap().is_empty());
        assert!(parse_quota_rules("version: 1\n").unwrap().is_empty());
    }

    #[test]
    fn rejects_inconsistent_bounds() {
        assert!(parse_quota_rules("tenants:\n  a: {initial: 99, floor: 0, ceiling: 8}\n").is_err());
        assert!(parse_quota_rules("tenants:\n  a: {initial: 1, floor: 4, ceiling: 2}\n").is_err());
        assert!(parse_quota_rules("version: 2\n").is_err());
    }

    #[test]
    fn rules_apply_per_model_and_absent_tenants_stay_unlimited() {
        // Two managers stand in for two models' routers sharing one key table.
        let keys = parse_key_table(KEYS, "1".to_string()).unwrap();
        let model_x = TenantManager::new(TenancyConfig::default());
        let model_y = TenantManager::new(TenancyConfig::default());
        model_x.install_table(keys.clone());
        model_y.install_table(keys);
        model_x.install_quota_rules(
            parse_quota_rules("tenants:\n  team-a: {initial: 8, floor: 0, ceiling: 16}\n").unwrap(),
        );
        model_y.install_quota_rules(
            parse_quota_rules("tenants:\n  team-a: {initial: 2, floor: 0, ceiling: 4}\n").unwrap(),
        );

        let a: Arc<str> = Arc::from("team-a");
        let b: Arc<str> = Arc::from("team-b");
        assert_eq!(model_x.allocated(&a), Some(8));
        assert_eq!(model_y.allocated(&a), Some(2));
        assert_eq!(model_x.allocated(&b), None);
        assert_eq!(model_y.allocated(&b), None);
    }

    #[test]
    fn a_key_reload_keeps_the_rules_and_a_rules_reload_keeps_the_keys() {
        let manager = TenantManager::new(TenancyConfig::default());
        manager.install_quota_rules(
            parse_quota_rules("tenants:\n  team-a: {initial: 8, floor: 0, ceiling: 16}\n").unwrap(),
        );
        // Rules installed before the key table must still apply once it lands.
        manager.install_table(parse_key_table(KEYS, "1".to_string()).unwrap());
        let a: Arc<str> = Arc::from("team-a");
        assert_eq!(manager.allocated(&a), Some(8));

        manager.install_quota_rules(QuotaRules::new());
        assert_eq!(manager.allocated(&a), None);
        assert!(manager.resolve_key("sa").is_some());
    }
}
