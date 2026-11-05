//! Per-model quota rules.
//!
//! Keys are global: one key table serves every model's router, and it carries
//! no quota. Quota is per model -- the same tenant can be capped differently on
//! each -- so each router reads its rules from its own ConfigMap, under
//! `data["quota.yaml"]`, beside the `alloc.yaml` it writes back:
//!
//!     version: 1
//!     tenants:
//!       chengsong: {initial: 16, floor: 4, ceiling: 48}
//!
//! This is the only place quota is configured. A tenant absent from the rules
//! is unlimited on this model.

use std::{collections::HashMap, sync::Arc};

use serde::Deserialize;
use tracing::warn;

use super::{manager::TenantManager, types::Quota};

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

impl TenantManager {
    /// Installs this model's rules. A rule naming a tenant the key table does
    /// not have is kept but has no effect until that tenant exists.
    pub fn install_quota_rules(&self, rules: QuotaRules) {
        let table = self.table();
        for tenant in rules.keys() {
            if !table.tenants.contains_key(tenant.as_str()) {
                warn!(
                    "quota rule for tenant {:?} has no matching tenant in the key table",
                    tenant
                );
            }
        }
        self.quota_rules.store(Arc::new(rules));
    }

    pub fn quota_rules(&self) -> arc_swap::Guard<Arc<QuotaRules>> {
        self.quota_rules.load()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::tenancy::{source::parse_key_table, types::AllocTable, TenancyConfig};

    const KEYS: &str = r#"
version: 1
tenants:
  - id: team-a
    keys: [{id: a1, secret: "sa"}]
  - id: team-b
    keys: [{id: b1, secret: "sb"}]
"#;

    fn q(initial: u32, floor: u32, ceiling: u32) -> Quota {
        Quota {
            initial,
            floor,
            ceiling,
        }
    }

    #[test]
    fn parses_rules_and_defaults_missing_fields() {
        let rules = parse_quota_rules(
            "version: 1\ntenants:\n  team-a: {initial: 16, floor: 4, ceiling: 48}\n  team-b: {ceiling: 8}\n",
        )
        .unwrap();
        assert_eq!(rules["team-a"], q(16, 4, 48));
        assert_eq!(rules["team-b"], q(0, 0, 8));
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
        model_x.install_quota_rules(QuotaRules::from([("team-a".to_string(), q(8, 0, 16))]));
        model_y.install_quota_rules(QuotaRules::from([("team-a".to_string(), q(2, 0, 4))]));

        let a: Arc<str> = Arc::from("team-a");
        let b: Arc<str> = Arc::from("team-b");
        assert_eq!(model_x.allocated(&a), Some(8));
        assert_eq!(model_y.allocated(&a), Some(2));
        assert_eq!(model_x.allocated(&b), None);
        assert_eq!(model_y.allocated(&b), None);
    }

    #[test]
    fn rules_and_keys_reload_independently() {
        let manager = TenantManager::new(TenancyConfig::default());
        manager.install_quota_rules(QuotaRules::from([("team-a".to_string(), q(8, 0, 16))]));
        // Rules installed before the key table apply once the tenant exists.
        let a: Arc<str> = Arc::from("team-a");
        assert_eq!(manager.allocated(&a), None);
        manager.install_table(parse_key_table(KEYS, "1".to_string()).unwrap());
        assert_eq!(manager.allocated(&a), Some(8));

        manager.install_quota_rules(QuotaRules::new());
        assert_eq!(manager.allocated(&a), None);
        assert!(manager.resolve_key("sa").is_some());
    }

    // An allocation persisted under an older rule must not escape a rule since
    // tightened by hand.
    #[test]
    fn a_persisted_allocation_is_clamped_to_the_current_rule() {
        let manager = TenantManager::new(TenancyConfig::default());
        manager.install_table(parse_key_table(KEYS, "1".to_string()).unwrap());
        let mut alloc = AllocTable::default();
        alloc.allocated.insert("team-a".to_string(), 10);
        alloc.allocated.insert("team-b".to_string(), 1);
        manager.install_alloc(alloc);
        manager.install_quota_rules(QuotaRules::from([
            ("team-a".to_string(), q(4, 1, 8)),
            ("team-b".to_string(), q(4, 2, 8)),
        ]));
        assert_eq!(manager.allocated(&Arc::from("team-a")), Some(8));
        assert_eq!(manager.allocated(&Arc::from("team-b")), Some(2));
    }
}
