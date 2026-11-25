//! Strategy orchestrator — runs all strategies in parallel, merges/deduplicates,
//! sorts by evidence tier, and applies rule-derived gates to produce the final
//! `Vec<AccIssue>`.

use std::collections::HashMap;
use std::path::Path;

use futures::future::join_all;

use crate::llm::LlmClient;
use crate::trace::TraceInventory;
use crate::types::{AccIssue, AccRootCause, FixLocus, RootObject};

use crate::accuracy::detector::{AnalysisCtx, Detector, JudgmentLog, RawIssue};
use crate::accuracy::extract::SharedExtraction;
use crate::accuracy::strategies::confirm_before_act::ConfirmBeforeActStrategy;
use crate::accuracy::strategies::experience_library::ExperienceLibraryStrategy;
use crate::accuracy::strategies::fact_check::FactCheckStrategy;
use crate::accuracy::strategies::requirement_check::RequirementCheckStrategy;
use crate::accuracy::strategies::verify_before_done::VerifyBeforeDoneStrategy;

/// Run all strategies in parallel, merge, deduplicate, sort, and apply gates.
pub async fn run_strategies(
    client: &LlmClient,
    inv: &TraceInventory,
    extraction: &SharedExtraction,
    repo_root: Option<&Path>,
    judgments: &JudgmentLog,
) -> Vec<AccIssue> {
    let ctx = AnalysisCtx {
        inv,
        client,
        repo_root,
        extraction,
        judgments,
    };

    // Build strategy list.
    let detectors: Vec<Box<dyn Detector>> = vec![
        Box::new(VerifyBeforeDoneStrategy::new()),
        Box::new(RequirementCheckStrategy::new()),
        Box::new(ConfirmBeforeActStrategy::new()),
        Box::new(FactCheckStrategy::new()),
        Box::new(ExperienceLibraryStrategy::new()),
    ];

    tracing::info!(
        "[accuracy] Running {} strategies in parallel...",
        detectors.len()
    );

    // Run all detectors concurrently.
    let futures: Vec<_> = detectors
        .iter()
        .map(|d| {
            let name = d.name();
            let fut = d.detect(&ctx);
            tracing::info!("[accuracy] Detector '{}' — running...", name);
            async move {
                let issues = fut.await;
                if issues.is_empty() {
                    tracing::info!("[accuracy] Detector '{}' ✗ no issues", name);
                } else {
                    tracing::info!(
                        "[accuracy] Detector '{}' ✓ produced {} issues",
                        name,
                        issues.len()
                    );
                }
                (name, issues)
            }
        })
        .collect();

    let results = join_all(futures).await;

    // Flatten all raw issues.
    let all_raw: Vec<(/*detector*/ &str, RawIssue)> = results
        .into_iter()
        .flat_map(|(name, issues)| issues.into_iter().map(move |i| (name, i)))
        .collect();

    tracing::info!(
        "[accuracy] Total raw issues before dedup: {}",
        all_raw.len()
    );

    // Deduplicate: same (defect_type, tool_call_id anchor) → keep strongest tier.
    let deduped = deduplicate(all_raw);

    // Apply rule-derived gates, then sort: strongest evidence first (L1 → L5),
    // same tier ordered by symptom for stable output.
    let mut issues: Vec<AccIssue> = deduped
        .into_iter()
        .map(|(_, raw)| raw_to_acc_issue(raw, inv))
        .collect();
    issues.sort_by(|a, b| {
        a.evidence_tier
            .cmp(&b.evidence_tier)
            .then_with(|| a.symptom.cmp(&b.symptom))
    });
    issues
}

/// Deduplicate issues: same (defect_type, tool_call_id) keeps the strongest tier.
fn deduplicate(raw_issues: Vec<(&str, RawIssue)>) -> Vec<(&str, RawIssue)> {
    // Key = (defect_type, tool_call_id or symptom hash).
    let mut best: HashMap<(String, String), (/*detector*/ &str, RawIssue)> = HashMap::new();

    for (detector, issue) in raw_issues {
        let key = (
            format!("{:?}", issue.defect_type),
            issue
                .tool_call_id
                .clone()
                .unwrap_or_else(|| issue.symptom.clone()),
        );

        let should_replace = match best.get(&key) {
            Some((_, existing)) => issue.evidence_tier < existing.evidence_tier,
            None => true,
        };

        if should_replace {
            best.insert(key, (detector, issue));
        }
    }

    // `best` is a HashMap, whose iteration order is randomized per map
    // instance. `run_strategies` then sorts by (evidence_tier, symptom)
    // only, so survivors that tie on both — e.g. the same command failing
    // twice, one issue per anchor — kept the map's random order and the
    // final issue list changed between runs of the same trace. Emit the
    // survivors ordered by their dedup key so the later stable sort starts
    // from a deterministic order.
    let mut survivors: Vec<((String, String), (_, RawIssue))> = best.into_iter().collect();
    survivors.sort_by(|a, b| a.0.cmp(&b.0));
    survivors.into_iter().map(|(_, issue)| issue).collect()
}

/// Convert a `RawIssue` to a finalized `AccIssue` by applying rule-derived gates.
fn raw_to_acc_issue(raw: RawIssue, inv: &TraceInventory) -> AccIssue {
    let fix_locus = FixLocus::from_primary(&raw.primary_object);
    let confidence = raw.evidence_tier.confidence().to_string();
    let auto_patch = raw.evidence_tier.allows_auto_patch();
    let optimizable = !matches!(raw.primary_object, RootObject::Env | RootObject::Input);

    // Recovery requires the same recorded file target, not only its preview.
    let (evidence, at, recovered) = match raw.tool_call_id.as_deref() {
        Some(id) => {
            if let Some((idx, rec)) = inv
                .tool_calls
                .iter()
                .enumerate()
                .find(|(_, c)| c.call_id == id)
            {
                let ev = format!(
                    "{} {}{}",
                    rec.name,
                    rec.cmd,
                    if rec.err { " ✗" } else { "" }
                );
                let at_str = format!("@{}s", rec.start.round() as i64);
                let rec_later = rec.err && recovered_later(&inv.tool_calls, idx);
                (ev, at_str, rec_later)
            } else {
                (String::new(), String::new(), false)
            }
        }
        None => (String::new(), String::new(), false),
    };

    let tier = if recovered {
        "internal-lead"
    } else {
        "user-failure"
    };

    AccIssue {
        symptom: raw.symptom,
        defect_type: raw.defect_type,
        root_cause: vec![AccRootCause {
            object: raw.primary_object,
            role: "主因".into(),
        }],
        fix_locus,
        confidence,
        optimizable,
        tier: tier.to_string(),
        recovered,
        evidence_tier: raw.evidence_tier,
        auto_patch,
        evidence,
        at,
        detail: raw.detail,
        verify: raw.verify,
        fix: raw.fix,
    }
}

/// Whether an errored call was later followed by a successful matching call.
fn recovered_later(calls: &[crate::types::ToolCallRecord], errored_idx: usize) -> bool {
    let Some(errored) = calls.get(errored_idx) else {
        return false;
    };
    let Some(later_calls) = errored_idx
        .checked_add(1)
        .and_then(|next| calls.get(next..))
    else {
        return false;
    };
    // `cmd` is a ~50-character display preview. Two Bash commands that share
    // that prefix are not the same command, and Bash records no separate
    // target, so recovery must compare complete execution evidence when the
    // record carries it. Legacy records without it keep the preview fallback.
    //
    // Require both a later slice position and a non-decreasing timestamp so
    // untimed traces (all `start == 0.0`) and same-second retries count as
    // recovered while earlier calls and reversed timestamps do not.
    let errored_command = errored.execution_command();
    later_calls.iter().any(|c| {
        let same_command = match (c.execution_command(), &errored_command) {
            (Some(candidate), Some(errored)) => candidate == *errored,
            _ => c.cmd == errored.cmd,
        };
        c.name == errored.name
            && same_command
            && c.target == errored.target
            && c.start >= errored.start
            && !c.err
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::trace::TraceInventory;
    use crate::types::{DefectType, EvidenceTier, ToolCallRecord};

    fn make_inv(calls: Vec<ToolCallRecord>) -> TraceInventory {
        TraceInventory {
            tool_calls: calls,
            user_turns: vec![],
            final_answer: String::new(),
            skill_contract: None,
        }
    }

    fn make_call(name: &str, cmd: &str, start: f64, err: bool) -> ToolCallRecord {
        ToolCallRecord {
            name: name.into(),
            call_id: format!("{name}_{start}"),
            start,
            dur: 1.0,
            cmd: cmd.into(),
            err,
            target: None,
            command: None,
            result_tokens: None,
        }
    }

    #[test]
    fn dedup_keeps_strongest_tier() {
        let issues = vec![
            (
                "d1",
                RawIssue {
                    symptom: "test".into(),
                    defect_type: DefectType::Workflow,
                    primary_object: RootObject::Skill,
                    evidence_tier: EvidenceTier::L4,
                    tool_call_id: Some("abc".into()),
                    detail: String::new(),
                    verify: String::new(),
                    fix: String::new(),
                },
            ),
            (
                "d2",
                RawIssue {
                    symptom: "test".into(),
                    defect_type: DefectType::Workflow,
                    primary_object: RootObject::Skill,
                    evidence_tier: EvidenceTier::L1,
                    tool_call_id: Some("abc".into()),
                    detail: String::new(),
                    verify: String::new(),
                    fix: String::new(),
                },
            ),
        ];

        let deduped = deduplicate(issues);
        assert_eq!(deduped.len(), 1);
        assert_eq!(deduped[0].1.evidence_tier, EvidenceTier::L1);
    }

    #[test]
    fn dedup_orders_equal_tier_and_symptom_deterministically() {
        // The same command failing twice yields one issue per anchor with
        // identical (evidence_tier, symptom) — both survive dedup (distinct
        // tool_call_id keys), and `run_strategies` then sorts by
        // (evidence_tier, symptom) only, so their relative order followed
        // the HashMap's randomized iteration order and the final issue
        // list changed between runs of the same trace.
        fn issue(call_id: &str) -> RawIssue {
            RawIssue {
                symptom: "声称完成但工具 bash 执行失败且未恢复: cargo build".into(),
                defect_type: DefectType::Workflow,
                primary_object: RootObject::Skill,
                evidence_tier: EvidenceTier::L1,
                tool_call_id: Some(call_id.into()),
                detail: String::new(),
                verify: String::new(),
                fix: String::new(),
            }
        }

        let input = vec![
            ("verify_before_done", issue("call-z")),
            ("verify_before_done", issue("call-a")),
            ("verify_before_done", issue("call-m")),
        ];

        // Every `deduplicate` call builds a fresh HashMap, whose iteration
        // order is randomized per map instance.
        for _ in 0..16 {
            let deduped = deduplicate(input.clone());
            let ids: Vec<&str> = deduped
                .iter()
                .map(|(_, i)| i.tool_call_id.as_deref().unwrap())
                .collect();
            assert_eq!(ids, vec!["call-a", "call-m", "call-z"]);
        }
    }

    #[test]
    fn recovered_matches_by_name_and_cmd() {
        let calls = vec![
            make_call("Bash", "git clone url1", 10.0, true),
            make_call("Bash", "git clone url2", 20.0, false), // different cmd → NOT recovered
        ];
        assert!(!recovered_later(&calls, 0));

        let calls2 = vec![
            make_call("Bash", "git clone url1", 10.0, true),
            make_call("Bash", "git clone url1", 20.0, false), // same cmd → recovered
        ];
        assert!(recovered_later(&calls2, 0));
    }

    /// `cmd` is a ~50-character preview; Bash has no separate `target`, so
    /// two different long commands sharing that prefix used to count as
    /// "recovered" and flipped a real failure to `internal-lead`.
    #[test]
    fn recovered_matches_the_full_command_not_the_truncated_preview() {
        let preview = r#"{"command":"cargo test --workspace --all-features …"#;
        let mut failed = make_call("Bash", preview, 10.0, true);
        failed.command = Some("cargo test --workspace --all-features --test alpha".to_string());
        let mut different = make_call("Bash", preview, 20.0, false);
        different.command = Some("cargo test --workspace --all-features --test beta".to_string());

        assert!(
            !recovered_later(&[failed.clone(), different.clone()], 0),
            "a different command sharing the preview must not count as recovery"
        );

        let mut same = different;
        same.command = failed.command.clone();
        assert!(recovered_later(&[failed.clone(), same], 0));
    }

    fn recorded_file_issue(first_target: &str, second_target: &str) -> AccIssue {
        let doc = serde_json::json!({
            "schema_version":"ATIF-v1.7", "session_id":"fixture-session",
            "agent":{"name":"fixture-agent","version":"1"},
            "steps":[
                {"step_id":1,"source":"agent","timestamp":"2026-10-10T00:00:01Z",
                 "message":"read the first fixture file", "tool_calls":[
                    {"tool_call_id":"first-read","function_name":"Read","arguments":{"file_path":first_target}}
                 ], "observation":{"results":[
                    {"source_call_id":"first-read","content":"fixture file not found", "extra":{"is_error":true}}
                 ]}},
                {"step_id":2,"source":"agent","timestamp":"2026-10-10T00:00:02Z",
                 "message":"read the second fixture file", "tool_calls":[
                    {"tool_call_id":"second-read","function_name":"Read","arguments":{"file_path":second_target}}
                 ], "observation":{"results":[
                    {"source_call_id":"second-read","content":"fixture text", "extra":{"is_error":false}}
                 ]}}
            ]
        });
        let schema: agentsight_atif::AtifTrajectory = serde_json::from_value(doc).unwrap();
        let trajectory =
            crate::atif::AtifTrajectory::from_json(&serde_json::to_string(&schema).unwrap())
                .unwrap();
        let inv = crate::trace::build_inventory(&trajectory);
        assert_eq!(inv.tool_calls.len(), 2);
        assert!(inv.tool_calls[0].err);
        assert!(!inv.tool_calls[1].err);
        assert_eq!(inv.tool_calls[0].cmd, inv.tool_calls[1].cmd);
        assert_eq!(inv.tool_calls[0].target.as_deref(), Some(first_target));
        assert_eq!(inv.tool_calls[1].target.as_deref(), Some(second_target));
        raw_to_acc_issue(
            RawIssue {
                symptom: "first file read failed".into(),
                defect_type: DefectType::Workflow,
                primary_object: RootObject::Env,
                evidence_tier: EvidenceTier::L2,
                tool_call_id: Some("first-read".into()),
                detail: String::new(),
                verify: String::new(),
                fix: String::new(),
            },
            &inv,
        )
    }

    #[test]
    fn a_successful_read_of_another_full_target_is_not_recovery() {
        let prefix = "/fixture/a-long-directory-name/shared-nested-directory/configuration/";
        let issue = recorded_file_issue(
            &format!("{prefix}missing.txt"),
            &format!("{prefix}present.txt"),
        );
        assert!(
            !issue.recovered,
            "another file's success does not recover the anchored failure"
        );
        assert_eq!(issue.tier, "user-failure");
    }

    #[test]
    fn a_successful_retry_of_the_same_full_target_is_recovery() {
        let target =
            "/fixture/a-long-directory-name/shared-nested-directory/configuration/input.txt";
        let issue = recorded_file_issue(target, target);
        assert!(issue.recovered);
        assert_eq!(issue.tier, "internal-lead");
    }

    fn untimed_read_issue(first_err: bool, second_err: bool, anchor_call_id: &str) -> AccIssue {
        let target = "/fixture/config.toml";
        let doc = serde_json::json!({
            "schema_version":"ATIF-v1.7", "session_id":"untimed-session",
            "agent":{"name":"fixture-agent","version":"1"},
            "steps":[
                {"step_id":1,"source":"agent","message":"first read", "tool_calls":[
                    {"tool_call_id":"call-1","function_name":"Read","arguments":{"file_path":target}}
                ], "observation":{"results":[
                    {"source_call_id":"call-1","content":"step 1","extra":{"is_error":first_err}}
                ]}},
                {"step_id":2,"source":"agent","message":"second read", "tool_calls":[
                    {"tool_call_id":"call-2","function_name":"Read","arguments":{"file_path":target}}
                ], "observation":{"results":[
                    {"source_call_id":"call-2","content":"step 2","extra":{"is_error":second_err}}
                ]}}
            ]
        });
        let trajectory = crate::atif::AtifTrajectory::from_json(&doc.to_string()).unwrap();
        let inv = crate::trace::build_inventory(&trajectory);
        assert_eq!(inv.tool_calls.len(), 2);
        assert_eq!(inv.tool_calls[0].start, 0.0);
        assert_eq!(inv.tool_calls[1].start, 0.0);
        raw_to_acc_issue(
            RawIssue {
                symptom: "untimed read failed".into(),
                defect_type: DefectType::Workflow,
                primary_object: RootObject::Skill,
                evidence_tier: EvidenceTier::L1,
                tool_call_id: Some(anchor_call_id.into()),
                detail: String::new(),
                verify: String::new(),
                fix: String::new(),
            },
            &inv,
        )
    }

    #[test]
    fn untimed_atif_retry_counts_as_recovered() {
        let issue = untimed_read_issue(true, false, "call-1");
        assert!(
            issue.recovered,
            "later successful retry on untimed ATIF trace must recover earlier error"
        );
        assert_eq!(issue.tier, "internal-lead");
    }

    #[test]
    fn untimed_atif_earlier_success_does_not_recover_later_error() {
        let issue = untimed_read_issue(false, true, "call-2");
        assert!(
            !issue.recovered,
            "earlier success on untimed ATIF trace must not recover later error"
        );
        assert_eq!(issue.tier, "user-failure");
    }

    fn issue_for_calls(calls: Vec<ToolCallRecord>, anchor_id: &str) -> AccIssue {
        let inv = make_inv(calls);
        raw_to_acc_issue(
            RawIssue {
                symptom: "bash failed".into(),
                defect_type: DefectType::Workflow,
                primary_object: RootObject::Skill,
                evidence_tier: EvidenceTier::L1,
                tool_call_id: Some(anchor_id.into()),
                detail: String::new(),
                verify: String::new(),
                fix: String::new(),
            },
            &inv,
        )
    }

    #[test]
    fn same_timestamp_retry_counts_as_recovered_and_respects_order() {
        let mut fail = make_call("Bash", "cargo test", 10.0, true);
        fail.call_id = "call-1".into();
        let mut ok = make_call("Bash", "cargo test", 10.0, false);
        ok.call_id = "call-2".into();

        let retry_issue = issue_for_calls(vec![fail.clone(), ok.clone()], "call-1");
        assert!(
            retry_issue.recovered,
            "same-second later retry must count as recovered"
        );
        assert_eq!(retry_issue.tier, "internal-lead");

        let earlier_ok_issue = issue_for_calls(vec![ok, fail], "call-1");
        assert!(
            !earlier_ok_issue.recovered,
            "same-second earlier success must not recover later error"
        );
        assert_eq!(earlier_ok_issue.tier, "user-failure");
    }

    #[test]
    fn negative_and_reversed_timestamps_preserve_recovery_order() {
        // Negative timestamps in chronological order still count as recovered.
        let mut neg_fail = make_call("Bash", "cargo test", -5.0, true);
        neg_fail.call_id = "neg-1".into();
        let mut neg_ok = make_call("Bash", "cargo test", -1.0, false);
        neg_ok.call_id = "neg-2".into();
        assert!(issue_for_calls(vec![neg_fail, neg_ok], "neg-1").recovered);

        // Later slice entry with an earlier timestamp does not recover.
        let mut late_fail = make_call("Bash", "cargo test", 20.0, true);
        late_fail.call_id = "rev-1".into();
        let mut early_ok = make_call("Bash", "cargo test", 10.0, false);
        early_ok.call_id = "rev-2".into();
        assert!(!issue_for_calls(vec![late_fail.clone(), early_ok.clone()], "rev-1").recovered);

        // Earlier slice entry with a later timestamp does not recover.
        let mut late_ok = make_call("Bash", "cargo test", 30.0, false);
        late_ok.call_id = "rev-3".into();
        assert!(!issue_for_calls(vec![late_ok, late_fail], "rev-1").recovered);
    }

    #[test]
    fn confidence_derived_from_tier() {
        assert_eq!(EvidenceTier::L1.confidence(), "高");
        assert_eq!(EvidenceTier::L2.confidence(), "高");
        assert_eq!(EvidenceTier::L3.confidence(), "中");
        assert_eq!(EvidenceTier::L4.confidence(), "中");
        assert_eq!(EvidenceTier::L5.confidence(), "低");
    }

    #[test]
    fn auto_patch_gate() {
        assert!(EvidenceTier::L1.allows_auto_patch());
        assert!(EvidenceTier::L2.allows_auto_patch());
        assert!(EvidenceTier::L3.allows_auto_patch());
        assert!(!EvidenceTier::L4.allows_auto_patch());
        assert!(!EvidenceTier::L5.allows_auto_patch());
    }

    #[test]
    fn fix_locus_from_primary() {
        assert_eq!(FixLocus::from_primary(&RootObject::Skill), FixLocus::Skill);
        assert_eq!(
            FixLocus::from_primary(&RootObject::Context),
            FixLocus::ContextPolicy
        );
        assert_eq!(
            FixLocus::from_primary(&RootObject::Model),
            FixLocus::ModelRouting
        );
        assert_eq!(FixLocus::from_primary(&RootObject::Tool), FixLocus::Tool);
        assert_eq!(FixLocus::from_primary(&RootObject::Env), FixLocus::None);
        assert_eq!(FixLocus::from_primary(&RootObject::Input), FixLocus::None);
    }

    #[test]
    fn raw_to_acc_issue_sets_gates_correctly() {
        let inv = make_inv(vec![make_call("Edit", "file.rs", 5.0, true)]);

        let raw = RawIssue {
            symptom: "test issue".into(),
            defect_type: DefectType::Knowledge,
            primary_object: RootObject::Skill,
            evidence_tier: EvidenceTier::L2,
            tool_call_id: Some("Edit_5".into()),
            detail: "detail".into(),
            verify: "verify".into(),
            fix: "fix".into(),
        };

        let issue = raw_to_acc_issue(raw, &inv);
        assert_eq!(issue.defect_type, DefectType::Knowledge);
        assert_eq!(issue.fix_locus, FixLocus::Skill);
        assert_eq!(issue.confidence, "高");
        assert!(issue.auto_patch);
        assert!(issue.optimizable);
        assert!(!issue.recovered); // no later success
        assert!(issue.evidence.contains("Edit"));
    }

    #[test]
    fn l4_issue_has_auto_patch_false() {
        let inv = make_inv(vec![]);
        let raw = RawIssue {
            symptom: "semantic issue".into(),
            defect_type: DefectType::Workflow,
            primary_object: RootObject::Skill,
            evidence_tier: EvidenceTier::L4,
            tool_call_id: None,
            detail: String::new(),
            verify: String::new(),
            fix: String::new(),
        };

        let issue = raw_to_acc_issue(raw, &inv);
        assert!(!issue.auto_patch);
        assert_eq!(issue.confidence, "中"); // L4 capped at 中
    }
}
