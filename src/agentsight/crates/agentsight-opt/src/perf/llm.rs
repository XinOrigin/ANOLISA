//! LLM perf strategy selection: Rust extracts structured perf data, then fires
//! one LLM call per strategy in parallel. Each call evaluates whether a single
//! strategy applies and returns a causal chain fragment (signal → cause → strategy).
//! Results are merged into a unified causal graph + issue table.

use anyhow::Result;

use crate::atif::AtifTrajectory;
use crate::llm::LlmClient;
use crate::perf::prompts::perf_identification::{build_strategy_prompt, STRATEGIES};
use crate::types::{
    PerfCandidateSet, PerfCausalEdge, PerfCausalGraph, PerfCausalNode, PerfIssue, PerfNodeKind,
    PerfReport, PerfStrategyEval,
};

/// Truncate a string to a short label (≤ 20 chars, UTF-8 safe).
fn short_label(s: &str) -> String {
    let truncated: String = s.chars().take(20).collect();
    if s.chars().count() > 20 {
        format!("{truncated}…")
    } else {
        truncated
    }
}

/// Check whether sufficient data exists to evaluate a given strategy.
/// Skip the LLM call entirely if required data was not collected.
fn has_sufficient_data(candidates: &PerfCandidateSet, strategy_id: &str) -> bool {
    match strategy_id {
        // prefix_cache requires cache token data from trajectory
        "prefix_cache" => !candidates.cache_turns.is_empty(),
        // fast_tool requires tool call records
        "fast_tool" => !candidates.top_tools.is_empty(),
        // experience_library uses raw trajectory (always available)
        "experience_library" => true,
        _ => true,
    }
}

/// Run full perf strategy selection: Rust data extraction → parallel per-strategy LLM eval → merge.
pub async fn identify_issues(
    client: &LlmClient,
    trajectory: &AtifTrajectory,
) -> Result<PerfReport> {
    let candidates = crate::perf::extract_perf_candidates(trajectory)?;
    if candidates.top_tools.is_empty() && candidates.wall_secs <= 0.0 {
        tracing::info!("Perf: no candidates extracted, skipping LLM judgment");
        return Ok(PerfReport {
            items: vec![],
            considered: 0,
            dismissed: 0,
            failed: 0,
            wall_secs: candidates.wall_secs,
            causal_graph: None,
        });
    }

    let wall_secs = candidates.wall_secs;
    let considered = candidates.tool_count;
    tracing::info!(
        "Perf: {} tool calls, wall {:.1}s — evaluating {} strategies in parallel...",
        considered,
        wall_secs,
        STRATEGIES.len()
    );

    // Fire one LLM call per strategy in parallel (skip strategies lacking required data).
    let evaluated: Vec<_> = STRATEGIES
        .iter()
        .filter(|strategy| has_sufficient_data(&candidates, strategy.id))
        .collect();
    let skipped: Vec<_> = STRATEGIES
        .iter()
        .filter(|strategy| !has_sufficient_data(&candidates, strategy.id))
        .collect();

    for s in &skipped {
        tracing::info!("Perf: strategy '{}' skipped (insufficient data)", s.id);
    }
    for s in &evaluated {
        tracing::info!("Perf: strategy '{}' — sending to LLM...", s.id);
    }

    let futures: Vec<_> = evaluated
        .iter()
        .map(|strategy| {
            let messages = build_strategy_prompt(&candidates, strategy, trajectory);
            let strategy_id = strategy.id;
            let strategy_name = strategy.name;
            let label = format!("perf:{strategy_id}");
            async move {
                let result: std::result::Result<PerfStrategyEval, _> = client
                    .chat_json_parsed_labeled(messages, Some(&label))
                    .await;
                (strategy_id, strategy_name, result)
            }
        })
        .collect();

    let results = futures::future::join_all(futures).await;
    let judged = results.len();

    // Merge results into causal graph + issue rows.
    let mut nodes: Vec<PerfCausalNode> = Vec::new();
    let mut edges: Vec<PerfCausalEdge> = Vec::new();
    let mut items: Vec<PerfIssue> = Vec::new();
    let mut failed = 0usize;
    let mut last_err = None;
    let mut idx = 0usize;

    for (strategy_id, strategy_name, result) in results {
        let eval = match result {
            Ok(e) if e.applies => {
                tracing::info!(
                    "Perf: strategy '{}' ✓ applies — {} (saving ~{:.0}s)",
                    strategy_id,
                    e.symptom.chars().take(60).collect::<String>(),
                    e.estimated_saving_secs
                );
                e
            }
            Ok(e) => {
                tracing::info!("Perf: strategy '{}' ✗ not applicable", strategy_id);
                let _ = e;
                continue;
            }
            Err(err) => {
                tracing::warn!(
                    "Perf: strategy '{}' LLM call failed: {:#}",
                    strategy_id,
                    err
                );
                failed += 1;
                last_err = Some(err);
                continue;
            }
        };

        // Skip if essential fields are empty.
        if eval.symptom.is_empty() && eval.action.is_empty() {
            continue;
        }

        idx += 1;
        let sig_id = format!("sig_{idx}");
        let cause_id = format!("cause_{idx}");
        let strat_id = format!("strat_{idx}");

        let saving = eval.estimated_saving_secs;
        let confidence = if eval.confidence.is_empty() {
            "中".to_string()
        } else {
            eval.confidence.clone()
        };

        // Build causal graph nodes from flat fields.
        nodes.push(PerfCausalNode {
            id: sig_id.clone(),
            kind: PerfNodeKind::Signal,
            label: short_label(&eval.symptom),
            detail: eval.evidence.clone(),
            strategy_id: None,
            estimated_saving_secs: None,
            confidence: None,
        });
        nodes.push(PerfCausalNode {
            id: cause_id.clone(),
            kind: PerfNodeKind::Cause,
            label: short_label(&eval.root_cause),
            detail: eval.root_cause.clone(),
            strategy_id: None,
            estimated_saving_secs: None,
            confidence: None,
        });
        nodes.push(PerfCausalNode {
            id: strat_id.clone(),
            kind: PerfNodeKind::Strategy,
            label: short_label(&eval.action),
            detail: eval.action.clone(),
            strategy_id: Some(strategy_id.to_string()),
            estimated_saving_secs: Some(saving),
            confidence: Some(confidence.clone()),
        });

        edges.push(PerfCausalEdge {
            from: sig_id,
            to: cause_id.clone(),
            label: "导致".to_string(),
        });
        edges.push(PerfCausalEdge {
            from: cause_id,
            to: strat_id,
            label: "推荐".to_string(),
        });

        // Build PerfIssue row for the frontend table.
        let pct = if wall_secs > 0.0 {
            (saving / wall_secs) * 100.0
        } else {
            0.0
        };
        items.push(PerfIssue {
            strategy_id: strategy_id.to_string(),
            symptom: eval.symptom,
            category: String::new(),
            subtype: strategy_name.to_string(),
            root_cause: eval.root_cause,
            optimization: eval.action,
            evidence: eval.evidence,
            at: String::new(),
            impact_secs: saving,
            pct,
            confidence,
        });
    }

    // Biggest estimated saving first.
    items.sort_by(|a, b| {
        b.impact_secs
            .partial_cmp(&a.impact_secs)
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    tracing::info!(
        "Perf: {}/{} strategies matched",
        items.len(),
        STRATEGIES.len()
    );

    // A report built from zero successful judgments reads as a clean "no
    // issues" verdict downstream (it is persisted and rendered as such), so an
    // all-failed run must surface its error instead.
    if judged > 0 && failed == judged {
        return Err(match last_err {
            Some(err) => anyhow::anyhow!("all {failed} perf LLM judgments failed: {err:#}"),
            None => anyhow::anyhow!("all {failed} perf LLM judgments failed"),
        });
    }

    // Only strategies actually evaluated by the LLM (`judged`) can be kept,
    // dismissed, or failed; strategies skipped pre-evaluation were never judged.
    let dismissed = judged.saturating_sub(items.len()).saturating_sub(failed);
    let causal_graph = if nodes.is_empty() {
        None
    } else {
        Some(PerfCausalGraph { nodes, edges })
    };

    Ok(PerfReport {
        items,
        considered,
        dismissed,
        failed,
        wall_secs,
        causal_graph,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::{
        atomic::{AtomicUsize, Ordering},
        Arc,
    };
    use tokio::io::{AsyncReadExt, AsyncWriteExt};

    async fn spawn_mock_llm_server<F>(
        handler: F,
    ) -> (String, Arc<AtomicUsize>, tokio::task::JoinHandle<()>)
    where
        F: Fn(&str) -> (u16, String) + Send + Sync + 'static,
    {
        let listener = tokio::net::TcpListener::bind("127.0.0.1:0")
            .await
            .expect("bind mock server");
        let url = format!("http://{}/v1", listener.local_addr().expect("local addr"));
        let requests = Arc::new(AtomicUsize::new(0));
        let received = Arc::clone(&requests);
        let handler = Arc::new(handler);

        let handle = tokio::spawn(async move {
            loop {
                let Ok((mut stream, _)) = listener.accept().await else {
                    break;
                };
                let received = Arc::clone(&received);
                let handler = Arc::clone(&handler);
                tokio::spawn(async move {
                    let mut req = Vec::new();
                    let body_start = loop {
                        let mut buf = [0u8; 4096];
                        let Ok(n) = stream.read(&mut buf).await else {
                            return;
                        };
                        if n == 0 {
                            return;
                        }
                        req.extend_from_slice(&buf[..n]);
                        if let Some(pos) = req.windows(4).position(|w| w == b"\r\n\r\n") {
                            let headers = String::from_utf8_lossy(&req[..pos]).to_lowercase();
                            let len: usize = headers
                                .lines()
                                .find_map(|l| l.strip_prefix("content-length:"))
                                .and_then(|v| v.trim().parse().ok())
                                .unwrap_or(0);
                            if req.len() >= pos + 4 + len {
                                break pos + 4;
                            }
                        }
                    };
                    received.fetch_add(1, Ordering::SeqCst);
                    let body_str = String::from_utf8_lossy(&req[body_start..]);
                    let (status, resp_body) = handler(&body_str);
                    let reason = if status == 200 {
                        "OK"
                    } else {
                        "Internal Server Error"
                    };
                    let resp = format!(
                        "HTTP/1.1 {status} {reason}\r\nContent-Type: application/json\r\nContent-Length: {}\r\nConnection: close\r\n\r\n{resp_body}",
                        resp_body.len()
                    );
                    let _ = stream.write_all(resp.as_bytes()).await;
                    let _ = stream.shutdown().await;
                });
            }
        });

        (url, requests, handle)
    }

    fn chat_completion_json(content: serde_json::Value) -> String {
        serde_json::json!({
            "id": "fixture",
            "object": "chat.completion",
            "created": 0,
            "model": "test-model",
            "choices": [{
                "index": 0,
                "finish_reason": "stop",
                "message": {
                    "role": "assistant",
                    "content": content.to_string()
                }
            }],
            "usage": {"prompt_tokens": 10, "completion_tokens": 20, "total_tokens": 30}
        })
        .to_string()
    }

    fn sample_trajectory_without_cache() -> AtifTrajectory {
        AtifTrajectory::from_json(
            r#"{"schema_version":"ATIF-v1.6","session_id":"s1",
                "agent":{"name":"a","version":"1","model_name":"m"},
                "steps":[
                  {"step_id":1,"source":"agent","timestamp":"2026-07-02T06:30:00.000Z",
                   "tool_calls":[{"tool_call_id":"c1","function_name":"Bash",
                                  "arguments":{"command":"cargo test"}}],
                   "observation":{"results":[{"source_call_id":"c1","content":"ok"}]}},
                  {"step_id":2,"source":"agent","timestamp":"2026-07-02T06:30:20.000Z",
                   "message":"done"}
                ]}"#,
        )
        .unwrap()
    }

    fn sample_trajectory_with_cache() -> AtifTrajectory {
        AtifTrajectory::from_json(
            r#"{"schema_version":"ATIF-v1.6","session_id":"s2",
                "agent":{"name":"a","version":"1","model_name":"m"},
                "steps":[
                  {"step_id":1,"source":"agent","timestamp":"2026-07-02T06:30:00.000Z",
                   "metrics":{"prompt_tokens":1000,"cached_tokens":100},
                   "tool_calls":[{"tool_call_id":"c1","function_name":"Bash",
                                  "arguments":{"command":"cargo test"}}],
                   "observation":{"results":[{"source_call_id":"c1","content":"ok"}]}},
                  {"step_id":2,"source":"agent","timestamp":"2026-07-02T06:30:20.000Z",
                   "message":"done"}
                ]}"#,
        )
        .unwrap()
    }

    /// When every per-strategy judgment call fails (endpoint down, bad key),
    /// the run must surface an error instead of returning a report: the caller
    /// persists an Ok report as the dimension result and the dashboard renders
    /// it as "evaluated N strategies, none apply" — a false clean verdict that
    /// outlives the outage as stored data, the same trap 8c141e618 closed for
    /// the cost dimension.
    #[tokio::test]
    async fn all_failed_judgments_error_instead_of_a_clean_report() {
        let t = sample_trajectory_without_cache();
        let client = LlmClient::with_config("http://127.0.0.1:1/v1", "key", "test-model");
        let report = identify_issues(&client, &t).await;
        assert!(
            report.is_err(),
            "every perf strategy failed against a dead endpoint; \
             expected Err, got {report:?}"
        );
    }

    #[tokio::test]
    async fn all_failed_http_responses_return_err_cleanly() {
        let t = sample_trajectory_without_cache();
        let (url, requests, server) = spawn_mock_llm_server(|_| {
            (
                500,
                serde_json::json!({"error": {"message": "upstream outage"}}).to_string(),
            )
        })
        .await;
        let client = LlmClient::with_config(url, "key", "test-model");
        let err = identify_issues(&client, &t)
            .await
            .expect_err("all 500 responses must surface Err without panicking");
        server.abort();
        assert!(requests.load(Ordering::SeqCst) > 0);
        assert!(
            err.to_string().contains("all 2 perf LLM judgments failed"),
            "unexpected error message: {err:#}"
        );
    }

    #[tokio::test]
    async fn skipped_strategies_are_not_counted_as_dismissed() {
        let t = sample_trajectory_without_cache();
        let (url, requests, server) = spawn_mock_llm_server(|body| {
            if body.contains("fast_tool") {
                (
                    200,
                    chat_completion_json(serde_json::json!({
                        "applies": true,
                        "symptom": "slow grep",
                        "root_cause": "used grep",
                        "action": "use rg",
                        "evidence": "step 1",
                        "estimated_saving_secs": 5.0,
                        "confidence": "高"
                    })),
                )
            } else {
                (
                    200,
                    chat_completion_json(serde_json::json!({
                        "applies": false,
                        "symptom": "",
                        "root_cause": "",
                        "action": "",
                        "evidence": "",
                        "estimated_saving_secs": 0.0,
                        "confidence": ""
                    })),
                )
            }
        })
        .await;

        let client = LlmClient::with_config(url, "key", "test-model");
        let report = identify_issues(&client, &t).await.unwrap();
        server.abort();

        // prefix_cache was skipped pre-eval due to missing cache_turns, so only 2 strategies ran.
        assert_eq!(requests.load(Ordering::SeqCst), 2);
        assert_eq!(report.items.len(), 1);
        assert_eq!(report.failed, 0);
        assert_eq!(
            report.dismissed, 1,
            "only evaluated not-applicable strategy (experience_library) should be dismissed"
        );
    }

    #[tokio::test]
    async fn evaluated_not_applicable_strategies_are_counted_as_dismissed() {
        let t = sample_trajectory_with_cache();
        let (url, requests, server) = spawn_mock_llm_server(|body| {
            if body.contains("fast_tool") {
                (
                    200,
                    chat_completion_json(serde_json::json!({
                        "applies": true,
                        "symptom": "slow grep",
                        "root_cause": "used grep",
                        "action": "use rg",
                        "evidence": "step 1",
                        "estimated_saving_secs": 5.0,
                        "confidence": "高"
                    })),
                )
            } else {
                (
                    200,
                    chat_completion_json(serde_json::json!({
                        "applies": false,
                        "symptom": "",
                        "root_cause": "",
                        "action": "",
                        "evidence": "",
                        "estimated_saving_secs": 0.0,
                        "confidence": ""
                    })),
                )
            }
        })
        .await;

        let client = LlmClient::with_config(url, "key", "test-model");
        let report = identify_issues(&client, &t).await.unwrap();
        server.abort();

        assert_eq!(requests.load(Ordering::SeqCst), 3);
        assert_eq!(report.items.len(), 1);
        assert_eq!(report.failed, 0);
        assert_eq!(report.dismissed, 2);
    }

    #[tokio::test]
    async fn partial_failure_with_skipped_strategy_keeps_dismissed_accurate() {
        let t = sample_trajectory_without_cache();
        let (url, requests, server) = spawn_mock_llm_server(|body| {
            if body.contains("fast_tool") {
                (
                    200,
                    chat_completion_json(serde_json::json!({
                        "applies": true,
                        "symptom": "slow grep",
                        "root_cause": "used grep",
                        "action": "use rg",
                        "evidence": "step 1",
                        "estimated_saving_secs": 5.0,
                        "confidence": "高"
                    })),
                )
            } else {
                (
                    500,
                    serde_json::json!({"error": {"message": "upstream error"}}).to_string(),
                )
            }
        })
        .await;

        let client = LlmClient::with_config(url, "key", "test-model");
        let report = identify_issues(&client, &t).await.unwrap();
        server.abort();

        assert!(requests.load(Ordering::SeqCst) >= 2);
        assert_eq!(report.items.len(), 1);
        assert_eq!(report.failed, 1);
        assert_eq!(report.dismissed, 0);
    }
}
