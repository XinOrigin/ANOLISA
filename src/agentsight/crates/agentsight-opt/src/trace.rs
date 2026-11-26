//! Shared trace parsing layer — `TraceInventory`.
//!
//! Extracts structured data from ATIF trajectories that all analysis
//! dimensions (accuracy, perf, cost) consume.

use crate::atif::{observation_result_is_error, AtifTrajectory};
use crate::types::ToolCallRecord;
use agentsight_atif::resolve_call_id;

// ── Public types ──

/// A user's natural-language turn in the conversation.
#[derive(Debug, Clone)]
pub struct UserTurn {
    pub turn: usize, // 1-based ordinal among genuine user text turns
    pub text: String,
}

/// Placeholder for the Skill contract (layer II oracle, not implemented yet).
#[derive(Debug, Clone)]
pub struct SkillContract {
    pub _placeholder: (),
}

/// Structured inventory of a trajectory — shared across all analysis dimensions.
#[derive(Debug, Clone)]
pub struct TraceInventory {
    pub tool_calls: Vec<ToolCallRecord>,
    pub user_turns: Vec<UserTurn>,
    pub final_answer: String,
    pub skill_contract: Option<SkillContract>,
}

/// Maximum characters for command summary truncation (UTF-8 safe).
const CMD_TRUNCATE_CHARS: usize = 50;

/// Maximum characters per user turn text (UTF-8 safe).
const USER_TURN_TEXT_CHARS: usize = 300;

/// System-injected XML tags to strip from user/assistant text.
///
/// These are injected by the IDEs and CLIs this pipeline reads (QoderWork/Qoder
/// reminders, Claude Code slash-command transcripts) and are not genuine user
/// content. The list must stay in step with
/// `agentsight_trajectory_collector::strip_system_context`, whose copy is the
/// documented contract for external callers: a tag only this list knows about is
/// stripped here and kept there, and the two views of one trajectory then
/// disagree about which turns the user actually wrote. The earlier two-pair list
/// made the optimization pipeline count a `/clear` transcript as a user turn
/// while the preferences page dropped it.
const SYSTEM_TAGS: &[(&str, &str)] = &[
    ("<system-reminder>", "</system-reminder>"),
    ("<current_notes_content>", "</current_notes_content>"),
    ("<command-message>", "</command-message>"),
    ("<command-name>", "</command-name>"),
    ("<command-args>", "</command-args>"),
    ("<local-command-caveat>", "</local-command-caveat>"),
    ("<local-command-stdout>", "</local-command-stdout>"),
    ("<local-command-stderr>", "</local-command-stderr>"),
];

/// Known plain-text patterns that indicate a QoderWork system-injected block.
/// When a text block contains one of these patterns near the start, the entire
/// block is system context (e.g., MEMORY.md dump, awareness-mode file injection)
/// and should be treated as non-user content.
const SYSTEM_TEXT_MARKERS: &[&str] = &["Target file this round:", "[SYSTEM NOTIFICATION"];

/// Maximum byte offset within which a system marker must appear to be
/// considered a system-only block (handles leading app name / path lines).
const SYSTEM_MARKER_SCAN_LIMIT: usize = 200;

/// System command suffixes injected by QoderWork at the end of text blocks.
const SYSTEM_TEXT_SUFFIXES: &[&str] = &["Please reflect and reorganize the target file."];

/// Return the smallest UTF-8 character boundary at or after `byte_limit`
/// (clamped to `text.len()`).
fn char_boundary_at_or_after(text: &str, byte_limit: usize) -> usize {
    if byte_limit >= text.len() {
        return text.len();
    }
    let mut end = byte_limit;
    while end < text.len() && !text.is_char_boundary(end) {
        end += 1;
    }
    end
}

/// Strip XML-tagged system blocks left-to-right by earliest opening tag so an
/// unclosed mention of one tag inside another closed block does not truncate
/// the outer closing tag and following user text.
fn strip_system_tags(text: &str) -> String {
    let mut out = String::with_capacity(text.len());
    let mut cursor = 0;

    while cursor < text.len() {
        let rest = &text[cursor..];
        let next = SYSTEM_TAGS
            .iter()
            .filter_map(|&(open, close)| rest.find(open).map(|rel| (rel, open, close)))
            .min_by_key(|&(rel, _, _)| rel);

        let Some((rel, open, close)) = next else {
            out.push_str(rest);
            break;
        };

        out.push_str(&rest[..rel]);
        let after_open = cursor + rel + open.len();
        if let Some(close_rel) = text[after_open..].find(close) {
            cursor = after_open + close_rel + close.len();
        } else {
            // An unclosed system tag swallows the remainder of the message.
            break;
        }
    }

    out
}

/// Check whether a text block is entirely system-injected context
/// (e.g., a QoderWork MEMORY.md dump with no real user message).
pub fn is_system_only_text(text: &str) -> bool {
    let trimmed = text.trim();
    // Allow markers that start before SYSTEM_MARKER_SCAN_LIMIT to finish past
    // the boundary without scanning the rest of a large message.
    SYSTEM_TEXT_MARKERS.iter().any(|marker| {
        let scan_end = char_boundary_at_or_after(
            trimmed,
            SYSTEM_MARKER_SCAN_LIMIT.saturating_add(marker.len()),
        );
        trimmed[..scan_end]
            .find(marker)
            .is_some_and(|pos| pos < SYSTEM_MARKER_SCAN_LIMIT)
    })
}

/// Remove system-injected context blocks from message text.
///
/// QoderWork / Qoder IDE injects environment info, MEMORY.md content,
/// MCP server lists, etc. in two formats:
/// 1. XML-tagged: `<system-reminder>...</system-reminder>`
/// 2. Plain-text: "Target file this round: ..." (entire block is system context)
///
/// Both formats are stripped. For plain-text injections, any trailing
/// system commands (e.g., "Please reflect and reorganize...") are also removed.
pub fn strip_system_context(text: &str) -> String {
    // Phase 1: Strip XML tags first so markers inside a reminder do not wipe a
    // genuine user turn, and leading tags do not hide a plain-text system dump.
    let mut result = strip_system_tags(text);

    // Phase 2: If the remaining text is entirely a system-injected block, return empty
    if is_system_only_text(&result) {
        return String::new();
    }

    // Phase 3: Strip trailing system command lines
    for suffix in SYSTEM_TEXT_SUFFIXES {
        if let Some(pos) = result.rfind(suffix) {
            result.truncate(pos);
        }
    }

    // Collapse multiple blank lines left by stripping
    while result.contains("\n\n\n") {
        result = result.replace("\n\n\n", "\n\n");
    }
    result.trim().to_string()
}

// ── Public API ──

/// Build a `TraceInventory` from an ATIF trajectory.
///
/// Tool calls with their observations live on the same agent step, so no
/// cross-event pairing is needed. Tool durations come from the step timing
/// model (see [`crate::atif`] module docs).
pub fn build_inventory(traj: &AtifTrajectory) -> TraceInventory {
    let tool_calls = collect_tool_calls(traj);
    let user_turns = extract_user_turns(traj);
    let final_answer = extract_final_answer(traj);

    TraceInventory {
        tool_calls,
        user_turns,
        final_answer,
        skill_contract: None, // Layer II — not implemented yet
    }
}

/// Collect tool calls across agent steps with timing and error detection.
///
/// Per-call duration = the step's tool window (next agent step start − this
/// step end) divided evenly among the step's parallel calls, so summing
/// durations stays consistent with wall-clock accounting. A valid recorded
/// result time instead gives the individual call's duration; parallel measured
/// durations can overlap, and performance totals count their union separately.
pub fn collect_tool_calls(traj: &AtifTrajectory) -> Vec<ToolCallRecord> {
    collect_tool_calls_with(traj, CMD_TRUNCATE_CHARS)
}

/// Same as [`collect_tool_calls`] with a custom command summary length.
pub fn collect_tool_calls_with(traj: &AtifTrajectory, cmd_chars: usize) -> Vec<ToolCallRecord> {
    let steps = &traj.steps;
    let origin = traj.origin_ts();

    let mut out = Vec::new();
    for (i, step) in steps.iter().enumerate() {
        if !step.is_agent() || step.calls().is_empty() {
            continue;
        }
        let window = tool_window_secs(traj, i);
        let n = step.calls().len();
        let per_call = if n > 0 { window / n as f64 } else { 0.0 };
        let start = step
            .end_ts()
            .zip(origin)
            .map(|(t, origin)| (t - origin).as_seconds_f64())
            .unwrap_or(0.0);

        // Match observations to calls by id; positional fallback only for
        // documents whose results carry no ids at all. Falling back whenever
        // the id lookup misses would hand a sibling's observation to a call
        // whose result never arrived (interrupted execution), reporting that
        // call as failed and feeding the misattribution into the accuracy
        // detectors and cost ledger. Ids are compared with the shared schema's
        // tolerant rule: our own converters pair results to calls that way and
        // then store the provider's echo, so a strict `==` misses exactly the
        // documents they wrote.
        for (k, call) in step.calls().iter().enumerate() {
            let result = step
                .results()
                .iter()
                .find(|r| {
                    r.source_call_id.as_deref().is_some_and(|id| {
                        resolve_call_id(
                            id,
                            step.calls().iter().map(|call| call.tool_call_id.as_str()),
                        ) == Some(call.tool_call_id.as_str())
                    })
                })
                .or_else(|| {
                    step.results()
                        .iter()
                        .all(|r| r.source_call_id.is_none())
                        .then(|| step.results().get(k))
                        .flatten()
                });
            // Structured flag first, text heuristic only for flag-less documents.
            let err = result.map(observation_result_is_error).unwrap_or(false);
            out.push(ToolCallRecord {
                name: call.display_name(),
                call_id: call.tool_call_id.clone(),
                start: step
                    .tool_start_ts(call)
                    .zip(origin)
                    .map(|(time, origin)| (time - origin).as_seconds_f64())
                    .unwrap_or(start),
                dur: step
                    .tool_result_ts(call)
                    .zip(step.tool_start_ts(call))
                    .map(|(result, start)| (result - start).as_seconds_f64())
                    .unwrap_or_else(|| {
                        let offset = step
                            .tool_start_ts(call)
                            .zip(step.end_ts())
                            .map(|(start, end)| (start - end).as_seconds_f64())
                            .unwrap_or(0.0);
                        per_call.min((window - offset).max(0.0)).max(0.0)
                    }),
                cmd: call.command_summary(cmd_chars),
                err,
                target: file_target(call),
                command: crate::tool_command::from_arguments(&call.arguments),
                result_tokens: None,
            });
        }
    }
    out
}

/// The file/path argument a call acts on, when it takes one. The command
/// summary is a JSON blob truncated at ~50 chars - too short for deep paths -
/// so consumers that need the target (e.g. files-touched aggregation) must
/// read the argument, not the summary.
fn file_target(call: &crate::atif::AtifToolCall) -> Option<String> {
    ["file_path", "path", "notebook_path", "filePath"]
        .iter()
        .find_map(|k| call.arguments.get(k).and_then(|v| v.as_str()))
        .map(str::to_string)
}

/// Tool window from recorded result times. When any completion is unrecorded,
/// extend it using the next agent step's start (or end), provided no user step
/// intervenes. Recorded completions remain usable even for terminal steps.
pub(crate) fn tool_window_secs(traj: &AtifTrajectory, idx: usize) -> f64 {
    let steps = &traj.steps;
    let Some(end) = steps[idx].end_ts() else {
        return 0.0;
    };
    let recorded: Vec<_> = steps[idx]
        .calls()
        .iter()
        .filter_map(|call| steps[idx].tool_result_ts(call))
        .map(|timestamp| (timestamp - end).as_seconds_f64())
        .collect();
    let measured = recorded.iter().copied().fold(0.0_f64, f64::max);
    if recorded.len() == steps[idx].calls().len() {
        return measured;
    }
    for next in &steps[idx + 1..] {
        if next.is_user() {
            return measured;
        }
        if next.is_agent() {
            let next_start = next.start_ts().or_else(|| next.end_ts());
            return measured.max(
                next_start
                    .map(|t| (t - end).as_seconds_f64().max(0.0))
                    .unwrap_or(0.0),
            );
        }
    }
    measured
}

/// Union of tool windows; parallel measured calls consume wall time once.
pub(crate) fn tool_busy_secs(traj: &AtifTrajectory) -> f64 {
    let Some(origin) = traj.origin_ts() else {
        return 0.0;
    };
    let windows: Vec<_> = traj
        .steps
        .iter()
        .enumerate()
        .filter(|(_, step)| step.is_agent() && !step.calls().is_empty())
        .flat_map(|(index, step)| {
            let fallback_end = step
                .end_ts()
                .map(|time| (time - origin).as_seconds_f64() + tool_window_secs(traj, index));
            step.calls().iter().filter_map(move |call| {
                let start = (step.tool_start_ts(call)? - origin).as_seconds_f64();
                let end = step
                    .tool_result_ts(call)
                    .map(|time| (time - origin).as_seconds_f64())
                    .or(fallback_end)?;
                Some((start, end.max(start)))
            })
        })
        .collect();
    busy_window_secs(windows)
}

/// Wall time covered by any of the supplied activity windows.
pub(crate) fn busy_window_secs(mut windows: Vec<(f64, f64)>) -> f64 {
    windows.sort_by(|left, right| left.0.total_cmp(&right.0));
    let mut end = f64::NEG_INFINITY;
    let mut total = 0.0;
    for (start, next_end) in windows {
        total += (next_end - start.max(end)).max(0.0);
        end = end.max(next_end);
    }
    total
}

// ── Internal implementation ──

/// Extract user natural-language turns from ATIF user steps.
fn extract_user_turns(traj: &AtifTrajectory) -> Vec<UserTurn> {
    let mut out = Vec::new();
    let mut turn = 0usize;

    for step in traj.steps.iter().filter(|s| s.is_user()) {
        let raw = step.message.as_deref().unwrap_or("");
        let text = strip_system_context(raw);
        let text = text.trim();
        if text.is_empty() {
            continue;
        }
        turn += 1;
        out.push(UserTurn {
            turn,
            text: text.chars().take(USER_TURN_TEXT_CHARS).collect::<String>(),
        });
    }

    out
}

/// Heuristic: extract the last agent text message as the "final answer".
/// This will be overridden by the LLM extractor in the accuracy pipeline.
/// Also strips system-injected context blocks.
fn extract_final_answer(traj: &AtifTrajectory) -> String {
    for step in traj.steps.iter().rev() {
        // An unanswered follow-up must not inherit a previous turn's claims.
        // Match user-turn extraction so injected context is not a boundary.
        if step.is_user()
            && !strip_system_context(step.message.as_deref().unwrap_or(""))
                .trim()
                .is_empty()
        {
            break;
        }
        if !step.is_agent() {
            continue;
        }
        if let Some(msg) = step.message.as_deref() {
            let cleaned = strip_system_context(msg);
            if !cleaned.is_empty() {
                return cleaned;
            }
        }
    }
    String::new()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::atif::AtifTrajectory;

    fn traj(steps_json: &str) -> AtifTrajectory {
        AtifTrajectory::from_json(&format!(
            r#"{{"schema_version":"ATIF-v1.6","session_id":"s1",
                "agent":{{"name":"a","version":"1"}},"steps":{steps_json}}}"#
        ))
        .unwrap()
    }

    #[test]
    fn parses_tool_calls_from_atif() {
        let inv = build_inventory(&traj(
            r#"[
            {"step_id":1,"source":"agent","timestamp":"2025-01-01T00:00:01Z",
             "extra":{"start_timestamp":"2025-01-01T00:00:00Z"},
             "tool_calls":[{"tool_call_id":"c1","function_name":"Bash","arguments":{"command":"ls -la"}}],
             "observation":{"results":[{"source_call_id":"c1","content":"file1\nfile2"}]}},
            {"step_id":2,"source":"agent","timestamp":"2025-01-01T00:00:07Z",
             "extra":{"start_timestamp":"2025-01-01T00:00:06Z"},
             "message":"Done!"}
        ]"#,
        ));

        assert_eq!(inv.tool_calls.len(), 1);
        assert_eq!(inv.tool_calls[0].name, "Bash");
        assert_eq!(inv.tool_calls[0].call_id, "c1");
        assert!(!inv.tool_calls[0].err);
        // window = step2.start (t=6) − step1.end (t=1) = 5s
        assert!((inv.tool_calls[0].dur - 5.0).abs() < 0.01);
        assert!(inv.final_answer.contains("Done"));
    }

    #[test]
    fn extracts_user_turns() {
        let inv = build_inventory(&traj(
            r#"[
            {"step_id":1,"source":"user","timestamp":"2025-01-01T00:00:00Z","message":"请帮我修复这个 bug"},
            {"step_id":2,"source":"agent","timestamp":"2025-01-01T00:00:01Z","message":"好的"},
            {"step_id":3,"source":"user","timestamp":"2025-01-01T00:00:03Z","message":"为什么不用 gh cli 呢"}
        ]"#,
        ));

        assert_eq!(inv.user_turns.len(), 2);
        assert_eq!(inv.user_turns[0].turn, 1);
        assert!(inv.user_turns[0].text.contains("修复"));
        assert!(inv.user_turns[1].text.contains("gh cli"));
    }

    #[test]
    fn detects_errored_observation() {
        let inv = build_inventory(&traj(
            r#"[
            {"step_id":1,"source":"agent","timestamp":"2025-01-01T00:00:01Z",
             "tool_calls":[{"tool_call_id":"c1","function_name":"Bash","arguments":{"command":"ls /nope"}}],
             "observation":{"results":[{"source_call_id":"c1","content":"ls: cannot access '/nope': No such file or directory"}]}}
        ]"#,
        ));
        assert_eq!(inv.tool_calls.len(), 1);
        assert!(inv.tool_calls[0].err);
    }

    #[test]
    fn handles_empty_trajectory() {
        let inv = build_inventory(&traj("[]"));
        assert!(inv.tool_calls.is_empty());
        assert!(inv.user_turns.is_empty());
        assert!(inv.final_answer.is_empty());
        assert!(inv.skill_contract.is_none());
    }

    #[test]
    fn strip_system_reminder_blocks() {
        let input = "用户真实问题\n\n<system-reminder>MEMORY content here</system-reminder>\n<system-reminder>MCP servers</system-reminder>";
        let result = strip_system_context(input);
        assert_eq!(result, "用户真实问题");
    }

    /// The tag list is shared with the trajectory collector, whose copy is the
    /// documented contract for external callers and covers the slash-command and
    /// local-command transcripts the IDEs inject as well. This copy had only the
    /// first two pairs, so the optimization pipeline fed the summary stage a
    /// `/clear` transcript as if the user had typed it — while the preferences
    /// page stripped the same turn from the same trajectory.
    #[test]
    fn slash_command_transcripts_are_not_user_turns() {
        for transcript in [
            "<command-message>clear</command-message>\n<command-name>/clear</command-name>",
            "<command-args></command-args>",
            "<local-command-caveat>Caveat: the messages below were generated by the user</local-command-caveat>",
            "<local-command-stdout>done</local-command-stdout>",
            "<local-command-stderr>failed</local-command-stderr>",
        ] {
            assert_eq!(
                strip_system_context(transcript),
                "",
                "{transcript} is system-injected, not user text"
            );
        }

        let inv = build_inventory(&traj(
            r#"[
            {"step_id":1,"source":"user","timestamp":"2025-01-01T00:00:00Z",
             "message":"<command-message>clear</command-message>\n<command-name>/clear</command-name>"},
            {"step_id":2,"source":"user","timestamp":"2025-01-01T00:00:01Z","message":"真正的问题"},
            {"step_id":3,"source":"agent","timestamp":"2025-01-01T00:00:02Z","message":"回答"}
        ]"#,
        ));

        assert_eq!(
            inv.user_turns.len(),
            1,
            "only the genuine question is a turn"
        );
        assert_eq!(inv.user_turns[0].text, "真正的问题");
    }

    #[test]
    fn strip_preserves_user_selected_text() {
        let input = "请分析这段代码\n<user-selected-text>fn main() {}</user-selected-text>";
        let result = strip_system_context(input);
        assert!(result.contains("user-selected-text"));
    }

    #[test]
    fn strip_unclosed_system_tag() {
        let input = "用户问题\n<system-reminder>unclosed content without end tag";
        let result = strip_system_context(input);
        assert_eq!(result, "用户问题");
    }

    #[test]
    fn user_turns_strip_system_context() {
        let inv = build_inventory(&traj(
            r#"[
            {"step_id":1,"source":"user","timestamp":"2025-01-01T00:00:00Z",
             "message":"如何判断相关性\n\n<system-reminder>MEMORY.md content\nMCP servers list</system-reminder>"},
            {"step_id":2,"source":"agent","timestamp":"2025-01-01T00:00:01Z","message":"用embedding"}
        ]"#,
        ));

        assert_eq!(inv.user_turns.len(), 1);
        assert_eq!(inv.user_turns[0].text, "如何判断相关性");
    }

    #[test]
    fn is_system_only_detects_target_file() {
        assert!(is_system_only_text(
            "Target file this round: MEMORY.md\nCurrent usage: 5949 bytes\n..."
        ));
        // Qoder format: app name + path + Target file
        assert!(is_system_only_text(
            "QoderWork\n/Users/cheng\nTarget file this round: MEMORY.md..."
        ));
        assert!(is_system_only_text(
            "[SYSTEM NOTIFICATION - NOT USER INPUT]\nThis is automated..."
        ));
        assert!(!is_system_only_text("请帮我分析这个代码"));
        // Multi-byte UTF-8 near the scan boundary must not panic
        let chinese = "功能输出核心功能输出核心功能输出核心功能输出核心功能输出核心功能输出核心功能输出核心功能输出核心功能输出核心功能输出核心";
        assert!(!is_system_only_text(chinese));
    }

    #[test]
    fn strip_target_file_returns_empty() {
        let input = "Target file this round: MEMORY.md\nCurrent usage: 5949 / 10240 bytes (58%).\n\nFull MEMORY.md entries...\nPlease reflect and reorganize the target file.";
        let result = strip_system_context(input);
        assert!(result.is_empty());
    }

    #[test]
    fn tool_calls_carry_file_target() {
        let inv = build_inventory(&traj(
            r#"[
            {"step_id":1,"source":"agent","timestamp":"2025-01-01T00:00:01Z",
             "tool_calls":[{"tool_call_id":"c1","function_name":"Edit",
               "arguments":{"file_path":"src/agentsight/deep/path/mod.rs","old_string":"a","new_string":"b"}}],
             "observation":{"results":[{"source_call_id":"c1","content":"ok"}]}},
            {"step_id":2,"source":"agent","timestamp":"2025-01-01T00:00:02Z",
             "tool_calls":[{"tool_call_id":"c2","function_name":"Bash","arguments":{"command":"cargo test"}}]}
        ]"#,
        ));
        assert_eq!(
            inv.tool_calls[0].target.as_deref(),
            Some("src/agentsight/deep/path/mod.rs")
        );
        assert_eq!(inv.tool_calls[1].target, None);
    }

    #[test]
    fn user_turns_skip_target_file_session() {
        let inv = build_inventory(&traj(
            r#"[
            {"step_id":1,"source":"user","timestamp":"2025-01-01T00:00:00Z",
             "message":"Target file this round: MEMORY.md\nFull MEMORY.md entries...\nPlease reflect and reorganize the target file."}
        ]"#,
        ));
        assert!(inv.user_turns.is_empty());
    }

    #[test]
    fn a_call_without_an_observation_does_not_inherit_a_siblings_error() {
        // c1's tool result never arrived (interrupted execution); c2's did and
        // failed. The positional fallback must not hand c2's observation to c1:
        // err then claims c1 failed, which feeds the accuracy detectors and the
        // cost ledger's churn accounting as a misattributed failure.
        let inv = build_inventory(&traj(
            r#"[
            {"step_id":1,"source":"agent","timestamp":"2025-01-01T00:00:01Z",
             "tool_calls":[
                {"tool_call_id":"c1","function_name":"Bash","arguments":{"command":"ls"}},
                {"tool_call_id":"c2","function_name":"Bash","arguments":{"command":"ls /nope"}}
             ],
             "observation":{"results":[
                {"source_call_id":"c2","content":"ls: cannot access '/nope': No such file or directory"}]}}
        ]"#,
        ));
        assert_eq!(inv.tool_calls.len(), 2);
        assert!(
            !inv.tool_calls[0].err,
            "a call with no observation must not inherit its sibling's failure"
        );
        assert!(inv.tool_calls[1].err, "c2's own observation is an error");
    }

    #[test]
    fn id_less_observation_sets_still_pair_positionally() {
        let inv = build_inventory(&traj(
            r#"[
            {"step_id":1,"source":"agent","timestamp":"2025-01-01T00:00:01Z",
             "tool_calls":[
                {"tool_call_id":"c1","function_name":"Bash","arguments":{"command":"ls"}},
                {"tool_call_id":"c2","function_name":"Bash","arguments":{"command":"ls /nope"}}
             ],
             "observation":{"results":[
                {"content":"file1"},
                {"content":"ls: cannot access '/nope': No such file or directory"}]}}
        ]"#,
        ));
        assert!(!inv.tool_calls[0].err);
        assert!(inv.tool_calls[1].err);
    }

    #[test]
    fn strip_system_tags_before_checking_system_only_markers() {
        let user_with_tagged_marker = "<system-reminder>[SYSTEM NOTIFICATION - NOT USER INPUT]\nWorkspace indexed.</system-reminder>\nFix the parser bug";
        assert_eq!(
            strip_system_context(user_with_tagged_marker),
            "Fix the parser bug"
        );

        let dump_after_tag = format!(
            "<system-reminder>{}</system-reminder>\nTarget file this round: MEMORY.md\nDump",
            "x".repeat(SYSTEM_MARKER_SCAN_LIMIT + 50)
        );
        assert_eq!(strip_system_context(&dump_after_tag), "");
    }

    #[test]
    fn strip_nested_unclosed_tag_mention_preserves_following_user_text() {
        let stdout = "<local-command-stdout>Hint: ignore <system-reminder> tags</local-command-stdout>\nExplain the output";
        assert_eq!(strip_system_context(stdout), "Explain the output");

        let notes = "<current_notes_content>Example <system-reminder> header</current_notes_content>\nRefactor trace.rs";
        assert_eq!(strip_system_context(notes), "Refactor trace.rs");
    }

    #[test]
    fn is_system_only_text_respects_marker_start_offset_across_utf8_boundary() {
        // 27 * 7 bytes ("路径/") + '\n' = 190 bytes (< 200); marker ends past byte 200.
        let near_limit = format!(
            "{}\nTarget file this round: MEMORY.md\n{}",
            "路径/".repeat(27),
            "正文".repeat(100)
        );
        assert!(is_system_only_text(&near_limit));
        assert_eq!(strip_system_context(&near_limit), "");

        let at_limit = format!(
            "{}Target file this round: example",
            "a".repeat(SYSTEM_MARKER_SCAN_LIMIT)
        );
        assert!(!is_system_only_text(&at_limit));
    }

    #[test]
    fn build_inventory_keeps_tagged_user_turns_and_skips_prefixed_system_dumps() {
        let pad = "r".repeat(SYSTEM_MARKER_SCAN_LIMIT + 40);
        let steps = format!(
            r#"[
            {{"step_id":1,"source":"user","timestamp":"2025-01-01T00:00:00Z",
              "message":"<system-reminder>[SYSTEM NOTIFICATION] ready</system-reminder>\nFirst question"}},
            {{"step_id":2,"source":"agent","timestamp":"2025-01-01T00:00:01Z","message":"First answer"}},
            {{"step_id":3,"source":"user","timestamp":"2025-01-01T00:00:02Z",
              "message":"<system-reminder>{pad}</system-reminder>\nTarget file this round: MEMORY.md\nDump"}},
            {{"step_id":4,"source":"user","timestamp":"2025-01-01T00:00:03Z",
              "message":"<current_notes_content>note <system-reminder></current_notes_content>\nFollow-up"}}
        ]"#
        );
        let inv = build_inventory(&traj(&steps));
        assert_eq!(inv.user_turns.len(), 2);
        assert_eq!(inv.user_turns[0].text, "First question");
        assert_eq!(inv.user_turns[1].text, "Follow-up");
        assert_eq!(inv.final_answer, "");
    }
}
