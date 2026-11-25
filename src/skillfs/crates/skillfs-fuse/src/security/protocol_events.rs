//! N3: SkillFS Protocol Event Log.
//!
//! Append-only JSONL event log for daemon reconcile, observability, and
//! troubleshooting. Independent from the existing audit JSONL
//! ([`super::audit`]) and security event stream ([`super::event_stream`]).
//!
//! The Ledger boundary follows §6 of `SKILL_LEDGER_SKILLFS_INTEGRATION_zh.md`;
//! N3 remains a SkillFS-local diagnostic protocol, not a Ledger identity source:
//!
//! ```json
//! {
//!   "schemaVersion": 1,
//!   "time": "2026-06-11T10:00:00.000Z",
//!   "skillDir": "/path/to/source/tianqi-weather",
//!   "skillName": "tianqi-weather",
//!   "eventKind": "write",
//!   "paths": ["SKILL.md"]
//! }
//! ```
//!
//! A4 adds an optional `reloadOutcome` field, present only when
//! `eventKind` is `"reload"`:
//!
//! ```json
//! {
//!   "schemaVersion": 1,
//!   "time": "2026-06-16T03:00:00.000Z",
//!   "skillDir": "/path/to/source/tianqi-weather",
//!   "skillName": "tianqi-weather",
//!   "eventKind": "reload",
//!   "paths": [],
//!   "reloadOutcome": "activation_updated"
//! }
//! ```
//!
//! Possible `reloadOutcome` values:
//! * `activation_updated` — resolver updated to a new target.
//! * `activation_unchanged` — daemon re-confirmed the same target.
//! * `activation_timeout` — poll timed out; current mapping kept.
//! * `activation_invalid_hidden` — invalid activation; fail-safe hidden.
//!
//! Three implementations:
//!
//! * [`NoopProtocolEventWriter`] — drops every event (default).
//! * [`JsonlProtocolEventWriter`] — best-effort append-only JSONL, one
//!   event per line, dedicated writer thread.
//! * [`InMemoryProtocolEventWriter`] — captures events in a `Vec` for
//!   tests.
//!
//! Write failures only warn; they never affect FUSE errno, notify
//! dispatch, or the active resolver.

use std::io::{BufWriter, Write};
use std::os::unix::fs::OpenOptionsExt;
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::mpsc::{SyncSender, TrySendError, sync_channel};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use parking_lot::Mutex;
use serde::Serialize;
use tracing::warn;

use crate::sys::openat_leaf;

pub const PROTOCOL_EVENT_SCHEMA_VERSION: u64 = 1;
pub const DEFAULT_PROTOCOL_EVENT_QUEUE_CAPACITY: usize = 256;

// ---------------------------------------------------------------------------
// ProtocolEvent
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct ProtocolEvent {
    pub schema_version: u64,
    pub time: String,
    pub skill_dir: String,
    pub skill_name: String,
    pub event_kind: String,
    pub paths: Vec<String>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub reload_outcome: Option<String>,
}

impl ProtocolEvent {
    pub fn new(
        skill_dir: impl Into<String>,
        skill_name: impl Into<String>,
        event_kind: impl Into<String>,
        paths: Vec<String>,
    ) -> Self {
        Self {
            schema_version: PROTOCOL_EVENT_SCHEMA_VERSION,
            time: current_rfc3339_utc(),
            skill_dir: skill_dir.into(),
            skill_name: skill_name.into(),
            event_kind: event_kind.into(),
            paths,
            reload_outcome: None,
        }
    }

    pub fn with_reload_outcome(
        skill_dir: impl Into<String>,
        skill_name: impl Into<String>,
        outcome: &str,
    ) -> Self {
        Self {
            schema_version: PROTOCOL_EVENT_SCHEMA_VERSION,
            time: current_rfc3339_utc(),
            skill_dir: skill_dir.into(),
            skill_name: skill_name.into(),
            event_kind: "reload".to_string(),
            paths: Vec::new(),
            reload_outcome: Some(outcome.to_string()),
        }
    }
}

// ---------------------------------------------------------------------------
// ProtocolEventWriter trait + implementations
// ---------------------------------------------------------------------------

pub trait ProtocolEventWriter: Send + Sync {
    fn emit(&self, event: &ProtocolEvent);
}

#[derive(Debug, Default, Clone, Copy)]
pub struct NoopProtocolEventWriter;

impl ProtocolEventWriter for NoopProtocolEventWriter {
    fn emit(&self, _event: &ProtocolEvent) {}
}

#[derive(Debug, Default)]
pub struct InMemoryProtocolEventWriter {
    events: Mutex<Vec<ProtocolEvent>>,
}

impl InMemoryProtocolEventWriter {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn events(&self) -> Vec<ProtocolEvent> {
        self.events.lock().clone()
    }

    pub fn len(&self) -> usize {
        self.events.lock().len()
    }

    pub fn is_empty(&self) -> bool {
        self.events.lock().is_empty()
    }
}

impl ProtocolEventWriter for InMemoryProtocolEventWriter {
    fn emit(&self, event: &ProtocolEvent) {
        self.events.lock().push(event.clone());
    }
}

pub struct JsonlProtocolEventWriter {
    tx: SyncSender<ProtocolEvent>,
    dropped: Arc<AtomicU64>,
    written: Arc<AtomicU64>,
    write_failures: Arc<AtomicU64>,
    _worker: Option<JoinHandle<()>>,
}

/// Open the protocol events log target without letting a symlinked parent,
/// symlinked leaf, or hard-linked source alias redirect writes into the
/// source tree.
///
/// Mirrors [`super::event_stream::open_events_log`]: each parent component is
/// pinned with `openat(O_PATH | O_DIRECTORY | O_NOFOLLOW)`, the leaf is
/// opened with `O_NOFOLLOW`, and [`refuse_aliased_target`] rejects targets
/// with `nlink() > 1`. Relative paths are refused so no unchecked `cwd`
/// component enters after validation.
fn open_protocol_events_log(path: &Path) -> std::io::Result<std::fs::File> {
    let parent = path
        .parent()
        .filter(|parent| parent.is_absolute())
        .ok_or_else(|| {
            std::io::Error::new(
                std::io::ErrorKind::InvalidInput,
                "protocol events log path must be absolute so every component can be checked",
            )
        })?;
    let leaf = path.file_name().ok_or_else(|| {
        std::io::Error::new(
            std::io::ErrorKind::InvalidInput,
            "protocol events log path has no file name component",
        )
    })?;

    let mut dir = std::fs::OpenOptions::new()
        .read(true)
        .custom_flags(libc::O_PATH | libc::O_DIRECTORY)
        .open("/")?;
    for component in parent.components() {
        match component {
            std::path::Component::RootDir => {}
            std::path::Component::Normal(name) => {
                dir = openat_leaf(
                    &dir,
                    name,
                    libc::O_PATH | libc::O_DIRECTORY | libc::O_NOFOLLOW,
                    0,
                )?;
            }
            other => {
                return Err(std::io::Error::new(
                    std::io::ErrorKind::InvalidInput,
                    format!(
                        "protocol events log path component {other:?} is not a plain directory name"
                    ),
                ));
            }
        }
    }

    let file = openat_leaf(
        &dir,
        leaf,
        libc::O_WRONLY | libc::O_CREAT | libc::O_APPEND | libc::O_NOFOLLOW,
        0o666,
    )?;
    refuse_aliased_target(&file, path)?;
    Ok(file)
}

/// Refuse a target whose inode has more than one hard link, since another name
/// inside the source tree could share the same inode.
fn refuse_aliased_target(file: &std::fs::File, path: &Path) -> std::io::Result<()> {
    use std::os::unix::fs::MetadataExt;

    let links = file.metadata()?.nlink();
    if links > 1 {
        return Err(std::io::Error::new(
            std::io::ErrorKind::PermissionDenied,
            format!(
                "refusing to open protocol events log '{}': it has {links} hard links, so a \
                 file inside the source tree could be this same inode and would receive these \
                 appends; only a singly linked target is proven disjoint from the source",
                path.display(),
            ),
        ));
    }
    Ok(())
}

impl JsonlProtocolEventWriter {
    pub fn new(path: impl Into<PathBuf>, queue_capacity: usize) -> std::io::Result<Self> {
        let path = path.into();
        let capacity = if queue_capacity == 0 {
            DEFAULT_PROTOCOL_EVENT_QUEUE_CAPACITY
        } else {
            queue_capacity
        };

        let file = open_protocol_events_log(&path)?;

        let (tx, rx) = sync_channel::<ProtocolEvent>(capacity);
        let dropped = Arc::new(AtomicU64::new(0));
        let written = Arc::new(AtomicU64::new(0));
        let write_failures = Arc::new(AtomicU64::new(0));

        let written_thread = written.clone();
        let write_failures_thread = write_failures.clone();
        let log_path = path.clone();
        let worker = std::thread::Builder::new()
            .name("skillfs-protocol-events".to_string())
            .spawn(move || {
                let mut writer = BufWriter::new(file);
                while let Ok(event) = rx.recv() {
                    let mut line = serialize_protocol_event(&event);
                    line.push('\n');
                    let res = writer
                        .write_all(line.as_bytes())
                        .and_then(|_| writer.flush());
                    match res {
                        Ok(()) => {
                            written_thread.fetch_add(1, Ordering::Relaxed);
                        }
                        Err(e) => {
                            write_failures_thread.fetch_add(1, Ordering::Relaxed);
                            warn!(
                                error = %e,
                                path = %log_path.display(),
                                "protocol events: write failed; event dropped"
                            );
                        }
                    }
                }
            })?;

        Ok(Self {
            tx,
            dropped,
            written,
            write_failures,
            _worker: Some(worker),
        })
    }

    pub fn dropped_count(&self) -> u64 {
        self.dropped.load(Ordering::Relaxed)
    }

    pub fn written_count(&self) -> u64 {
        self.written.load(Ordering::Relaxed)
    }

    pub fn write_failure_count(&self) -> u64 {
        self.write_failures.load(Ordering::Relaxed)
    }
}

impl ProtocolEventWriter for JsonlProtocolEventWriter {
    fn emit(&self, event: &ProtocolEvent) {
        match self.tx.try_send(event.clone()) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) | Err(TrySendError::Disconnected(_)) => {
                self.dropped.fetch_add(1, Ordering::Relaxed);
            }
        }
    }
}

impl Drop for JsonlProtocolEventWriter {
    fn drop(&mut self) {
        let _ = self._worker.take();
    }
}

// ---------------------------------------------------------------------------
// Serialization
// ---------------------------------------------------------------------------

pub fn serialize_protocol_event(event: &ProtocolEvent) -> String {
    serde_json::to_string(event)
        .unwrap_or_else(|_| String::from("{\"schemaVersion\":1,\"error\":\"unserializable\"}"))
}

/// Resolve a protocol events log path to the location a later `open` would act
/// on: follow every symlinked final component (bounded, like the kernel) and
/// canonicalize the rest, including dangling symlinks and bare relative
/// filenames resolved against the process current working directory.
pub fn resolve_protocol_events_path(path: &Path) -> std::io::Result<PathBuf> {
    let path = if path.is_absolute() {
        path.to_path_buf()
    } else {
        std::env::current_dir()?.join(path)
    };
    let mut current = path;
    for _ in 0..32 {
        match std::fs::symlink_metadata(&current) {
            // A real entry ends the chain: canonicalize still resolves any
            // symlinked parent components.
            Ok(meta) if !meta.file_type().is_symlink() => return current.canonicalize(),
            // Follow symlinks including dangling ones whose target open(O_CREAT)
            // would create, so containment validation sees the real target.
            Ok(_) => {
                let target = std::fs::read_link(&current)?;
                current = if target.is_absolute() {
                    target
                } else {
                    current
                        .parent()
                        .unwrap_or_else(|| Path::new("/"))
                        .join(target)
                };
            }
            // Nothing exists at the final component and its parent is
            // nameable: open(O_CREAT) would create the file right here.
            Err(e) if e.kind() == std::io::ErrorKind::NotFound => {
                let parent = current.parent().unwrap_or_else(|| Path::new("."));
                let parent_canonical = parent.canonicalize()?;
                let file_name = current.file_name().ok_or_else(|| {
                    std::io::Error::new(
                        std::io::ErrorKind::InvalidInput,
                        "protocol events path has no file name component",
                    )
                })?;
                return Ok(parent_canonical.join(file_name));
            }
            Err(e) => return Err(e),
        }
    }
    Err(std::io::Error::from_raw_os_error(libc::ELOOP))
}

// ---------------------------------------------------------------------------
// Source-tree guard
// ---------------------------------------------------------------------------

#[derive(Debug)]
pub enum ProtocolEventsPathError {
    /// The target could not be resolved, so no object can be approved for the
    /// writer to open.
    Unresolvable { log_path: PathBuf, reason: String },
    InsideSource {
        log_path: PathBuf,
        source_canonical: PathBuf,
        log_canonical: PathBuf,
    },
}

impl std::fmt::Display for ProtocolEventsPathError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Unresolvable { log_path, reason } => {
                write!(
                    f,
                    "--activation-events-log path '{}' cannot be resolved: {reason}",
                    log_path.display(),
                )
            }
            Self::InsideSource {
                log_path,
                source_canonical,
                log_canonical,
            } => {
                write!(
                    f,
                    "--activation-events-log path '{}' (canonical '{}') lies inside the \
                     SkillFS source root '{}'. Pick a location outside the source tree so \
                     event log writes cannot pollute skill workspaces or trigger scan noise.",
                    log_path.display(),
                    log_canonical.display(),
                    source_canonical.display(),
                )
            }
        }
    }
}

impl std::error::Error for ProtocolEventsPathError {}

/// Resolve a protocol events log path, reject targets inside the source tree,
/// and return the approved canonical path for [`JsonlProtocolEventWriter::new`].
pub fn resolve_protocol_events_path_outside_source(
    log_path: &Path,
    source_canonical: &Path,
) -> Result<PathBuf, ProtocolEventsPathError> {
    let log_canonical = resolve_protocol_events_path(log_path).map_err(|e| {
        ProtocolEventsPathError::Unresolvable {
            log_path: log_path.to_path_buf(),
            reason: e.to_string(),
        }
    })?;
    if log_canonical.starts_with(source_canonical) {
        return Err(ProtocolEventsPathError::InsideSource {
            log_path: log_path.to_path_buf(),
            source_canonical: source_canonical.to_path_buf(),
            log_canonical,
        });
    }
    Ok(log_canonical)
}

/// Reject a protocol events log path that resolves inside the source tree.
///
/// Mirrors [`super::audit::AuditRuntimeConfig::validate_audit_path_outside_source`]:
/// the log must not land in the skill workspace where daemon scan/reconcile
/// would pick it up as skill content. Resolution failures are deferred to
/// the file-open step.
pub fn validate_protocol_events_path_outside_source(
    log_path: &Path,
    source_canonical: &Path,
) -> Result<(), ProtocolEventsPathError> {
    let log_canonical = match resolve_protocol_events_path(log_path) {
        Ok(p) => p,
        Err(_) => return Ok(()),
    };
    if log_canonical.starts_with(source_canonical) {
        return Err(ProtocolEventsPathError::InsideSource {
            log_path: log_path.to_path_buf(),
            source_canonical: source_canonical.to_path_buf(),
            log_canonical,
        });
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Timestamp
// ---------------------------------------------------------------------------

fn current_rfc3339_utc() -> String {
    let now = SystemTime::now();
    let epoch = match now.duration_since(UNIX_EPOCH) {
        Ok(d) => d,
        Err(_) => std::time::Duration::from_secs(0),
    };
    let total_secs = epoch.as_secs() as i64;
    let millis = epoch.subsec_millis();
    let (year, month, day, hour, minute, second) = secs_to_civil(total_secs);
    format!(
        "{:04}-{:02}-{:02}T{:02}:{:02}:{:02}.{:03}Z",
        year, month, day, hour, minute, second, millis
    )
}

fn secs_to_civil(secs: i64) -> (i32, u32, u32, u32, u32, u32) {
    let days = secs.div_euclid(86_400);
    let secs_of_day = secs.rem_euclid(86_400) as u32;
    let hour = secs_of_day / 3600;
    let minute = (secs_of_day % 3600) / 60;
    let second = secs_of_day % 60;

    let z = days + 719_468;
    let era = if z >= 0 { z } else { z - 146_096 } / 146_097;
    let doe = (z - era * 146_097) as u32;
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let y = yoe as i32 + (era as i32) * 400;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let year = if m <= 2 { y + 1 } else { y };
    (year, m, d, hour, minute, second)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn protocol_event_json_shape() {
        let event = ProtocolEvent::new(
            "/srv/skills/tianqi-weather",
            "tianqi-weather",
            "write",
            vec!["SKILL.md".to_string()],
        );
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["schemaVersion"], 1);
        assert!(json["time"].as_str().unwrap().ends_with('Z'));
        assert_eq!(json["skillDir"], "/srv/skills/tianqi-weather");
        assert_eq!(json["skillName"], "tianqi-weather");
        assert_eq!(json["eventKind"], "write");
        assert_eq!(json["paths"], serde_json::json!(["SKILL.md"]));
    }

    #[test]
    fn protocol_event_without_reload_outcome_omits_field() {
        let event = ProtocolEvent::new("/a", "a", "write", vec![]);
        let line = serialize_protocol_event(&event);
        assert!(
            !line.contains("reloadOutcome"),
            "reload_outcome=None must not appear in JSON: {line}"
        );
    }

    #[test]
    fn protocol_event_with_reload_outcome_includes_field() {
        let event = ProtocolEvent::with_reload_outcome("/a/alpha", "alpha", "activation_updated");
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["eventKind"], "reload");
        assert_eq!(json["reloadOutcome"], "activation_updated");
        assert_eq!(json["paths"], serde_json::json!([]));
        assert_eq!(json["schemaVersion"], 1);
        assert!(json["time"].as_str().unwrap().ends_with('Z'));
    }

    #[test]
    fn protocol_event_reload_outcome_all_labels() {
        for label in &[
            "activation_updated",
            "activation_unchanged",
            "activation_timeout",
            "activation_invalid_hidden",
        ] {
            let event = ProtocolEvent::with_reload_outcome("/a/alpha", "alpha", label);
            assert_eq!(event.reload_outcome.as_deref(), Some(*label));
            let line = serialize_protocol_event(&event);
            assert!(
                line.contains(label),
                "label {label} must appear in serialized event: {line}"
            );
        }
    }

    #[test]
    fn protocol_event_json_field_names_are_camel_case() {
        let event = ProtocolEvent::new("/a", "a", "mkdir", vec![]);
        let line = serialize_protocol_event(&event);
        assert!(line.contains("\"schemaVersion\""), "line={line}");
        assert!(line.contains("\"skillDir\""), "line={line}");
        assert!(line.contains("\"skillName\""), "line={line}");
        assert!(line.contains("\"eventKind\""), "line={line}");
        assert!(
            !line.contains("\"schema_version\""),
            "snake_case leaked: {line}"
        );
    }

    #[test]
    fn protocol_event_time_is_rfc3339_utc() {
        let event = ProtocolEvent::new("/a", "a", "write", vec![]);
        let t = &event.time;
        assert_eq!(t.len(), 24, "expected 24-char RFC3339 timestamp, got {t:?}");
        assert_eq!(t.chars().nth(4), Some('-'));
        assert_eq!(t.chars().nth(7), Some('-'));
        assert_eq!(t.chars().nth(10), Some('T'));
        assert_eq!(t.chars().nth(13), Some(':'));
        assert_eq!(t.chars().nth(16), Some(':'));
        assert_eq!(t.chars().nth(19), Some('.'));
        assert!(t.ends_with('Z'));
    }

    #[test]
    fn protocol_event_empty_paths() {
        let event = ProtocolEvent::new("/a", "a", "mkdir", vec![]);
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(json["paths"], serde_json::json!([]));
    }

    #[test]
    fn protocol_event_multiple_paths() {
        let event = ProtocolEvent::new(
            "/a",
            "a",
            "write",
            vec!["SKILL.md".to_string(), "scripts/run.sh".to_string()],
        );
        let json = serde_json::to_value(&event).unwrap();
        assert_eq!(
            json["paths"],
            serde_json::json!(["SKILL.md", "scripts/run.sh"])
        );
    }

    #[test]
    fn noop_writer_drops_events_silently() {
        let w = NoopProtocolEventWriter;
        for _ in 0..16 {
            w.emit(&ProtocolEvent::new("/a", "a", "write", vec![]));
        }
    }

    #[test]
    fn in_memory_writer_records_events() {
        let w = InMemoryProtocolEventWriter::new();
        assert!(w.is_empty());
        w.emit(&ProtocolEvent::new(
            "/a/alpha",
            "alpha",
            "write",
            vec!["SKILL.md".to_string()],
        ));
        w.emit(&ProtocolEvent::new("/a/beta", "beta", "mkdir", vec![]));
        assert_eq!(w.len(), 2);
        let events = w.events();
        assert_eq!(events[0].skill_name, "alpha");
        assert_eq!(events[0].event_kind, "write");
        assert_eq!(events[0].paths, vec!["SKILL.md"]);
        assert_eq!(events[1].skill_name, "beta");
        assert_eq!(events[1].event_kind, "mkdir");
        assert!(events[1].paths.is_empty());
    }

    #[test]
    fn jsonl_writer_round_trips_through_disk() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("protocol-events.jsonl");
        {
            let writer = JsonlProtocolEventWriter::new(&path, 0).expect("open writer");
            writer.emit(&ProtocolEvent::new(
                "/a/alpha",
                "alpha",
                "write",
                vec!["SKILL.md".to_string()],
            ));
            writer.emit(&ProtocolEvent::new(
                "/a/beta",
                "beta",
                "rename",
                vec!["old.txt".to_string(), "new.txt".to_string()],
            ));
            std::thread::sleep(std::time::Duration::from_millis(150));
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
        let body = std::fs::read_to_string(&path).expect("read protocol events");
        let lines: Vec<&str> = body.lines().collect();
        assert_eq!(lines.len(), 2, "expected two JSONL lines, got {body:?}");
        for line in &lines {
            let parsed: serde_json::Value = serde_json::from_str(line).expect("valid JSON");
            assert_eq!(parsed["schemaVersion"], 1);
            assert!(parsed["time"].as_str().unwrap().ends_with('Z'));
            assert!(parsed.get("skillDir").is_some());
            assert!(parsed.get("skillName").is_some());
            assert!(parsed.get("eventKind").is_some());
            assert!(parsed.get("paths").is_some());
        }
    }

    #[test]
    fn jsonl_writer_open_failure_surfaces_io_error() {
        let dir = tempfile::tempdir().unwrap();
        match JsonlProtocolEventWriter::new(dir.path(), 0) {
            Ok(_) => panic!("expected open to fail when target is a directory"),
            Err(err) => {
                assert!(
                    err.kind() == std::io::ErrorKind::IsADirectory
                        || err.raw_os_error() == Some(libc::EISDIR),
                    "expected EISDIR-ish, got {err:?}"
                );
            }
        }
    }

    #[test]
    fn resolve_protocol_events_path_uses_parent_when_file_missing() {
        let dir = tempfile::tempdir().unwrap();
        let target = dir.path().join("does-not-exist-yet.jsonl");
        let resolved = resolve_protocol_events_path(&target).expect("resolve");
        assert!(resolved.is_absolute());
        assert!(resolved.ends_with("does-not-exist-yet.jsonl"));
    }

    #[test]
    fn secs_to_civil_pins_known_values() {
        assert_eq!(secs_to_civil(0), (1970, 1, 1, 0, 0, 0));
        assert_eq!(secs_to_civil(951_825_600), (2000, 2, 29, 12, 0, 0));
        assert_eq!(secs_to_civil(1_779_982_212), (2026, 5, 28, 15, 30, 12));
    }

    #[test]
    fn validate_path_outside_source_accepts_disjoint_directory() {
        let source = tempfile::tempdir().unwrap();
        let log_dir = tempfile::tempdir().unwrap();
        let log_path = log_dir.path().join("events.jsonl");
        let source_canonical = source.path().canonicalize().unwrap();
        assert!(validate_protocol_events_path_outside_source(&log_path, &source_canonical).is_ok());
    }

    #[test]
    fn validate_path_outside_source_rejects_path_inside_source() {
        let source = tempfile::tempdir().unwrap();
        let log_path = source.path().join("alpha/events.jsonl");
        std::fs::create_dir_all(source.path().join("alpha")).unwrap();
        let source_canonical = source.path().canonicalize().unwrap();
        let result = validate_protocol_events_path_outside_source(&log_path, &source_canonical);
        assert!(
            matches!(result, Err(ProtocolEventsPathError::InsideSource { .. })),
            "path inside source must be rejected: {result:?}"
        );
        let err = result.unwrap_err();
        assert!(
            err.to_string().contains("--activation-events-log"),
            "error message should name the flag: {err}"
        );
    }

    #[test]
    fn validate_path_outside_source_defers_unresolvable_paths() {
        let source = tempfile::tempdir().unwrap();
        let source_canonical = source.path().canonicalize().unwrap();
        let log_path = PathBuf::from("/nonexistent-parent-9999/events.jsonl");
        assert!(
            validate_protocol_events_path_outside_source(&log_path, &source_canonical).is_ok(),
            "unresolvable paths should defer to file-open step"
        );
    }

    #[test]
    fn resolve_protocol_events_path_follows_dangling_symlink() {
        let dir = tempfile::tempdir().unwrap();
        let nested = dir.path().join("nested");
        std::fs::create_dir(&nested).unwrap();
        let link = dir.path().join("link.jsonl");
        let target = nested.join("target.jsonl");
        std::os::unix::fs::symlink(&target, &link).unwrap();
        let resolved = resolve_protocol_events_path(&link).expect("resolve dangling symlink");
        let expected_parent = nested.canonicalize().unwrap();
        assert_eq!(
            resolved,
            expected_parent.join("target.jsonl"),
            "dangling symlink must resolve to its target's canonical location"
        );
    }

    #[test]
    fn resolve_protocol_events_path_handles_bare_relative_filename() {
        let resolved =
            resolve_protocol_events_path(Path::new("relative-protocol-events-resolve.jsonl"))
                .expect("bare relative filename must resolve against cwd");
        let cwd_canonical = std::env::current_dir().unwrap().canonicalize().unwrap();
        assert_eq!(
            resolved,
            cwd_canonical.join("relative-protocol-events-resolve.jsonl")
        );

        let err = validate_protocol_events_path_outside_source(
            Path::new("relative-protocol-events-resolve.jsonl"),
            &cwd_canonical,
        )
        .expect_err("bare relative path inside cwd/source must be rejected");
        assert!(matches!(err, ProtocolEventsPathError::InsideSource { .. }));
    }

    #[test]
    fn resolve_protocol_events_path_rejects_symlink_loop() {
        let dir = tempfile::tempdir().unwrap();
        let a = dir.path().join("a.jsonl");
        let b = dir.path().join("b.jsonl");
        std::os::unix::fs::symlink(&b, &a).unwrap();
        std::os::unix::fs::symlink(&a, &b).unwrap();
        let err = resolve_protocol_events_path(&a).expect_err("symlink loop must fail");
        assert_eq!(err.raw_os_error(), Some(libc::ELOOP));
    }

    #[test]
    fn validate_path_outside_source_rejects_dangling_symlink_into_source() {
        let source = tempfile::tempdir().unwrap();
        let source_canonical = source.path().canonicalize().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let link = outside.path().join("events.jsonl");
        let planted = source.path().join("planted-events.jsonl");
        std::os::unix::fs::symlink(&planted, &link).unwrap();

        let err = validate_protocol_events_path_outside_source(&link, &source_canonical)
            .expect_err("dangling symlink into source must be rejected");
        assert!(
            matches!(err, ProtocolEventsPathError::InsideSource { .. }),
            "expected InsideSource, got {err:?}"
        );
        assert!(!planted.exists(), "rejected path must not create target");
    }

    #[test]
    fn jsonl_writer_refuses_dangling_symlink_at_open() {
        let source = tempfile::tempdir().unwrap();
        let outside = tempfile::tempdir().unwrap();
        let link = outside.path().join("events.jsonl");
        let planted = source.path().join("planted-events.jsonl");
        std::os::unix::fs::symlink(&planted, &link).unwrap();

        let err = match JsonlProtocolEventWriter::new(&link, 0) {
            Ok(_) => panic!("dangling symlink leaf must be refused at open"),
            Err(e) => e,
        };
        assert_eq!(err.raw_os_error(), Some(libc::ELOOP));
        assert!(
            !planted.exists(),
            "refused open must not create target inside source"
        );
    }

    #[test]
    fn jsonl_writer_refuses_symlinked_leaf_at_open() {
        let source = tempfile::tempdir().unwrap();
        let existing = source.path().join("existing.jsonl");
        std::fs::write(&existing, "untouched\n").unwrap();
        let outside = tempfile::tempdir().unwrap();
        let link = outside.path().join("events.jsonl");
        std::os::unix::fs::symlink(&existing, &link).unwrap();

        let err = match JsonlProtocolEventWriter::new(&link, 0) {
            Ok(_) => panic!("symlinked leaf must be refused at open"),
            Err(e) => e,
        };
        assert_eq!(err.raw_os_error(), Some(libc::ELOOP));
        assert_eq!(std::fs::read_to_string(&existing).unwrap(), "untouched\n");
    }

    #[test]
    fn jsonl_writer_refuses_symlinked_parent_at_open() {
        let source = tempfile::tempdir().unwrap();
        let outside_dir = tempfile::tempdir().unwrap();
        let outside = outside_dir.path().canonicalize().unwrap();
        let log_dir = outside.join("events-dir");
        std::os::unix::fs::symlink(source.path(), &log_dir).unwrap();
        let log_path = log_dir.join("events.jsonl");

        let err = match JsonlProtocolEventWriter::new(&log_path, 0) {
            Ok(_) => panic!("symlinked parent must be refused at open"),
            Err(e) => e,
        };
        assert!(
            matches!(err.raw_os_error(), Some(libc::ELOOP | libc::ENOTDIR)),
            "expected ELOOP/ENOTDIR, got {err:?}"
        );
        assert!(
            !source.path().join("events.jsonl").exists(),
            "refused open must not create log inside source"
        );
    }

    #[test]
    fn jsonl_writer_refuses_hard_link_to_source_file() {
        let source = tempfile::tempdir().unwrap();
        let source_canonical = source.path().canonicalize().unwrap();
        let manifest = source.path().join("SKILL.md");
        std::fs::write(&manifest, "# trusted manifest\n").unwrap();
        let outside = tempfile::tempdir().unwrap();
        let link = outside.path().join("events.jsonl");
        std::fs::hard_link(&manifest, &link).unwrap();

        assert!(validate_protocol_events_path_outside_source(&link, &source_canonical).is_ok());

        let err = match JsonlProtocolEventWriter::new(&link, 0) {
            Ok(_) => panic!("hard-linked target must be refused at open"),
            Err(e) => e,
        };
        assert!(
            err.to_string().contains("hard links"),
            "expected hard-link refusal, got {err}"
        );
        assert_eq!(
            std::fs::read(&manifest).unwrap(),
            b"# trusted manifest\n",
            "refused open must not modify aliased source file"
        );
    }

    #[test]
    fn jsonl_writer_refuses_relative_path() {
        let err = match JsonlProtocolEventWriter::new(
            Path::new("relative-protocol-events-open.jsonl"),
            0,
        ) {
            Ok(_) => panic!("relative path must be refused at open"),
            Err(e) => e,
        };
        assert_eq!(err.kind(), std::io::ErrorKind::InvalidInput);
    }

    #[test]
    fn jsonl_writer_works_under_write_only_parent() {
        use std::os::unix::fs::PermissionsExt;

        let outside_dir = tempfile::tempdir().unwrap();
        let outside = outside_dir.path().canonicalize().unwrap();
        let log_dir = outside.join("write-only");
        std::fs::create_dir(&log_dir).unwrap();
        std::fs::set_permissions(&log_dir, std::fs::Permissions::from_mode(0o300)).unwrap();
        let log_path = log_dir.join("protocol-events.jsonl");

        {
            let writer = JsonlProtocolEventWriter::new(&log_path, 0)
                .expect("write-only parent must allow opening protocol events log");
            writer.emit(&ProtocolEvent::new("/a/alpha", "alpha", "write", vec![]));
            std::thread::sleep(std::time::Duration::from_millis(150));
        }
        std::thread::sleep(std::time::Duration::from_millis(100));

        let body = std::fs::read_to_string(&log_path).expect("read protocol events log");
        assert_eq!(body.lines().count(), 1, "expected 1 line, got {body:?}");

        std::fs::set_permissions(&log_dir, std::fs::Permissions::from_mode(0o700)).unwrap();
    }

    #[test]
    fn resolve_outside_source_to_writer_supports_relative_and_symlinked_parent_while_rejecting_source_dangling_link()
     {
        struct RemoveOnDrop(PathBuf);
        impl Drop for RemoveOnDrop {
            fn drop(&mut self) {
                let _ = std::fs::remove_file(&self.0);
            }
        }

        let source = tempfile::tempdir().unwrap();
        let source_canonical = source.path().canonicalize().unwrap();

        // 1. Bare relative filename resolved outside source opens and writes.
        let rel_name = format!("tmp-protocol-events-{}.jsonl", std::process::id());
        let resolved_rel =
            resolve_protocol_events_path_outside_source(Path::new(&rel_name), &source_canonical)
                .expect("relative path outside source must resolve");
        let _cleanup = RemoveOnDrop(resolved_rel.clone());
        {
            let writer = JsonlProtocolEventWriter::new(&resolved_rel, 0)
                .expect("resolved relative path must open");
            writer.emit(&ProtocolEvent::new("/a/alpha", "alpha", "write", vec![]));
            std::thread::sleep(std::time::Duration::from_millis(150));
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(resolved_rel.exists(), "relative log file should be created");

        // 2. External path through a symlinked parent resolves and opens.
        let outside = tempfile::tempdir().unwrap();
        let real_dir = outside.path().join("real-dir");
        std::fs::create_dir(&real_dir).unwrap();
        let link_dir = outside.path().join("link-dir");
        std::os::unix::fs::symlink(&real_dir, &link_dir).unwrap();
        let through_symlink = link_dir.join("events.jsonl");
        let resolved_ext =
            resolve_protocol_events_path_outside_source(&through_symlink, &source_canonical)
                .expect("external symlinked parent must resolve");
        assert_eq!(
            resolved_ext,
            real_dir.canonicalize().unwrap().join("events.jsonl")
        );
        {
            let writer = JsonlProtocolEventWriter::new(&resolved_ext, 0)
                .expect("resolved external symlinked-parent path must open");
            writer.emit(&ProtocolEvent::new("/a/beta", "beta", "write", vec![]));
            std::thread::sleep(std::time::Duration::from_millis(150));
        }
        std::thread::sleep(std::time::Duration::from_millis(100));
        assert!(
            real_dir.join("events.jsonl").exists(),
            "resolved log must land in real target directory"
        );

        // 3. Dangling symlink into source is rejected before opening.
        let planted = source.path().join("planted-events.jsonl");
        let bad_link = outside.path().join("dangling-to-source.jsonl");
        std::os::unix::fs::symlink(&planted, &bad_link).unwrap();
        let err = resolve_protocol_events_path_outside_source(&bad_link, &source_canonical)
            .expect_err("dangling symlink into source must be rejected");
        assert!(matches!(err, ProtocolEventsPathError::InsideSource { .. }));
        assert!(!planted.exists(), "rejected symlink must not create target");
    }
}
