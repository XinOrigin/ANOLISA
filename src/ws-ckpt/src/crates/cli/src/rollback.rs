//! JSON boundary for the rollback CLI and the daemon's guarded protocol.

use std::io::{self, IsTerminal, Read};

use anyhow::{Context, Result};
use serde_json::{json, Value};
use ws_ckpt_common::{
    ErrorCode, GuardedRollbackEvidenceV2, GuardedRollbackOutcomeV2, Request, Response,
    WorkspaceGenerationTokenV2, GUARDED_CHECKPOINT_PROTOCOL_VERSION_V2,
};

use super::{configured_peer_uid, effective_uid, send_request, IpcFailure};

// CLI JSON contract: docs/design/guarded-checkpoint-v2.md.
const SCHEMA: &str = "ws-ckpt.rollback.v1";
const MAX_CONFIRMATION_BYTES: u64 = 4096;

#[derive(Clone, Copy)]
pub(super) enum Mode {
    Check,
    Preview,
    Execute,
    Result,
}

pub(super) fn succeeded(output: &Value) -> bool {
    matches!(
        output["status"].as_str(),
        Some("compatible" | "preview" | "succeeded" | "result")
    )
}

pub(super) async fn run_json(
    workspace: String,
    snapshot: Option<String>,
    num_ancestors: Option<u32>,
    guarded: bool,
    mode: Mode,
) -> Value {
    let result = if guarded {
        guarded_json(workspace, snapshot, mode).await
    } else {
        ordinary_json(workspace, snapshot, num_ancestors, mode).await
    };
    let mut output = result.unwrap_or_else(|error| {
        let status = if matches!(mode, Mode::Execute) {
            "known_no_effect"
        } else {
            "error"
        };
        failure(status, "PreflightFailed", &format!("{error:#}"))
    });
    output["schema"] = json!(SCHEMA);
    output
}

fn failure(status: &str, code: &str, message: &str) -> Value {
    json!({"status": status, "code": code, "message": message})
}

fn ipc_failure(error: IpcFailure, mode: Mode) -> Value {
    let status = match (mode, &error) {
        (Mode::Execute, IpcFailure::PossiblyApplied(_)) => "unknown",
        (Mode::Execute, _) => "known_no_effect",
        _ => "error",
    };
    failure(status, "IpcError", &format!("{:#}", error.into_error()))
}

async fn ordinary_json(
    workspace: String,
    to: Option<String>,
    num_ancestors: Option<u32>,
    mode: Mode,
) -> Result<Value> {
    anyhow::ensure!(
        to.is_some() || num_ancestors.is_some(),
        "either --snapshot/-s or --num-ancestors/-n must be specified"
    );
    let request = if matches!(mode, Mode::Preview) {
        Request::RollbackPreview {
            workspace,
            to,
            num_ancestors,
        }
    } else {
        Request::Rollback {
            workspace,
            to,
            num_ancestors,
        }
    };
    let output = match (mode, send_request(&request, configured_peer_uid()?).await) {
        (Mode::Preview, Ok(Response::RollbackPreviewOk { to, changes })) => {
            json!({"status": "preview", "snapshot_id": to, "changes": changes})
        }
        (Mode::Execute, Ok(Response::RollbackOk { from, to })) => {
            json!({"status": "succeeded", "workspace": from, "snapshot_id": to})
        }
        (_, Ok(Response::Error { code, message })) => {
            let status = match (mode, &code) {
                (
                    Mode::Execute,
                    ErrorCode::WorkspaceNotFound
                    | ErrorCode::SnapshotNotFound
                    | ErrorCode::InvalidPath
                    | ErrorCode::CwdOccupied
                    | ErrorCode::CwdScanFailed
                    | ErrorCode::DiskSpaceInsufficient,
                ) => "known_no_effect",
                (Mode::Execute, _) => "unknown",
                _ => "error",
            };
            failure(status, &format!("{code:?}"), &message)
        }
        (_, Err(error)) => ipc_failure(error, mode),
        _ => unexpected(mode),
    };
    Ok(output)
}

async fn workspace_identity(path: &str, uid: u32) -> Result<(String, WorkspaceGenerationTokenV2)> {
    match send_request(
        &Request::WorkspaceIdentityV2 {
            registration_path: path.to_owned(),
        },
        Some(uid),
    )
    .await
    .map_err(IpcFailure::into_error)?
    {
        Response::WorkspaceIdentityV2Ok {
            protocol_version,
            ws_id,
            registered_path,
            generation,
        } => {
            anyhow::ensure!(
                protocol_version == GUARDED_CHECKPOINT_PROTOCOL_VERSION_V2,
                "unsupported guarded rollback protocol version {protocol_version}"
            );
            anyhow::ensure!(registered_path == path, "workspace registration mismatch");
            Ok((ws_id, generation))
        }
        Response::Error { code, message } => anyhow::bail!("{code:?}: {message}"),
        _ => anyhow::bail!("unexpected response while checking guarded rollback protocol"),
    }
}

async fn guarded_json(path: String, snapshot: Option<String>, mode: Mode) -> Result<Value> {
    let uid = effective_uid()?;
    Ok(match mode {
        Mode::Check => {
            let (ws_id, _) = workspace_identity(&path, uid).await?;
            json!({"status": "compatible", "ws_id": ws_id, "registered_path": path, "diff_json": true})
        }
        Mode::Preview => {
            let snapshot = snapshot.context("guarded preview requires --snapshot/-s")?;
            let (ws_id, generation) = workspace_identity(&path, uid).await?;
            let request = Request::GuardedRollbackPreviewV2 {
                registered_path: path.clone(),
                ws_id: ws_id.clone(),
                expected_generation: generation,
                target_snapshot_id: snapshot.clone(),
            };
            match send_request(&request, Some(uid)).await {
                Ok(Response::GuardedRollbackPreviewV2Ok {
                    protocol_version,
                    registered_path,
                    ws_id: actual_ws_id,
                    generation: actual_generation,
                    target_snapshot_id,
                    diff_digest,
                    changes,
                    caller_uid,
                }) if protocol_version == GUARDED_CHECKPOINT_PROTOCOL_VERSION_V2
                    && registered_path == path
                    && actual_ws_id == ws_id
                    && actual_generation == generation
                    && target_snapshot_id == snapshot
                    && caller_uid == uid =>
                {
                    json!({
                        "status": "preview", "ws_id": ws_id, "registered_path": path,
                        "snapshot_id": snapshot, "generation": generation, "diff_digest": diff_digest,
                        "changes": changes
                    })
                }
                response => guarded_reply(response, mode),
            }
        }
        Mode::Execute => {
            let snapshot = snapshot.context("guarded rollback requires --snapshot/-s")?;
            let confirmation = Confirmation::from_stdin()?;
            let (ws_id, _) = workspace_identity(&path, uid).await?;
            anyhow::ensure!(ws_id == confirmation.ws_id, "workspace ID mismatch");
            let request = Request::GuardedRollbackV2 {
                registered_path: path.clone(),
                ws_id,
                expected_generation: confirmation.generation,
                target_snapshot_id: snapshot.clone(),
                expected_diff_digest: confirmation.diff_digest,
                operation_id: confirmation.operation_id.clone(),
                operation_digest: confirmation.operation_digest,
            };
            match send_request(&request, Some(uid)).await {
                Ok(Response::GuardedRollbackV2Ok { evidence })
                    if confirmation.matches(&evidence, uid)
                        && evidence.registered_path == path
                        && evidence.target_snapshot_id == snapshot
                        && matches!(
                            evidence.outcome,
                            GuardedRollbackOutcomeV2::Succeeded { .. }
                        ) =>
                {
                    json!({"status": "succeeded", "evidence": evidence})
                }
                Ok(Response::GuardedRollbackV2Uncertain { evidence })
                    if confirmation.matches(&evidence, uid)
                        && evidence.registered_path == path
                        && evidence.target_snapshot_id == snapshot
                        && matches!(
                            evidence.outcome,
                            GuardedRollbackOutcomeV2::Started
                                | GuardedRollbackOutcomeV2::Unknown { .. }
                        ) =>
                {
                    let message = match &evidence.outcome {
                        GuardedRollbackOutcomeV2::Unknown { reason } => reason,
                        _ => "daemon could not confirm the rollback outcome",
                    };
                    failure("unknown", "UncertainOutcome", message)
                }
                response => guarded_reply(response, mode),
            }
        }
        Mode::Result => {
            let confirmation = Confirmation::from_stdin()?;
            anyhow::ensure!(path == confirmation.ws_id, "workspace ID mismatch");
            let request = Request::GuardedRollbackEvidenceV2 {
                ws_id: path,
                operation_id: confirmation.operation_id.clone(),
                operation_digest: confirmation.operation_digest,
            };
            match send_request(&request, Some(uid)).await {
                Ok(Response::GuardedRollbackEvidenceV2Ok { evidence })
                    if evidence
                        .as_ref()
                        .is_none_or(|evidence| confirmation.matches(evidence, uid)) =>
                {
                    json!({"status": "result", "evidence": evidence})
                }
                response => guarded_reply(response, mode),
            }
        }
    })
}

fn guarded_reply(response: std::result::Result<Response, IpcFailure>, mode: Mode) -> Value {
    match response {
        Ok(Response::GuardedRollbackV2Rejected { code, message }) => {
            let status = if matches!(mode, Mode::Execute) {
                "known_no_effect"
            } else {
                "error"
            };
            failure(status, &format!("{code:?}"), &message)
        }
        Ok(Response::Error { code, message }) => {
            let status = if matches!(mode, Mode::Execute) {
                "unknown"
            } else {
                "error"
            };
            failure(status, &format!("{code:?}"), &message)
        }
        Err(error) => ipc_failure(error, mode),
        _ => unexpected(mode),
    }
}

fn unexpected(mode: Mode) -> Value {
    let status = if matches!(mode, Mode::Execute) {
        "unknown"
    } else {
        "error"
    };
    failure(
        status,
        "UnexpectedResponse",
        "unexpected daemon rollback response",
    )
}

struct Confirmation {
    ws_id: String,
    generation: WorkspaceGenerationTokenV2,
    diff_digest: [u8; 32],
    operation_id: String,
    operation_digest: [u8; 32],
}

impl Confirmation {
    fn from_stdin() -> Result<Self> {
        let stdin = io::stdin();
        anyhow::ensure!(
            !stdin.is_terminal(),
            "read the guarded confirmation JSON from stdin"
        );
        let mut input = Vec::new();
        stdin
            .take(MAX_CONFIRMATION_BYTES + 1)
            .read_to_end(&mut input)?;
        anyhow::ensure!(
            input.len() as u64 <= MAX_CONFIRMATION_BYTES,
            "guarded confirmation exceeds {MAX_CONFIRMATION_BYTES} bytes"
        );
        let mut fields: serde_json::Map<String, Value> =
            serde_json::from_slice(&input).context("invalid guarded confirmation JSON object")?;
        Ok(Self {
            ws_id: serde_json::from_value(fields.remove("ws_id").unwrap_or_default())
                .context("invalid ws_id")?,
            generation: serde_json::from_value(fields.remove("generation").unwrap_or_default())
                .context("invalid generation")?,
            diff_digest: serde_json::from_value(fields.remove("diff_digest").unwrap_or_default())
                .context("invalid diff_digest")?,
            operation_id: serde_json::from_value(fields.remove("operation_id").unwrap_or_default())
                .context("invalid operation_id")?,
            operation_digest: serde_json::from_value(
                fields.remove("operation_digest").unwrap_or_default(),
            )
            .context("invalid operation_digest")?,
        })
    }

    fn matches(&self, evidence: &GuardedRollbackEvidenceV2, uid: u32) -> bool {
        evidence.ws_id == self.ws_id
            && evidence.expected_generation == self.generation
            && evidence.expected_diff_digest == self.diff_digest
            && evidence.operation_id == self.operation_id
            && evidence.operation_digest == self.operation_digest
            && evidence.caller_uid == uid
    }
}

#[cfg(test)]
mod tests {
    use std::io::{Read, Write};
    use std::os::unix::net::UnixListener;
    use std::path::PathBuf;
    use std::process::{Command, Stdio};
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::time::{Duration, Instant};

    use serde_json::{json, Value};
    use ws_ckpt_common::{
        decode_payload, encode_frame, ErrorCode, GuardedRollbackEvidenceV2,
        GuardedRollbackOutcomeV2, Request, Response, WorkspaceGenerationTokenV2,
        GUARDED_CHECKPOINT_PROTOCOL_VERSION_V2,
    };

    use super::{effective_uid, run_json, Mode};

    static NEXT_SOCKET_ID: AtomicUsize = AtomicUsize::new(0);
    const CHILD_CASE_ENV: &str = "WS_CKPT_ROLLBACK_UNIT_CHILD_CASE";

    struct TestSocketDir(PathBuf);

    impl TestSocketDir {
        fn new() -> Self {
            let path = std::env::temp_dir().join(format!(
                "ws-ckpt-rollback-unit-{}-{}",
                std::process::id(),
                NEXT_SOCKET_ID.fetch_add(1, Ordering::Relaxed)
            ));
            std::fs::create_dir(&path).unwrap();
            Self(path)
        }
    }

    impl Drop for TestSocketDir {
        fn drop(&mut self) {
            let _ = std::fs::remove_dir_all(&self.0);
        }
    }

    fn spawn_mock_peer(
        responses: Vec<Response>,
    ) -> (
        TestSocketDir,
        PathBuf,
        std::thread::JoinHandle<Vec<Request>>,
    ) {
        let dir = TestSocketDir::new();
        let socket = dir.0.join("daemon.sock");
        let listener = UnixListener::bind(&socket).unwrap();
        listener.set_nonblocking(true).unwrap();
        let handle = std::thread::spawn(move || {
            let deadline = Instant::now() + Duration::from_secs(10);
            let mut requests = Vec::with_capacity(responses.len());
            for response in responses {
                let mut stream = loop {
                    match listener.accept() {
                        Ok((stream, _)) => break stream,
                        Err(e) if e.kind() == std::io::ErrorKind::WouldBlock => {
                            assert!(
                                Instant::now() < deadline,
                                "mock daemon timed out waiting for CLI"
                            );
                            std::thread::sleep(Duration::from_millis(5));
                        }
                        Err(e) => panic!("accept failed: {e}"),
                    }
                };
                stream
                    .set_read_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                stream
                    .set_write_timeout(Some(Duration::from_secs(5)))
                    .unwrap();
                let mut len = [0u8; 4];
                stream.read_exact(&mut len).unwrap();
                let mut payload = vec![0u8; u32::from_le_bytes(len) as usize];
                stream.read_exact(&mut payload).unwrap();
                requests.push(decode_payload(&payload).unwrap());
                stream.write_all(&encode_frame(&response).unwrap()).unwrap();
            }
            requests
        });
        (dir, socket, handle)
    }

    fn sample_evidence(uid: u32, outcome: GuardedRollbackOutcomeV2) -> GuardedRollbackEvidenceV2 {
        GuardedRollbackEvidenceV2 {
            ws_id: "ws-abc123".to_string(),
            registered_path: "/workspace".to_string(),
            expected_generation: WorkspaceGenerationTokenV2::from_bytes([7; 32]),
            target_snapshot_id: "s1".to_string(),
            expected_diff_digest: [8; 32],
            operation_id: "op-123".to_string(),
            operation_digest: [9; 32],
            caller_uid: uid,
            outcome,
        }
    }

    fn sample_confirmation() -> Value {
        json!({
            "ws_id": "ws-abc123",
            "generation": ([7_u8; 32]),
            "diff_digest": ([8_u8; 32]),
            "operation_id": "op-123",
            "operation_digest": ([9_u8; 32]),
        })
    }

    #[tokio::test]
    async fn ordinary_execute_classifies_invalid_path_as_known_no_effect() {
        let (_dir, socket, peer) = spawn_mock_peer(vec![Response::Error {
            code: ErrorCode::InvalidPath,
            message: "snapshot reference '__live__' resolved to a rejected id".to_string(),
        }]);
        std::env::set_var("WS_CKPT_SOCKET", &socket);
        let output = run_json(
            "ws-abc123".to_string(),
            Some("__live__".to_string()),
            None,
            false,
            Mode::Execute,
        )
        .await;
        let requests = peer.join().unwrap();
        assert_eq!(requests.len(), 1);
        assert_eq!(output["schema"], "ws-ckpt.rollback.v1");
        assert_eq!(output["code"], "InvalidPath");
        assert_eq!(output["status"], "known_no_effect");
    }

    #[tokio::test]
    async fn guarded_execute_validates_uncertain_evidence_binding() {
        if let Ok(case) = std::env::var(CHILD_CASE_ENV) {
            assert_eq!(case, "run");
            let output = run_json(
                "/workspace".to_string(),
                Some("s1".to_string()),
                None,
                true,
                Mode::Execute,
            )
            .await;
            println!("CHILD_JSON:{}", serde_json::to_string(&output).unwrap());
            return;
        }

        let uid = effective_uid().unwrap();
        let identity = || Response::WorkspaceIdentityV2Ok {
            protocol_version: GUARDED_CHECKPOINT_PROTOCOL_VERSION_V2,
            ws_id: "ws-abc123".to_string(),
            registered_path: "/workspace".to_string(),
            generation: WorkspaceGenerationTokenV2::from_bytes([7; 32]),
        };

        let mut wrong_op = sample_evidence(
            uid,
            GuardedRollbackOutcomeV2::Unknown {
                reason: "unverified peer diagnostic".to_string(),
            },
        );
        wrong_op.operation_id = "op-other".to_string();

        let mut wrong_path = sample_evidence(
            uid,
            GuardedRollbackOutcomeV2::Unknown {
                reason: "wrong path diagnostic".to_string(),
            },
        );
        wrong_path.registered_path = "/other-workspace".to_string();

        let contradictory_outcome = sample_evidence(
            uid,
            GuardedRollbackOutcomeV2::Succeeded {
                resulting_generation: WorkspaceGenerationTokenV2::from_bytes([10; 32]),
            },
        );

        for invalid_uncertain in [wrong_op, wrong_path, contradictory_outcome] {
            let (_dir, socket, peer) = spawn_mock_peer(vec![
                identity(),
                Response::GuardedRollbackV2Uncertain {
                    evidence: invalid_uncertain,
                },
            ]);
            let output = run_child_with_stdin(&socket, &sample_confirmation());
            peer.join().unwrap();
            assert_eq!(output["status"], "unknown");
            assert_eq!(output["code"], "UnexpectedResponse");
        }

        let valid_uncertain = sample_evidence(
            uid,
            GuardedRollbackOutcomeV2::Unknown {
                reason: "durable index save failed after switch".to_string(),
            },
        );
        let (_dir, socket, peer) = spawn_mock_peer(vec![
            identity(),
            Response::GuardedRollbackV2Uncertain {
                evidence: valid_uncertain,
            },
        ]);
        let output = run_child_with_stdin(&socket, &sample_confirmation());
        peer.join().unwrap();
        assert_eq!(output["status"], "unknown");
        assert_eq!(output["code"], "UncertainOutcome");
        assert_eq!(output["message"], "durable index save failed after switch");
    }

    fn run_child_with_stdin(socket: &std::path::Path, confirmation: &Value) -> Value {
        let mut child = Command::new(std::env::current_exe().unwrap())
            .args([
                "--exact",
                "rollback::tests::guarded_execute_validates_uncertain_evidence_binding",
                "--nocapture",
            ])
            .env(CHILD_CASE_ENV, "run")
            .env("WS_CKPT_SOCKET", socket)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()
            .unwrap();
        serde_json::to_writer(child.stdin.take().unwrap(), confirmation).unwrap();
        let output = child.wait_with_output().unwrap();
        assert!(output.status.success(), "child test failed: {output:?}");
        let stdout = String::from_utf8(output.stdout).unwrap();
        let line = stdout
            .lines()
            .find_map(|line| line.strip_prefix("CHILD_JSON:"))
            .expect("missing CHILD_JSON line from child test");
        serde_json::from_str(line).unwrap()
    }
}
