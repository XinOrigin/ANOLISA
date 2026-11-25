use std::sync::Arc;
use std::time::Duration;

use anyhow::Context;
use tokio::io::{AsyncRead, AsyncReadExt, AsyncWrite, AsyncWriteExt};
use tokio::net::UnixListener;
use tokio::sync::Semaphore;
use tokio::task::JoinSet;
use tokio_util::sync::CancellationToken;
use tracing::{error, info};

use crate::state::DaemonState;
use ws_ckpt_common::{
    decode_payload, encode_frame, ErrorCode, Request, Response, WsCkptError, MAX_FRAME_SIZE,
};

const MAX_CONCURRENT_CONNECTIONS: usize = 64;
const HEADER_READ_TIMEOUT: Duration = Duration::from_secs(10);
const PAYLOAD_READ_TIMEOUT: Duration = Duration::from_secs(30);
const RESPONSE_WRITE_TIMEOUT: Duration = Duration::from_secs(30);
const INITIAL_PAYLOAD_CAPACITY: usize = 64 * 1024;

#[derive(Debug, Clone, Copy)]
struct ListenerConfig {
    max_concurrent_connections: usize,
    header_read_timeout: Duration,
    payload_read_timeout: Duration,
    write_timeout: Duration,
}

impl Default for ListenerConfig {
    fn default() -> Self {
        Self {
            max_concurrent_connections: MAX_CONCURRENT_CONNECTIONS,
            header_read_timeout: HEADER_READ_TIMEOUT,
            payload_read_timeout: PAYLOAD_READ_TIMEOUT,
            write_timeout: RESPONSE_WRITE_TIMEOUT,
        }
    }
}

pub async fn run_listener(
    state: Arc<DaemonState>,
    cancel: CancellationToken,
) -> anyhow::Result<()> {
    run_listener_with_config(state, cancel, ListenerConfig::default()).await
}

async fn run_listener_with_config(
    state: Arc<DaemonState>,
    cancel: CancellationToken,
    config: ListenerConfig,
) -> anyhow::Result<()> {
    // 1. Clean up residual socket file
    let _ = std::fs::remove_file(&state.socket_path);

    // 2. Ensure socket parent directory exists
    if let Some(parent) = state.socket_path.parent() {
        tokio::fs::create_dir_all(parent)
            .await
            .context("Failed to create socket parent directory")?;
    }

    // 3. Bind the Unix listener
    let listener = UnixListener::bind(&state.socket_path).context("Failed to bind Unix socket")?;
    info!("Listening on {:?}", state.socket_path);

    // 4. Set socket permissions to 0o666
    use std::os::unix::fs::PermissionsExt;
    std::fs::set_permissions(&state.socket_path, std::fs::Permissions::from_mode(0o666))
        .context("Failed to set socket permissions")?;

    // 5. Accept loop bounded by a connection semaphore so stalled peers cannot
    // exhaust file descriptors or spawn unbounded tasks.
    let connection_permits = Arc::new(Semaphore::new(config.max_concurrent_connections.max(1)));
    let mut join_set = JoinSet::new();

    loop {
        let permit = tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                info!("Listener received cancellation signal");
                break;
            }
            permit = Arc::clone(&connection_permits).acquire_owned() => {
                match permit {
                    Ok(permit) => permit,
                    Err(_) => {
                        error!("Connection semaphore closed unexpectedly");
                        break;
                    }
                }
            }
        };

        // Reap finished connections before waiting for the next accept.
        while join_set.try_join_next().is_some() {}

        tokio::select! {
            biased;
            _ = cancel.cancelled() => {
                drop(permit);
                info!("Listener received cancellation signal");
                break;
            }
            result = listener.accept() => {
                match result {
                    Ok((stream, _addr)) => {
                        let state = Arc::clone(&state);
                        join_set.spawn(async move {
                            let _permit = permit;
                            if let Err(e) = handle_connection_with_config(stream, state, config).await {
                                error!("Connection error: {:#}", e);
                            }
                        });
                    }
                    Err(e) => {
                        drop(permit);
                        error!("Accept error: {}", e);
                    }
                }
            }
        }

        // Reap finished connections. JoinSet holds each task's handle and output
        // slot until it is joined, so without this the set grows by one entry per
        // CLI invocation and is only released at shutdown. try_join_next never
        // awaits, so reaping cannot delay the next accept; a join_next branch in
        // the select! above would instead need a second mutable borrow of
        // join_set while the accept arm still spawns into it.
        while join_set.try_join_next().is_some() {}
    }

    // 7. Wait for in-flight tasks to complete (with timeout)
    info!("Waiting for in-flight connections to complete...");
    let drain = async { while join_set.join_next().await.is_some() {} };
    if tokio::time::timeout(Duration::from_secs(10), drain)
        .await
        .is_err()
    {
        error!("Timed out waiting for in-flight connections; aborting remaining tasks");
        join_set.abort_all();
    }

    // Clean up socket file
    let _ = std::fs::remove_file(&state.socket_path);
    info!("Listener shut down");
    Ok(())
}

fn new_payload_buffer(expected_len: usize) -> Vec<u8> {
    Vec::with_capacity(expected_len.min(INITIAL_PAYLOAD_CAPACITY))
}

async fn read_frame_payload_into<R: AsyncRead + Unpin>(
    reader: &mut R,
    payload: &mut Vec<u8>,
    expected_len: u32,
    timeout: Duration,
) -> anyhow::Result<()> {
    let expected_usize =
        usize::try_from(expected_len).context("Frame length overflows platform usize")?;
    let read_result = tokio::time::timeout(timeout, async {
        let mut limited = (&mut *reader).take(u64::from(expected_len));
        while payload.len() < expected_usize {
            let remaining = expected_usize - payload.len();
            if payload.capacity() == payload.len() {
                let grow_by = remaining.min(payload.capacity().max(INITIAL_PAYLOAD_CAPACITY));
                payload.try_reserve_exact(grow_by).map_err(|err| {
                    std::io::Error::new(
                        std::io::ErrorKind::OutOfMemory,
                        format!("Failed to reserve payload buffer: {err}"),
                    )
                })?;
            }
            let bytes_read = limited.read_buf(payload).await?;
            if bytes_read == 0 {
                break;
            }
        }
        Ok::<(), std::io::Error>(())
    })
    .await;

    match read_result {
        Ok(Ok(())) => {
            if payload.len() != expected_usize {
                anyhow::bail!(
                    "Truncated frame payload: expected {} bytes, got {} bytes",
                    expected_usize,
                    payload.len()
                );
            }
            Ok(())
        }
        Ok(Err(err)) => Err(err).context("Failed to read frame payload"),
        Err(_) => anyhow::bail!(
            "Timed out reading frame payload after {:?} ({}/{} bytes received)",
            timeout,
            payload.len(),
            expected_usize
        ),
    }
}

async fn read_frame_payload<R: AsyncRead + Unpin>(
    reader: &mut R,
    expected_len: u32,
    timeout: Duration,
) -> anyhow::Result<Vec<u8>> {
    let expected_usize =
        usize::try_from(expected_len).context("Frame length overflows platform usize")?;
    let mut payload = new_payload_buffer(expected_usize);
    read_frame_payload_into(reader, &mut payload, expected_len, timeout).await?;
    Ok(payload)
}

async fn write_frame_with_timeout<W: AsyncWrite + Unpin>(
    writer: &mut W,
    frame: &[u8],
    timeout: Duration,
) -> anyhow::Result<()> {
    tokio::time::timeout(timeout, writer.write_all(frame))
        .await
        .map_err(|_| anyhow::anyhow!("Timed out writing response after {:?}", timeout))?
        .context("Failed to write response")
}

#[cfg(test)]
async fn handle_connection(
    stream: tokio::net::UnixStream,
    state: Arc<DaemonState>,
) -> anyhow::Result<()> {
    handle_connection_with_config(stream, state, ListenerConfig::default()).await
}

async fn handle_connection_with_config(
    mut stream: tokio::net::UnixStream,
    state: Arc<DaemonState>,
    config: ListenerConfig,
) -> anyhow::Result<()> {
    // Read 4-byte LE length with deadline
    let len = tokio::time::timeout(config.header_read_timeout, stream.read_u32_le())
        .await
        .map_err(|_| {
            anyhow::anyhow!(
                "Timed out reading frame length after {:?}",
                config.header_read_timeout
            )
        })?
        .context("Failed to read frame length")?;

    // Validate frame size
    if len > MAX_FRAME_SIZE {
        let err_resp = Response::Error {
            code: ErrorCode::InternalError,
            message: format!("Frame too large: {} bytes (max {})", len, MAX_FRAME_SIZE),
        };
        let frame = encode_frame(&err_resp)?;
        write_frame_with_timeout(&mut stream, &frame, config.write_timeout).await?;
        anyhow::bail!("Frame too large: {} bytes", len);
    }

    // Read payload incrementally with bounded initial allocation and deadline
    let payload = read_frame_payload(&mut stream, len, config.payload_read_timeout).await?;

    // Decode request
    let request: Request = decode_payload(&payload).context("Failed to decode request")?;

    let peer_cred = stream.peer_cred().ok();
    let agent_name = peer_cred
        .as_ref()
        .and_then(|cred| cred.pid().map(|p| p as u32))
        .map(crate::ops_log::detect_agent_name)
        .unwrap_or_else(|| "user".to_string());
    let ops_name = crate::ops_log::ops_name_from_request(&request);

    // Dispatch (intentionally unbounded by I/O timeouts because Btrfs operations can take longer)
    let context = crate::dispatcher::DispatchContext::new(peer_cred.map(|cred| cred.uid()));
    let mut response = crate::dispatcher::dispatch_with_context(&state, request, context).await;

    let frame = match encode_frame(&response) {
        Err(err @ WsCkptError::FrameTooLarge { .. }) => {
            let advice = if matches!(response, Response::ListOk { .. }) {
                "; use list --limit <N> and the returned cursor; use status for snapshot counts"
            } else {
                ""
            };
            response = Response::Error {
                code: ErrorCode::InternalError,
                message: format!("{err}{advice}"),
            };
            encode_frame(&response)?
        }
        result => result.context("Failed to encode response")?,
    };

    if let Some(name) = ops_name {
        crate::ops_log::log_operation(name, &agent_name, &response);
    }

    // Write encoded response with deadline
    write_frame_with_timeout(&mut stream, &frame, config.write_timeout).await?;

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use ws_ckpt_common::{DaemonConfig, SnapshotIndex, SnapshotMeta};

    async fn exchange(state: &Arc<DaemonState>, request: Request) -> Response {
        // Exercise the real connection handler without a daemon or background task.
        let (mut client, server) = tokio::net::UnixStream::pair().unwrap();
        let client_io = async {
            client
                .write_all(&encode_frame(&request).unwrap())
                .await
                .unwrap();
            let len = client.read_u32_le().await.unwrap();
            assert!(len <= MAX_FRAME_SIZE);
            let mut payload = vec![0; len as usize];
            client.read_exact(&mut payload).await.unwrap();
            decode_payload(&payload).unwrap()
        };
        tokio::time::timeout(Duration::from_secs(30), async {
            let (server, response) =
                tokio::join!(handle_connection(server, state.clone()), client_io);
            server.unwrap();
            response
        })
        .await
        .expect("IPC round trip timed out")
    }

    #[tokio::test]
    async fn list_above_frame_limit_remains_manageable() {
        let temp = tempfile::tempdir().unwrap();
        let backend = Arc::new(crate::backends::btrfs_loop::BtrfsLoopBackend::new(
            temp.path().join("data"),
            temp.path().join("test.img"),
        ));
        let state = Arc::new(DaemonState::new(
            DaemonConfig::default(),
            backend,
            temp.path().join("state"),
        ));
        let created_at = chrono::Utc::now();
        for (ws_id, count) in [("ws-a", 100_000), ("ws-b", 2)] {
            let subvol = state.backend.data_root().join(ws_id);
            std::fs::create_dir_all(&subvol).unwrap();
            let path = temp.path().join(ws_id);
            std::os::unix::fs::symlink(subvol, &path).unwrap();
            let mut index = SnapshotIndex::new(path.clone());
            for i in (0..count).rev() {
                index.snapshots.insert(
                    format!("s-{i:06}"),
                    SnapshotMeta {
                        message: Some("x".repeat(128)),
                        metadata: None,
                        pinned: false,
                        created_at,
                        missing: false,
                        parent_id: None,
                        child_ids: vec![],
                    },
                );
            }
            state.register_workspace(ws_id.into(), path, index).unwrap();
        }

        for workspace in [Some("ws-a".to_string()), None] {
            let response = exchange(
                &state,
                Request::List {
                    workspace: workspace.clone(),
                    format: None,
                },
            )
            .await;
            assert!(matches!(response, Response::Error { message, .. }
                if message.contains("frame too large") && message.contains("--limit")));

            let first = exchange(
                &state,
                Request::ListPage {
                    orphans_only: false,
                    workspace: workspace.clone(),
                    limit: 2,
                    cursor: None,
                },
            )
            .await;
            let Response::ListPageOk {
                snapshots,
                next_cursor,
            } = first
            else {
                panic!("expected first page")
            };
            let mut ids = snapshots
                .into_iter()
                .map(|entry| match entry {
                    ws_ckpt_common::SnapshotListItem::Full(entry) => entry.id,
                    ws_ckpt_common::SnapshotListItem::Summary(summary) => summary.id,
                })
                .collect::<Vec<_>>();
            let second = exchange(
                &state,
                Request::ListPage {
                    orphans_only: false,
                    workspace: workspace.clone(),
                    limit: 2,
                    cursor: next_cursor,
                },
            )
            .await;
            let Response::ListPageOk { snapshots, .. } = second else {
                panic!("expected second page")
            };
            ids.extend(snapshots.into_iter().map(|entry| match entry {
                ws_ckpt_common::SnapshotListItem::Full(entry) => entry.id,
                ws_ckpt_common::SnapshotListItem::Summary(summary) => summary.id,
            }));
            assert_eq!(ids, ["s-000000", "s-000001", "s-000002", "s-000003"]);

            let response = exchange(
                &state,
                Request::ListPage {
                    orphans_only: false,
                    workspace: workspace.clone(),
                    limit: 2,
                    cursor: Some("not-a-cursor".into()),
                },
            )
            .await;
            assert!(matches!(response, Response::Error { message, .. }
                if message.contains("cursor")));
            let response = exchange(
                &state,
                Request::ListPage {
                    orphans_only: false,
                    workspace,
                    limit: 0,
                    cursor: None,
                },
            )
            .await;
            assert!(
                matches!(response, Response::Error { message, .. } if message.contains("between 1"))
            );
        }

        let first = exchange(
            &state,
            Request::ListPage {
                orphans_only: false,
                workspace: Some("ws-b".into()),
                limit: 1,
                cursor: None,
            },
        )
        .await;
        let Response::ListPageOk {
            next_cursor: Some(ws_b_cursor),
            ..
        } = first
        else {
            panic!("expected ws-b continuation cursor")
        };
        let scope_error = exchange(
            &state,
            Request::ListPage {
                orphans_only: false,
                workspace: Some("ws-a".into()),
                limit: 1,
                cursor: Some(ws_b_cursor.clone()),
            },
        )
        .await;
        assert!(matches!(scope_error, Response::Error { message, .. }
            if message.contains("does not match")));

        {
            let arc = state.get_by_wsid("ws-b").unwrap();
            let mut guard = arc.write().await;
            guard.remove_snapshot("s-000001");
            guard.insert_snapshot(
                "s-000000a".into(),
                SnapshotMeta {
                    message: Some("inserted within first-page bound".into()),
                    metadata: None,
                    pinned: false,
                    created_at,
                    missing: false,
                    parent_id: None,
                    child_ids: vec![],
                },
            );
            guard.insert_snapshot(
                "s-newer".into(),
                SnapshotMeta {
                    message: Some("newer than first-page bound".into()),
                    metadata: None,
                    pinned: false,
                    created_at: created_at + chrono::Duration::seconds(1),
                    missing: false,
                    parent_id: None,
                    child_ids: vec![],
                },
            );
        }
        let second = exchange(
            &state,
            Request::ListPage {
                orphans_only: false,
                workspace: Some("ws-b".into()),
                limit: 10,
                cursor: Some(ws_b_cursor),
            },
        )
        .await;
        let Response::ListPageOk {
            snapshots,
            next_cursor,
        } = second
        else {
            panic!("expected ws-b continuation page")
        };
        assert!(next_cursor.is_none());
        assert!(matches!(snapshots.as_slice(),
            [ws_ckpt_common::SnapshotListItem::Full(entry)] if entry.id == "s-000000a"));

        let response = exchange(
            &state,
            Request::Status {
                workspace: Some("ws-a".into()),
            },
        )
        .await;
        assert!(
            matches!(response, Response::StatusOk { report } if report.workspaces[0].snapshot_count == 100_000)
        );
        let response = exchange(
            &state,
            Request::List {
                workspace: Some("ws-b".into()),
                format: None,
            },
        )
        .await;
        assert!(matches!(response, Response::ListOk { snapshots } if snapshots.len() == 3));
        let response = exchange(
            &state,
            Request::ListPage {
                orphans_only: false,
                workspace: Some("unknown".into()),
                limit: 2,
                cursor: None,
            },
        )
        .await;
        assert!(matches!(
            response,
            Response::Error {
                code: ErrorCode::WorkspaceNotFound,
                ..
            }
        ));

        let arc = state.get_by_wsid("ws-a").unwrap();
        arc.write()
            .await
            .index
            .snapshots
            .get_mut("s-000000")
            .unwrap()
            .message = Some("x".repeat(MAX_FRAME_SIZE as usize));
        let response = exchange(
            &state,
            Request::ListPage {
                orphans_only: false,
                workspace: Some("ws-a".into()),
                limit: 1,
                cursor: None,
            },
        )
        .await;
        let Response::ListPageOk { snapshots, .. } = response else {
            panic!("expected summary page")
        };
        assert_eq!(snapshots.len(), 1);
        let ws_ckpt_common::SnapshotListItem::Summary(summary) = &snapshots[0] else {
            panic!("expected oversized entry summary, got {:?}", snapshots[0])
        };
        assert_eq!(summary.id, "s-000000");
        assert_eq!(
            summary.omitted_fields,
            ["message", "metadata", "parent_id", "child_ids"]
        );
        std::fs::remove_file(temp.path().join("ws-a")).unwrap();
        let response = exchange(
            &state,
            Request::ListPage {
                orphans_only: false,
                workspace: Some("ws-a".into()),
                limit: 1,
                cursor: None,
            },
        )
        .await;
        assert!(matches!(response, Response::Error { message, .. } if message.contains("recover")));
    }

    fn test_state(temp: &tempfile::TempDir) -> Arc<DaemonState> {
        let backend = Arc::new(crate::backends::btrfs_loop::BtrfsLoopBackend::new(
            temp.path().join("data"),
            temp.path().join("test.img"),
        ));
        let config = DaemonConfig {
            socket_path: temp.path().join("daemon.sock"),
            ..DaemonConfig::default()
        };
        Arc::new(DaemonState::new(config, backend, temp.path().join("state")))
    }

    async fn connect_when_ready(socket_path: &std::path::Path) -> tokio::net::UnixStream {
        tokio::time::timeout(Duration::from_secs(2), async {
            loop {
                match tokio::net::UnixStream::connect(socket_path).await {
                    Ok(stream) => return stream,
                    Err(_) => tokio::time::sleep(Duration::from_millis(5)).await,
                }
            }
        })
        .await
        .expect("listener socket did not become ready in time")
    }

    async fn round_trip_on_stream(
        stream: &mut tokio::net::UnixStream,
        request: &Request,
    ) -> Response {
        stream
            .write_all(&encode_frame(request).unwrap())
            .await
            .unwrap();
        let len = stream.read_u32_le().await.unwrap();
        assert!(len <= MAX_FRAME_SIZE);
        let mut payload = vec![0u8; len as usize];
        stream.read_exact(&mut payload).await.unwrap();
        decode_payload(&payload).unwrap()
    }

    async fn round_trip_socket(socket_path: &std::path::Path, request: &Request) -> Response {
        let mut stream = connect_when_ready(socket_path).await;
        round_trip_on_stream(&mut stream, request).await
    }

    #[tokio::test]
    async fn read_frame_payload_stalled_max_frame_times_out_without_preallocating() {
        let (mut server, _client) = tokio::io::duplex(64);
        let mut buf = new_payload_buffer(MAX_FRAME_SIZE as usize);
        assert_eq!(buf.capacity(), INITIAL_PAYLOAD_CAPACITY);

        let err = read_frame_payload_into(
            &mut server,
            &mut buf,
            MAX_FRAME_SIZE,
            Duration::from_millis(40),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("Timed out reading frame payload"),
            "unexpected error: {err:#}"
        );
        assert!(
            buf.capacity() <= INITIAL_PAYLOAD_CAPACITY,
            "stalled 16 MiB header grew buffer capacity to {}",
            buf.capacity()
        );
    }

    #[tokio::test]
    async fn read_frame_payload_truncated_eof_fails_cleanly_without_preallocating() {
        let (mut server, mut client) = tokio::io::duplex(64);
        client.write_all(&[0x5Au8; 32]).await.unwrap();
        drop(client);

        let mut buf = new_payload_buffer(MAX_FRAME_SIZE as usize);
        let err = read_frame_payload_into(
            &mut server,
            &mut buf,
            MAX_FRAME_SIZE,
            Duration::from_secs(5),
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("Truncated frame payload"),
            "unexpected error: {err:#}"
        );
        assert_eq!(buf.len(), 32);
        assert!(
            buf.capacity() <= INITIAL_PAYLOAD_CAPACITY,
            "truncated 16 MiB header grew buffer capacity to {}",
            buf.capacity()
        );
    }

    #[tokio::test]
    async fn read_frame_payload_grows_incrementally_to_exact_length() {
        const PAYLOAD_LEN: usize = 200_000;
        let expected: Vec<u8> = (0..PAYLOAD_LEN).map(|i| (i % 251) as u8).collect();
        let (mut server, mut client) = tokio::io::duplex(8 * 1024);

        let to_send = expected.clone();
        let writer = tokio::spawn(async move {
            client.write_all(&to_send).await.unwrap();
            // Keep write half open to verify `.take()` stops at exact frame boundary.
            client
        });

        let payload = read_frame_payload(&mut server, PAYLOAD_LEN as u32, Duration::from_secs(5))
            .await
            .unwrap();
        assert_eq!(payload, expected);
        assert_eq!(payload.capacity(), PAYLOAD_LEN);
        drop(writer.await.unwrap());
    }

    #[tokio::test]
    async fn write_frame_times_out_when_peer_stops_reading() {
        let (_reader, mut writer) = tokio::io::duplex(16);
        let err = write_frame_with_timeout(&mut writer, &[0xAB; 1024], Duration::from_millis(40))
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("Timed out writing response"),
            "unexpected error: {err:#}"
        );
    }

    #[tokio::test]
    async fn handle_connection_rejects_stalled_truncated_and_oversized_frames() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp);
        let fast_config = ListenerConfig {
            max_concurrent_connections: 2,
            header_read_timeout: Duration::from_millis(40),
            payload_read_timeout: Duration::from_millis(40),
            write_timeout: Duration::from_millis(40),
        };

        // 1. Stalled before sending 4-byte length header
        let (_client_idle, server_idle) = tokio::net::UnixStream::pair().unwrap();
        let err = handle_connection_with_config(server_idle, state.clone(), fast_config)
            .await
            .unwrap_err();
        assert!(
            err.to_string().contains("Timed out reading frame length"),
            "unexpected error: {err:#}"
        );

        // 2. Truncated payload claiming MAX_FRAME_SIZE fails immediately on EOF
        let (mut client_trunc, server_trunc) = tokio::net::UnixStream::pair().unwrap();
        client_trunc
            .write_all(&MAX_FRAME_SIZE.to_le_bytes())
            .await
            .unwrap();
        client_trunc.write_all(&[0u8; 16]).await.unwrap();
        client_trunc.shutdown().await.unwrap();
        let err = handle_connection_with_config(
            server_trunc,
            state.clone(),
            ListenerConfig {
                payload_read_timeout: Duration::from_secs(5),
                ..fast_config
            },
        )
        .await
        .unwrap_err();
        assert!(
            err.to_string().contains("Truncated frame payload"),
            "unexpected error: {err:#}"
        );

        // 3. Oversized frame (> MAX_FRAME_SIZE) returns Error response and bails cleanly
        let (mut client_big, server_big) = tokio::net::UnixStream::pair().unwrap();
        let oversized_len = MAX_FRAME_SIZE + 1;
        let client_task = async {
            client_big
                .write_all(&oversized_len.to_le_bytes())
                .await
                .unwrap();
            let resp_len = client_big.read_u32_le().await.unwrap();
            let mut resp_payload = vec![0u8; resp_len as usize];
            client_big.read_exact(&mut resp_payload).await.unwrap();
            decode_payload::<Response>(&resp_payload).unwrap()
        };
        let (server_res, resp) = tokio::join!(
            handle_connection_with_config(server_big, state, fast_config),
            client_task
        );
        let err = server_res.unwrap_err();
        assert!(
            err.to_string().contains("Frame too large"),
            "unexpected error: {err:#}"
        );
        assert!(matches!(
            resp,
            Response::Error {
                code: ErrorCode::InternalError,
                message,
            } if message.contains("Frame too large")
        ));
    }

    #[tokio::test]
    async fn listener_stalled_max_frame_clients_timeout_and_valid_requests_succeed() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp);
        let socket_path = state.socket_path.clone();
        let cancel = CancellationToken::new();
        let config = ListenerConfig {
            max_concurrent_connections: 2,
            header_read_timeout: Duration::from_millis(50),
            payload_read_timeout: Duration::from_millis(50),
            write_timeout: Duration::from_millis(50),
        };

        let listener_task = tokio::spawn(run_listener_with_config(
            state.clone(),
            cancel.clone(),
            config,
        ));

        // Ensure listener is bound and accepting before launching stalled peers.
        let initial = round_trip_socket(&socket_path, &Request::Config).await;
        assert!(matches!(initial, Response::ConfigOk { .. }));

        // Launch 4 stalled peers (> concurrency cap of 2): two send nothing, two send
        // 4-byte MAX_FRAME_SIZE headers without payload bytes.
        let mut stalled_clients = Vec::new();
        for idx in 0..4 {
            let mut client = connect_when_ready(&socket_path).await;
            if idx % 2 == 1 {
                client
                    .write_all(&MAX_FRAME_SIZE.to_le_bytes())
                    .await
                    .unwrap();
            }
            stalled_clients.push(client);
        }

        // All 4 stalled connections must be timed out and closed by the daemon.
        for mut client in stalled_clients {
            let eof = tokio::time::timeout(Duration::from_secs(2), client.read_u8())
                .await
                .expect("stalled client was not closed before deadline")
                .unwrap_err();
            assert_eq!(eof.kind(), std::io::ErrorKind::UnexpectedEof);
        }

        // Subsequent normal requests must succeed cleanly.
        let config_resp = tokio::time::timeout(
            Duration::from_secs(2),
            round_trip_socket(&socket_path, &Request::Config),
        )
        .await
        .expect("Config request timed out after stalled clients");
        assert!(matches!(config_resp, Response::ConfigOk { .. }));

        let status_resp = tokio::time::timeout(
            Duration::from_secs(2),
            round_trip_socket(&socket_path, &Request::Status { workspace: None }),
        )
        .await
        .expect("Status request timed out after stalled clients");
        assert!(matches!(status_resp, Response::StatusOk { .. }));

        cancel.cancel();
        listener_task.await.unwrap().unwrap();
    }

    #[tokio::test]
    async fn listener_semaphore_caps_concurrent_connections() {
        let temp = tempfile::tempdir().unwrap();
        let state = test_state(&temp);
        let socket_path = state.socket_path.clone();
        let cancel = CancellationToken::new();
        let config = ListenerConfig {
            max_concurrent_connections: 2,
            header_read_timeout: Duration::from_secs(5),
            payload_read_timeout: Duration::from_secs(5),
            write_timeout: Duration::from_secs(5),
        };

        let listener_task = tokio::spawn(run_listener_with_config(
            state.clone(),
            cancel.clone(),
            config,
        ));

        let initial = round_trip_socket(&socket_path, &Request::Config).await;
        assert!(matches!(initial, Response::ConfigOk { .. }));

        // Hold both concurrency permits with two stalled payload clients.
        let mut c1 = connect_when_ready(&socket_path).await;
        c1.write_all(&1024u32.to_le_bytes()).await.unwrap();
        let mut c2 = connect_when_ready(&socket_path).await;
        c2.write_all(&1024u32.to_le_bytes()).await.unwrap();

        // A third client connects and writes a valid Status request, but cannot be
        // served until one of the two active permits is released.
        let mut c3 = connect_when_ready(&socket_path).await;
        c3.write_all(&encode_frame(&Request::Status { workspace: None }).unwrap())
            .await
            .unwrap();
        assert!(
            tokio::time::timeout(Duration::from_millis(60), c3.readable())
                .await
                .is_err(),
            "third connection should wait while all connection permits are held"
        );

        // Closing c1 frees a permit immediately; c3 should now receive StatusOk.
        drop(c1);
        let response = tokio::time::timeout(Duration::from_secs(2), async {
            let len = c3.read_u32_le().await.unwrap();
            let mut payload = vec![0u8; len as usize];
            c3.read_exact(&mut payload).await.unwrap();
            decode_payload::<Response>(&payload).unwrap()
        })
        .await
        .expect("queued client did not complete after permit was released");
        assert!(matches!(response, Response::StatusOk { .. }));

        drop(c2);
        cancel.cancel();
        listener_task.await.unwrap().unwrap();
    }
}
