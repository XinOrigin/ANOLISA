//! Compact SSE state is selected for semantic-only FFI delivery.

use super::super::sse_stream::SseStream;
use super::{ConnectionId, ConnectionState, HttpConnectionAggregator};
use crate::aggregator::{AggregatedResult, RemoveReason};
use crate::parser::http::ParsedResponse;
use crate::parser::sse::ParsedSseEvent;
use crate::probes::sslsniff::SslEvent;
use std::cell::RefCell;
use std::rc::Rc;

impl HttpConnectionAggregator {
    /// Ignore tails of explicitly abandoned streams until a new exchange begins.
    pub(crate) fn discard_retired_sse(&mut self, event: &SslEvent) -> bool {
        if event.rw != 0 {
            return false;
        }
        let id = ConnectionId::from_ssl_event(event);
        let bytes = &event.buf[..event.buf_size() as usize];
        if self
            .retired_sse
            .peek(&id)
            .is_some_and(|(_, reason)| *reason == RemoveReason::CapacityRejected)
        {
            metrics::counter!("agentsight_http_sse_discarded_bytes_total", "reason" => "capacity_rejected")
                .increment(bytes.len() as u64);
            return true;
        }
        // A fresh HTTP response is a new exchange, even if its request was missed.
        if bytes.starts_with(b"HTTP/1.") {
            return false;
        }
        let Some((_, reason)) = self.retired_sse.get(&id) else {
            return false;
        };
        metrics::counter!("agentsight_http_sse_discarded_bytes_total", "reason" => reason.as_str())
            .increment(bytes.len() as u64);
        true
    }

    pub(crate) fn set_compact_sse(&mut self, enabled: bool) {
        self.compact_sse = enabled;
    }

    pub(super) fn sse_entry_state_for_connection(
        &mut self,
        id: ConnectionId,
        response: &ParsedResponse,
        request: Option<&crate::parser::http::ParsedRequest>,
    ) -> (
        Vec<ParsedSseEvent>,
        Option<Vec<u8>>,
        Option<String>,
        Option<Rc<RefCell<crate::utils::decompress::ZstdStreamDecoder>>>,
    ) {
        let qoder = request.is_some_and(|r| crate::parser::llm::is_qoder_path(&r.path));
        // The legacy raw buffer recognizes only single codings. A stack must
        // use incremental decoding too; retain_raw_events below still keeps
        // every original decoded frame when raw retention is requested.
        let stacked = response
            .content_encoding()
            .is_some_and(|value| value.contains(','));
        if !self.compact_sse && !qoder && !stacked {
            return Self::sse_entry_state(response);
        }
        match SseStream::new(response, self.max_sse_retained_bytes.max(1024)) {
            Ok(mut stream) => {
                if !self.compact_sse {
                    stream.retain_raw_events();
                }
                if let Some(request) = request
                    .filter(|request| crate::parser::llm::is_dashscope_native_path(&request.path))
                {
                    stream.set_native_incremental(request.json_body().and_then(|value| {
                        value
                            .pointer("/parameters/incremental_output")
                            .and_then(serde_json::Value::as_bool)
                    }));
                }
                let error = stream.feed(response.body(), &response.source_event).err();
                let done = error.is_none() && stream.done();
                self.sse_streams.put(id, stream);
                if let Some(error) = error {
                    log::warn!("Initial SSE fragment cannot be decoded | conn={id:?}: {error}");
                    self.sse_errors.put(id, reason(&error));
                }
                let marker = done.then(|| {
                    ParsedSseEvent::new_done_marker(Rc::new(response.source_event.metadata_only()))
                });
                (
                    marker.into_iter().collect(),
                    None,
                    response.content_encoding().map(str::to_owned),
                    None,
                )
            }
            Err(error) => {
                log::warn!("Streaming SSE decoder unavailable | conn={id:?}: {error}");
                Self::sse_entry_state(response)
            }
        }
    }

    fn active_sse_direction(&self, id: &ConnectionId) -> Option<i32> {
        match self.connections.peek(id) {
            Some(ConnectionState::SseActive {
                response_headers, ..
            }) => Some(response_headers.source_event.rw),
            _ => None,
        }
    }

    pub(crate) fn accepts_compact_sse_bytes(&self, event: &SslEvent) -> bool {
        let id = ConnectionId::from_ssl_event(event);
        self.sse_streams.contains(&id) && self.active_sse_direction(&id) == Some(event.rw)
    }

    /// Ignore reads when `SseStream` owns the stream or the read belongs to the opposite direction.
    pub(super) fn ignores_legacy_sse(&self, id: &ConnectionId, rw: i32) -> bool {
        self.active_sse_direction(id)
            .is_some_and(|response_rw| self.sse_streams.contains(id) || response_rw != rw)
    }

    pub(crate) fn accepts_unmatched_sse(&self, event: &SslEvent) -> bool {
        event.rw == 0
            && !self
                .connections
                .contains(&ConnectionId::from_ssl_event(event))
    }

    pub(crate) fn start_unmatched_sse(&mut self, event: &SslEvent) -> Option<AggregatedResult> {
        let bytes = &event.buf[..event.buf_size() as usize];
        let mut headers =
            std::collections::HashMap::from([("content-type".into(), "text/event-stream".into())]);
        let chunked = bytes
            .windows(2)
            .position(|pair| pair == b"\r\n")
            .and_then(|end| std::str::from_utf8(&bytes[..end]).ok())
            .is_some_and(|line| {
                usize::from_str_radix(line.split(';').next().unwrap_or(""), 16).is_ok()
            });
        if chunked {
            headers.insert("transfer-encoding".into(), "chunked".into());
        }
        self.process_response(ParsedResponse {
            version: 11,
            status_code: 0,
            reason: String::new(),
            headers,
            body_offset: 0,
            body_len: bytes.len(),
            source_event: Rc::new(event.clone()),
        })
    }

    pub(super) fn process_compact_sse_bytes(
        &mut self,
        event: &SslEvent,
    ) -> Option<AggregatedResult> {
        let id = ConnectionId::from_ssl_event(event);
        self.touch(&id);
        let stream = self.sse_streams.get_mut(&id)?;
        let error = stream
            .feed(&event.buf[..event.buf_size() as usize], event)
            .err();
        let done = error.is_none() && stream.done();
        if let Some(error) = error {
            log::warn!("SSE stream finalized early | conn={id:?}: {error}");
            self.abandon(&id, reason(&error));
        } else if done {
            if let Some(ConnectionState::SseActive {
                request,
                response_headers,
                ..
            }) = self.connections.pop(&id)
            {
                let result = self.finish_sse(id, request, response_headers, Vec::new());
                return Some(result);
            }
        }
        self.refresh_retained(&id);
        self.enforce_connection_limits(&id);
        None
    }
}

fn reason(error: &std::io::Error) -> RemoveReason {
    if error.kind() == std::io::ErrorKind::OutOfMemory {
        RemoveReason::OverBudget
    } else {
        RemoveReason::ProtocolError
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::parser::http::ParsedRequest;
    use std::collections::HashMap;

    fn ssl_event(ssl_ptr: u64, rw: i32, timestamp_ns: u64, bytes: &[u8]) -> Rc<SslEvent> {
        Rc::new(SslEvent {
            source: 0,
            timestamp_ns,
            delta_ns: 0,
            pid: std::process::id(),
            tid: 1,
            uid: 0,
            len: bytes.len() as u32,
            rw,
            comm: "fixture".into(),
            buf: bytes.to_vec(),
            is_handshake: false,
            ssl_ptr,
        })
    }

    fn post_request(ssl_ptr: u64, rw: i32) -> ParsedRequest {
        let body = b"{}";
        ParsedRequest {
            method: "POST".into(),
            path: "/v1/chat/completions".into(),
            version: 11,
            headers: HashMap::from([("content-length".into(), body.len().to_string())]),
            body_offset: 0,
            body_len: body.len(),
            source_event: ssl_event(ssl_ptr, rw, 1_000, body),
            reassembled_body: None,
        }
    }

    fn chunked_sse_headers(ssl_ptr: u64, rw: i32) -> ParsedResponse {
        ParsedResponse {
            version: 11,
            status_code: 200,
            reason: "OK".into(),
            headers: HashMap::from([
                ("content-type".into(), "text/event-stream".into()),
                ("transfer-encoding".into(), "chunked".into()),
            ]),
            body_offset: 0,
            body_len: 0,
            source_event: ssl_event(ssl_ptr, rw, 1_500, b""),
        }
    }

    fn chunk(payload: &[u8]) -> Vec<u8> {
        let mut framed = format!("{:x}\r\n", payload.len()).into_bytes();
        framed.extend_from_slice(payload);
        framed.extend_from_slice(b"\r\n");
        framed
    }

    #[test]
    fn compact_sse_accepts_server_side_response_direction() {
        let mut http = HttpConnectionAggregator::new();
        http.set_compact_sse(true);
        http.process_request(post_request(7, 0));
        assert!(http.process_response(chunked_sse_headers(7, 1)).is_none());

        let delta = chunk(
            b"data: {\"id\":\"srv\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"hello\"}}]}\n\n",
        );
        let delta_ev = ssl_event(7, 1, 2_000, &delta);
        assert!(http.accepts_compact_sse_bytes(&delta_ev));
        assert!(http.process_raw_body_data(&delta_ev).is_none());

        let end_ev = ssl_event(7, 1, 3_000, b"0\r\n\r\n");
        assert!(http.accepts_compact_sse_bytes(&end_ev));
        let Some(AggregatedResult::SseComplete(pair)) = http.process_raw_body_data(&end_ev) else {
            panic!("server-side compact SSE stream must complete on chunk terminator");
        };
        assert_eq!(pair.response.captured_sse_event_count, Some(1));
        assert_eq!(pair.response.completion_timestamp_ns, Some(3_000));
        assert!(pair.response.body_string().contains("hello"));
        assert!(!http.accepts_compact_sse_bytes(&end_ev));
        assert!(http.take_ready().is_empty());
    }

    #[test]
    fn compact_sse_rejects_opposite_direction_bytes_on_server_side_stream() {
        let mut http = HttpConnectionAggregator::new();
        http.set_compact_sse(true);
        http.process_request(post_request(8, 0));
        assert!(http.process_response(chunked_sse_headers(8, 1)).is_none());

        let opposite = ssl_event(
            8,
            0,
            2_000,
            b"POST /v1/chat/completions HTTP/1.1\r\nHost: localhost\r\n\r\n",
        );
        assert!(
            !http.accepts_compact_sse_bytes(&opposite),
            "opposite-direction bytes must not be fed into a server-side SSE stream"
        );
        assert!(http.process_raw_body_data(&opposite).is_none());
        assert!(
            http.take_ready().is_empty(),
            "opposite-direction bytes must not abort the active SSE stream"
        );

        let payload = chunk(
            b"data: {\"id\":\"srv2\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"intact\"}}]}\n\n",
        );
        let mut final_bytes = payload;
        final_bytes.extend_from_slice(b"0\r\n\r\n");
        let end_ev = ssl_event(8, 1, 3_500, &final_bytes);
        let Some(AggregatedResult::SseComplete(pair)) = http.process_raw_body_data(&end_ev) else {
            panic!("server-side compact SSE stream must remain intact");
        };
        assert!(pair.response.body_string().contains("intact"));
        assert!(pair.response.sse_continuation_bytes.is_none());
    }

    #[test]
    fn compact_sse_preserves_client_side_direction_matching() {
        let mut http = HttpConnectionAggregator::new();
        http.set_compact_sse(true);
        http.process_request(post_request(9, 1));
        assert!(http.process_response(chunked_sse_headers(9, 0)).is_none());

        let req_dir = ssl_event(9, 1, 2_000, b"POST /next HTTP/1.1\r\n\r\n");
        assert!(!http.accepts_compact_sse_bytes(&req_dir));

        let mut payload = chunk(
            b"data: {\"id\":\"cli\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"ok\"}}]}\n\n",
        );
        payload.extend_from_slice(b"0\r\n\r\n");
        let resp_dir = ssl_event(9, 0, 2_500, &payload);
        assert!(http.accepts_compact_sse_bytes(&resp_dir));
        let Some(AggregatedResult::SseComplete(pair)) = http.process_raw_body_data(&resp_dir)
        else {
            panic!("client-side compact SSE stream must complete");
        };
        assert!(pair.response.body_string().contains("ok"));
    }
}
