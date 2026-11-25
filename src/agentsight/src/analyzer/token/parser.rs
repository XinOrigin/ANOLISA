//! Token Parser - Extract token usage from SSE events
//!
//! This module provides functionality to extract LLM token usage information
//! from SSE (Server-Sent Events) streaming responses.
//!
//! Supported providers:
//! - OpenAI (GPT-4, GPT-3.5, etc.)
//! - Anthropic (Claude)
//! - Gemini
//!
//! # Example
//! ```rust,ignore
//! use agentsight::analyzer::token::{TokenParser, TokenUsage};
//! use agentsight::parser::ParsedSseEvent;
//!
//! let parser = TokenParser::new();
//! for event in sse_events {
//!     if let Some(usage) = parser.parse_event(&event) {
//!         println!("Tokens: {} in, {} out", usage.input_tokens, usage.output_tokens);
//!     }
//! }
//! ```

use super::{LLMProvider, TokenUsage, detect_provider_from_usage, extract_usage_object};
use crate::parser::sse::ParsedSseEvent;
use std::borrow::Cow;

/// Standard protocol envelope keys whose object values may hold or wrap a
/// canonical `usage` / `usageMetadata` block.
const PROTOCOL_WRAPPER_KEYS: &[&str] = &[
    "response", "message", "data", "event", "delta", "payload", "chunk", "result",
];

/// Maximum nesting depth of protocol wrapper objects above `usage` /
/// `usageMetadata` (e.g. `data -> event -> response -> usage`).
const MAX_WRAPPER_DEPTH: usize = 4;

fn is_protocol_wrapper_key(key: &str) -> bool {
    PROTOCOL_WRAPPER_KEYS.contains(&key)
}

fn is_usage_container_key(key: &str) -> bool {
    matches!(key, "usage" | "usageMetadata" | "usage_metadata")
}

/// Keys recognized directly inside a canonical or bare usage object.
///
/// Restricting bare-root recovery to objects whose top-level keys all belong to
/// this set keeps existing bare-usage fragment tests working while preventing
/// arbitrary non-usage or schema objects from being mistaken for token usage.
fn is_bare_usage_key(key: &str) -> bool {
    matches!(
        key,
        "input_tokens"
            | "output_tokens"
            | "total_tokens"
            | "prompt_tokens"
            | "completion_tokens"
            | "prompt_tokens_details"
            | "completion_tokens_details"
            | "input_tokens_details"
            | "output_tokens_details"
            | "cache_creation_input_tokens"
            | "cache_read_input_tokens"
            | "cached_tokens"
            | "cache_creation"
            | "cache_type"
            | "promptTokenCount"
            | "candidatesTokenCount"
            | "totalTokenCount"
            | "thoughtsTokenCount"
            | "cachedContentTokenCount"
            | "toolUsePromptTokenCount"
            | "prompt_token_count"
            | "candidates_token_count"
            | "total_token_count"
            | "thoughts_token_count"
            | "cached_content_token_count"
            | "tool_use_prompt_token_count"
            | "promptTokensDetails"
            | "candidatesTokensDetails"
            | "cacheTokensDetails"
            | "toolUsePromptTokensDetails"
            | "trafficType"
            | "image_tokens"
            | "audio_tokens"
            | "video_tokens"
            | "text_tokens"
            | "reasoning_tokens"
            | "service_tier"
            | "server_tool_use"
            | "inference_geo"
            | "x_details"
    )
}

#[derive(Default)]
struct RawUsageCounters {
    input_tokens: Option<u64>,
    output_tokens: Option<u64>,
    prompt_tokens: Option<u64>,
    completion_tokens: Option<u64>,
    prompt_token_count: Option<u64>,
    prompt_token_count_snake: Option<u64>,
    candidates_token_count: Option<u64>,
    candidates_token_count_snake: Option<u64>,
    thoughts_token_count: Option<u64>,
    thoughts_token_count_snake: Option<u64>,
    tool_use_prompt_token_count: Option<u64>,
    tool_use_prompt_token_count_snake: Option<u64>,
    cache_creation_top: Option<u64>,
    cache_creation_details: Option<u64>,
    cache_read_anthropic: Option<u64>,
    cached_tokens_prompt_details: Option<u64>,
    cached_tokens_input_details: Option<u64>,
    cached_content_token_count: Option<u64>,
    cached_content_token_count_snake: Option<u64>,
    cached_tokens_top: Option<u64>,
    has_responses_details: bool,
    has_dashscope_modality: bool,
}

impl RawUsageCounters {
    fn set_first(slot: &mut Option<u64>, value: Option<u64>) {
        if slot.is_none() {
            *slot = value;
        }
    }

    fn into_token_usage(self) -> Option<TokenUsage> {
        // Gemini's wire counters are camelCase (`promptTokenCount` /
        // `candidatesTokenCount` under `usageMetadata`); the snake_case
        // spellings are the gateway fallback `extract_usage_object`'s Gemini
        // arm keeps, so the scan reads them the same way and in the same order.
        let tool_prompt = self
            .tool_use_prompt_token_count
            .or(self.tool_use_prompt_token_count_snake)
            .unwrap_or(0);
        let gemini_input = self
            .prompt_token_count
            .or(self.prompt_token_count_snake)
            .map(|prompt| prompt.saturating_add(tool_prompt));
        // Thinking models report the reasoning budget in a counter of its own,
        // outside `candidatesTokenCount`, while `totalTokenCount` covers both —
        // fold it into output just like `extract_usage_object`'s Gemini arm.
        let thoughts = self
            .thoughts_token_count
            .or(self.thoughts_token_count_snake);
        let gemini_output = self
            .candidates_token_count
            .or(self.candidates_token_count_snake)
            .map(|candidates| candidates.saturating_add(thoughts.unwrap_or(0)))
            .or(thoughts);
        let is_gemini = gemini_input.is_some() || gemini_output.is_some();

        let input = self.input_tokens.or(self.prompt_tokens).or(gemini_input);
        let output = self
            .output_tokens
            .or(self.completion_tokens)
            .or(gemini_output);
        if input.is_none() && output.is_none() {
            return None;
        }

        let cache_creation_input_tokens = self.cache_creation_top.or(self.cache_creation_details);
        let cache_read_input_tokens = self
            .cache_read_anthropic
            .or(self.cached_tokens_prompt_details)
            .or(self.cached_tokens_input_details)
            .or(self.cached_content_token_count)
            .or(self.cached_content_token_count_snake)
            .or(self.cached_tokens_top);

        // Provider decides whether the cache counters are billed on top of the
        // input count (Anthropic) or sit inside it (OpenAI-compatible / Gemini /
        // DashScope). DashScope compatible mode nests
        // `cache_creation_input_tokens` under `prompt_tokens_details` alongside
        // `prompt_tokens`, and Responses API nests `cached_tokens` under
        // `input_tokens_details`; neither may be misclassified as Anthropic.
        let provider = if is_gemini {
            LLMProvider::Gemini
        } else if self.input_tokens.is_some() && self.has_dashscope_modality {
            LLMProvider::DashScope
        } else if self.prompt_tokens.is_none()
            && !self.has_responses_details
            && (self.cache_creation_top.is_some() || self.cache_read_anthropic.is_some())
        {
            LLMProvider::Anthropic
        } else {
            LLMProvider::OpenAI
        };

        Some(TokenUsage {
            input_tokens: input.unwrap_or(0),
            output_tokens: output.unwrap_or(0),
            cache_creation_input_tokens,
            cache_read_input_tokens,
            model: None,
            provider,
        })
    }
}

/// Structural partial-JSON scanner that recovers token counters only from
/// protocol usage locations (or bare top-level usage fragments), skipping
/// strings, escapes, arrays, and unrelated subtrees such as echoed tool schemas.
struct PartialUsageScanner<'a> {
    src: &'a str,
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> PartialUsageScanner<'a> {
    fn new(src: &'a str) -> Self {
        Self {
            src,
            bytes: src.as_bytes(),
            pos: 0,
        }
    }

    fn peek(&self) -> Option<u8> {
        self.bytes.get(self.pos).copied()
    }

    fn skip_ws(&mut self) {
        while let Some(b) = self.peek() {
            if matches!(b, b' ' | b'\t' | b'\r' | b'\n') {
                self.pos += 1;
            } else {
                break;
            }
        }
    }

    fn skip_string(&mut self) -> bool {
        if self.peek() != Some(b'"') {
            return false;
        }
        self.pos += 1;
        while let Some(b) = self.peek() {
            match b {
                b'\\' => {
                    self.pos += 1;
                    if self.pos < self.bytes.len() {
                        self.pos += 1;
                    } else {
                        return false;
                    }
                }
                b'"' => {
                    self.pos += 1;
                    return true;
                }
                _ => {
                    self.pos += 1;
                }
            }
        }
        false
    }

    fn parse_string_key(&mut self) -> Option<Cow<'a, str>> {
        if self.peek() != Some(b'"') {
            return None;
        }
        let quote_start = self.pos;
        self.pos += 1;
        let content_start = self.pos;
        let mut has_escape = false;
        while let Some(b) = self.peek() {
            match b {
                b'\\' => {
                    has_escape = true;
                    self.pos += 1;
                    if self.pos < self.bytes.len() {
                        self.pos += 1;
                    } else {
                        return None;
                    }
                }
                b'"' => {
                    let content_end = self.pos;
                    self.pos += 1;
                    if !has_escape {
                        return Some(Cow::Borrowed(&self.src[content_start..content_end]));
                    }
                    let raw_literal = &self.src[quote_start..self.pos];
                    return serde_json::from_str::<String>(raw_literal)
                        .ok()
                        .map(Cow::Owned);
                }
                _ => {
                    self.pos += 1;
                }
            }
        }
        None
    }

    fn skip_primitive(&mut self) -> bool {
        let start = self.pos;
        while let Some(b) = self.peek() {
            if matches!(
                b,
                b' ' | b'\t' | b'\r' | b'\n' | b',' | b'}' | b']' | b':' | b'{' | b'[' | b'"'
            ) {
                break;
            }
            self.pos += 1;
        }
        self.pos > start
    }

    /// Structurally skip one JSON value using an explicit delimiter stack so
    /// arbitrarily nested tool schemas or arrays never overflow the call stack.
    fn skip_value(&mut self) -> bool {
        self.skip_ws();
        let Some(first) = self.peek() else {
            return false;
        };
        match first {
            b'"' => self.skip_string(),
            b'{' | b'[' => {
                let mut expected_closers: Vec<u8> = Vec::new();
                while let Some(b) = self.peek() {
                    match b {
                        b'"' => {
                            if !self.skip_string() {
                                return false;
                            }
                        }
                        b'{' => {
                            self.pos += 1;
                            expected_closers.push(b'}');
                        }
                        b'[' => {
                            self.pos += 1;
                            expected_closers.push(b']');
                        }
                        b'}' | b']' => {
                            if expected_closers.last() != Some(&b) {
                                return false;
                            }
                            expected_closers.pop();
                            self.pos += 1;
                            if expected_closers.is_empty() {
                                return true;
                            }
                        }
                        _ => {
                            self.pos += 1;
                        }
                    }
                }
                false
            }
            _ => self.skip_primitive(),
        }
    }

    /// Parse a non-negative JSON integer value at the current cursor, or
    /// structurally skip any non-u64 value (string, float, negative, object,
    /// array, boolean, null) so sibling keys remain aligned.
    fn parse_u64_or_skip(&mut self) -> Option<u64> {
        self.skip_ws();
        let first = self.peek()?;
        if !first.is_ascii_digit() {
            self.skip_value();
            return None;
        }
        let start = self.pos;
        while let Some(b) = self.peek() {
            if b.is_ascii_digit() {
                self.pos += 1;
            } else {
                break;
            }
        }
        let digits = &self.src[start..self.pos];
        if let Some(next) = self.peek() {
            if !matches!(next, b' ' | b'\t' | b'\r' | b'\n' | b',' | b'}' | b']') {
                self.skip_primitive();
                return None;
            }
        }
        if digits.len() > 1 && digits.starts_with('0') {
            return None;
        }
        digits.parse::<u64>().ok()
    }

    fn parse_details_object(&mut self, counters: &mut RawUsageCounters, is_input_details: bool) {
        if self.peek() != Some(b'{') {
            self.skip_value();
            return;
        }
        self.pos += 1;
        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                Some(b'"') => {}
                _ => break,
            }
            let Some(key) = self.parse_string_key() else {
                break;
            };
            self.skip_ws();
            if self.peek() != Some(b':') {
                break;
            }
            self.pos += 1;
            self.skip_ws();
            match key.as_ref() {
                "cached_tokens" => {
                    let val = self.parse_u64_or_skip();
                    if is_input_details {
                        RawUsageCounters::set_first(&mut counters.cached_tokens_input_details, val);
                    } else {
                        RawUsageCounters::set_first(
                            &mut counters.cached_tokens_prompt_details,
                            val,
                        );
                    }
                }
                "cache_creation_input_tokens" if !is_input_details => {
                    let val = self.parse_u64_or_skip();
                    RawUsageCounters::set_first(&mut counters.cache_creation_details, val);
                }
                _ => {
                    if !self.skip_value() {
                        break;
                    }
                }
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => break,
            }
        }
    }

    fn consume_usage_field(&mut self, key: &str, counters: &mut RawUsageCounters) -> bool {
        match key {
            "input_tokens" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.input_tokens, val);
            }
            "output_tokens" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.output_tokens, val);
            }
            "prompt_tokens" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.prompt_tokens, val);
            }
            "completion_tokens" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.completion_tokens, val);
            }
            "promptTokenCount" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.prompt_token_count, val);
            }
            "prompt_token_count" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.prompt_token_count_snake, val);
            }
            "candidatesTokenCount" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.candidates_token_count, val);
            }
            "candidates_token_count" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.candidates_token_count_snake, val);
            }
            "thoughtsTokenCount" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.thoughts_token_count, val);
            }
            "thoughts_token_count" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.thoughts_token_count_snake, val);
            }
            "toolUsePromptTokenCount" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.tool_use_prompt_token_count, val);
            }
            "tool_use_prompt_token_count" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.tool_use_prompt_token_count_snake, val);
            }
            "cache_creation_input_tokens" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.cache_creation_top, val);
            }
            "cache_read_input_tokens" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.cache_read_anthropic, val);
            }
            "cached_tokens" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.cached_tokens_top, val);
            }
            "cachedContentTokenCount" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.cached_content_token_count, val);
            }
            "cached_content_token_count" => {
                let val = self.parse_u64_or_skip();
                RawUsageCounters::set_first(&mut counters.cached_content_token_count_snake, val);
            }
            "prompt_tokens_details" => {
                self.parse_details_object(counters, false);
            }
            "input_tokens_details" => {
                if self.peek() == Some(b'{') {
                    counters.has_responses_details = true;
                    self.parse_details_object(counters, true);
                } else {
                    return self.skip_value();
                }
            }
            "output_tokens_details" => {
                if self.peek() == Some(b'{') {
                    counters.has_responses_details = true;
                }
                return self.skip_value();
            }
            "image_tokens" | "audio_tokens" | "video_tokens" => {
                if self.parse_u64_or_skip().is_some() {
                    counters.has_dashscope_modality = true;
                }
            }
            _ => {
                return self.skip_value();
            }
        }
        true
    }

    fn parse_usage_object(&mut self) -> Option<TokenUsage> {
        if self.peek() != Some(b'{') {
            self.skip_value();
            return None;
        }
        self.pos += 1;
        let mut counters = RawUsageCounters::default();
        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                Some(b'"') => {}
                _ => break,
            }
            let Some(key) = self.parse_string_key() else {
                break;
            };
            self.skip_ws();
            if self.peek() != Some(b':') {
                break;
            }
            self.pos += 1;
            self.skip_ws();
            if !self.consume_usage_field(key.as_ref(), &mut counters) {
                break;
            }
            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => break,
            }
        }
        counters.into_token_usage()
    }

    fn scan_envelope_object(&mut self, depth: usize) -> Option<TokenUsage> {
        if self.peek() != Some(b'{') {
            return None;
        }
        self.pos += 1;
        let mut root_is_bare_usage = depth == 0;
        let mut bare_counters = RawUsageCounters::default();

        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                Some(b'"') => {}
                _ => {
                    root_is_bare_usage = false;
                    break;
                }
            }
            let Some(key) = self.parse_string_key() else {
                break;
            };
            self.skip_ws();
            if self.peek() != Some(b':') {
                break;
            }
            self.pos += 1;
            self.skip_ws();

            if is_usage_container_key(key.as_ref()) {
                root_is_bare_usage = false;
                if self.peek() == Some(b'{') {
                    if let Some(usage) = self.parse_usage_object() {
                        return Some(usage);
                    }
                } else if !self.skip_value() {
                    break;
                }
            } else if depth < MAX_WRAPPER_DEPTH && is_protocol_wrapper_key(key.as_ref()) {
                root_is_bare_usage = false;
                if self.peek() == Some(b'{') {
                    if let Some(usage) = self.scan_envelope_object(depth + 1) {
                        return Some(usage);
                    }
                } else if !self.skip_value() {
                    break;
                }
            } else if root_is_bare_usage && is_bare_usage_key(key.as_ref()) {
                if !self.consume_usage_field(key.as_ref(), &mut bare_counters) {
                    break;
                }
            } else {
                root_is_bare_usage = false;
                if !self.skip_value() {
                    break;
                }
            }

            self.skip_ws();
            match self.peek() {
                Some(b',') => {
                    self.pos += 1;
                }
                Some(b'}') => {
                    self.pos += 1;
                    break;
                }
                _ => break,
            }
        }

        if root_is_bare_usage {
            bare_counters.into_token_usage()
        } else {
            None
        }
    }

    /// Scan a brace-less top-level bare usage fragment such as
    /// `"input_tokens": 42, "output_tokens": 7`. Rejects any non-usage key or
    /// stray closing brace so truncated fragments inside schema examples never
    /// match.
    fn scan_braceless_bare_usage(&mut self) -> Option<TokenUsage> {
        let mut counters = RawUsageCounters::default();
        let mut saw_field = false;
        loop {
            self.skip_ws();
            match self.peek() {
                None => break,
                Some(b'"') => {}
                _ => return None,
            }
            let Some(key) = self.parse_string_key() else {
                break;
            };
            if !is_bare_usage_key(key.as_ref()) {
                return None;
            }
            self.skip_ws();
            if self.peek() != Some(b':') {
                return None;
            }
            self.pos += 1;
            self.skip_ws();
            saw_field = true;
            if !self.consume_usage_field(key.as_ref(), &mut counters) {
                break;
            }
            self.skip_ws();
            match self.peek() {
                None => break,
                Some(b',') => {
                    self.pos += 1;
                }
                _ => return None,
            }
        }
        if saw_field {
            counters.into_token_usage()
        } else {
            None
        }
    }
}

fn scan_json_payload(payload: &str) -> Option<TokenUsage> {
    let trimmed = payload
        .strip_prefix('\u{feff}')
        .unwrap_or(payload)
        .trim_start();
    if trimmed.is_empty() || trimmed == "[DONE]" || trimmed == "[END]" {
        return None;
    }

    let mut scanner = PartialUsageScanner::new(trimmed);
    if scanner.peek() == Some(b'{') {
        while let Some(b) = scanner.peek() {
            if b != b'{' {
                break;
            }
            let before = scanner.pos;
            if let Some(usage) = scanner.scan_envelope_object(0) {
                return Some(usage);
            }
            if scanner.pos == before {
                break;
            }
            scanner.skip_ws();
        }
        return None;
    }

    if scanner.peek() == Some(b'"') {
        return scanner.scan_braceless_bare_usage();
    }

    None
}

fn merge_partial_usage(acc: Option<TokenUsage>, next: TokenUsage) -> Option<TokenUsage> {
    let Some(cur) = acc else {
        return Some(next);
    };
    let cur_empty = cur.is_empty();
    let next_provider = next.provider;
    let next_empty = next.is_empty();
    let mut merged = super::merge_usage(Some(cur), next)?;
    if cur_empty && !next_empty && next_provider != LLMProvider::Unknown {
        merged.provider = next_provider;
    }
    Some(merged)
}

fn scan_sse_frame(frame_data: &str) -> Option<TokenUsage> {
    let trimmed = frame_data.trim();
    if trimmed.is_empty() || trimmed == "[DONE]" || trimmed == "[END]" {
        return None;
    }
    if let Ok(json) = serde_json::from_str::<serde_json::Value>(trimmed) {
        if let Some(usage) = TokenParser::new().parse_json(&json) {
            return Some(usage);
        }
    }
    scan_json_payload(trimmed)
}

/// Token parser for extracting usage from SSE events
pub struct TokenParser;

impl TokenParser {
    /// Create a new token parser
    pub fn new() -> Self {
        TokenParser
    }

    /// Parse token usage from a ParsedSseEvent
    ///
    /// Returns `Some(TokenUsage)` if the event contains usage information,
    /// `None` otherwise.
    pub fn parse_event(&self, event: &ParsedSseEvent) -> Option<TokenUsage> {
        // Get event data as string
        let data = event.data();
        let data_str = std::str::from_utf8(data).ok()?;

        // Parse as JSON
        self.parse_data(data_str)
    }

    /// Parse token usage from raw SSE data string
    ///
    /// This is useful when you have raw SSE data without a ParsedSseEvent.
    pub fn parse_data(&self, data: &str) -> Option<TokenUsage> {
        // Skip done markers
        if data.trim() == "[DONE]" || data.trim() == "[END]" {
            return None;
        }

        // Try strict JSON parse first
        if let Ok(json) = serde_json::from_str::<serde_json::Value>(data) {
            return self.parse_json(&json).inspect(|_usage| {
                log::debug!("token usage parsed from data: {data}");
            });
        }

        // Fallback: OpenAI Responses API embeds usage in a final
        // `response.completed` event whose payload (instructions + tools +
        // output) routinely exceeds a single TLS record. We may be looking
        // at a concatenation of SSE chunks that together don't form a
        // single valid JSON object. Recover input/output token counts via
        // a regex-free string scan when the buffer references usage fields.
        // Gemini's counters are camelCase under `usageMetadata`
        // (`promptTokenCount`/`candidatesTokenCount`, 2e018e7d9), so the
        // gate must recognize them too or a Gemini chunk split across TLS
        // records recovers nothing while every other provider still does.
        // Its snake_case spellings are the gateway fallback the direct parse
        // keeps, and they have to be gated on for the same reason: a buffer
        // that names them is turned away before the scan ever sees it.
        if data.contains("\"input_tokens\"")
            || data.contains("\"output_tokens\"")
            || data.contains("\"prompt_tokens\"")
            || data.contains("\"completion_tokens\"")
            || data.contains("\"promptTokenCount\"")
            || data.contains("\"candidatesTokenCount\"")
            || data.contains("\"prompt_token_count\"")
            || data.contains("\"candidates_token_count\"")
            || data.contains("\"thoughtsTokenCount\"")
            || data.contains("\"thoughts_token_count\"")
        {
            let usage = Self::scan_partial_usage(data);
            if usage.is_some() {
                log::debug!("token usage recovered from continuation buffer");
            }
            return usage;
        }

        None
    }

    /// Recover token usage fields from a possibly-truncated JSON or SSE
    /// continuation buffer by walking JSON structure (strings, escapes,
    /// braces, brackets, and object key paths) rather than raw text matching.
    ///
    /// Counters are extracted only from canonical protocol `usage` /
    /// `usageMetadata` objects (at root or inside standard protocol wrappers
    /// such as `response`, `message`, `data`, `event`, `delta`, `payload`,
    /// `chunk`, or `result`) or from bare top-level usage fragments. Unrelated
    /// subtrees such as echoed tool schemas (`tools`, `parameters`,
    /// `properties`, `default`, `examples`, etc.) and string literals are
    /// skipped structurally so numeric schema examples neither override real
    /// usage nor invent usage before the protocol usage block arrives.
    fn scan_partial_usage(data: &str) -> Option<TokenUsage> {
        let stripped = data.strip_prefix('\u{feff}').unwrap_or(data);
        let trimmed = stripped.trim_start();
        if trimmed.is_empty() {
            return None;
        }

        // Leading JSON object(s) may appear without a `data:` prefix either
        // for direct truncated JSON buffers or when a TLS split landed right
        // after an earlier event's `data: ` prefix; continue scanning any
        // subsequent SSE `data:` lines in the same continuation buffer.
        let starts_with_brace = trimmed.starts_with('{');
        let mut merged = if starts_with_brace {
            scan_json_payload(trimmed)
        } else {
            None
        };

        // Extract SSE frames (`data:` lines delimited by blank lines, plus any
        // trailing unterminated frame at EOF) so multi-event continuation
        // buffers are scanned frame-by-frame and merged across cumulative
        // snapshots (preventing an earlier zero placeholder from masking a
        // later terminal usage frame).
        let mut saw_data_line = false;
        let mut current_data = String::new();
        for raw_line in stripped.split('\n') {
            let line = raw_line.strip_suffix('\r').unwrap_or(raw_line);
            if line.is_empty() {
                if !current_data.is_empty() {
                    if let Some(usage) = scan_sse_frame(&current_data) {
                        merged = merge_partial_usage(merged, usage);
                    }
                    current_data.clear();
                }
                continue;
            }
            if let Some(rest) = line.strip_prefix("data:") {
                saw_data_line = true;
                let value = rest.strip_prefix(' ').unwrap_or(rest);
                if !current_data.is_empty() {
                    current_data.push('\n');
                }
                current_data.push_str(value);
            }
        }
        if !current_data.is_empty() {
            if let Some(usage) = scan_sse_frame(&current_data) {
                merged = merge_partial_usage(merged, usage);
            }
        }
        if saw_data_line || starts_with_brace {
            return merged;
        }

        // Fallback for bare top-level usage fragments without SSE framing or
        // an outer `{` (e.g. `"input_tokens": 42, "output_tokens": 7`).
        scan_json_payload(trimmed)
    }

    /// Internal method to parse JSON and extract token usage
    pub fn parse_json(&self, json: &serde_json::Value) -> Option<TokenUsage> {
        if let Some(inner) = crate::parser::llm::unwrap_qoder_envelope(json) {
            let usage = inner.get("usage")?;
            return extract_usage_object(usage, detect_provider_from_usage(usage), &inner);
        }
        // 1. Check for message_start event (Anthropic streaming)
        if json.get("type").and_then(|v| v.as_str()) == Some("message_start") {
            if let Some(message) = json.get("message") {
                if let Some(usage) = message.get("usage") {
                    return extract_usage_object(usage, LLMProvider::Anthropic, json);
                }
            }
        }

        // 2. Check for message_delta event (Anthropic streaming final)
        if json.get("type").and_then(|v| v.as_str()) == Some("message_delta") {
            if let Some(usage) = json.get("usage") {
                return extract_usage_object(usage, LLMProvider::Anthropic, json);
            }
        }

        // 3. DashScope/Bailian native protocol envelope.
        //
        // Must precede the generic `usage` branch: the native `usage` field
        // names are identical to Anthropic's, and real Anthropic-gateway
        // traffic even ships the same `total_tokens` sibling, so usage shape
        // alone cannot tell them apart. The envelope can: an `output` *object*
        // paired with `usage` is unique to the native protocol (the OpenAI
        // Responses API nests `output` as an array under `response`).
        if json.get("output").is_some_and(|o| o.is_object()) {
            if let Some(usage) = json.get("usage") {
                return extract_usage_object(usage, LLMProvider::DashScope, json);
            }
        }

        // 4. Gemini: usage rides under `usageMetadata` with camelCase counters
        // (`promptTokenCount`/`candidatesTokenCount`), never under `usage`, and
        // the model under `modelVersion`. The endpoint detector already knows
        // generativelanguage.googleapis.com, but without this branch the
        // parser never read the metadata, so every Gemini stream produced no
        // token record at all.
        if let Some(usage) = json.get("usageMetadata").filter(|u| u.is_object()) {
            return extract_usage_object(usage, LLMProvider::Gemini, json);
        }

        // 5. Check for usage object directly (OpenAI and compatible APIs)
        if let Some(usage) = json.get("usage") {
            let provider = detect_provider_from_usage(usage);
            return extract_usage_object(usage, provider, json);
        }

        // 6. Responses API: usage nested in the terminal response event —
        // `response.completed`, or `response.incomplete` when the output cap
        // cut the stream. The terminal event carries the final usage either
        // way (the live message parser reads both, e534bec1b). The event
        // type is the endpoint knowledge detect_provider_from_usage lacks:
        // the Responses usage object reuses Anthropic's field names, and one
        // without `*_details` would otherwise be mislabeled Anthropic and
        // double-count its cache hits in billed_input_tokens.
        if matches!(
            json.get("type").and_then(|v| v.as_str()),
            Some("response.completed" | "response.incomplete")
        ) {
            if let Some(resp) = json.get("response") {
                if let Some(usage) = resp.get("usage") {
                    return extract_usage_object(usage, LLMProvider::OpenAI, json);
                }
            }
        }

        None
    }
}

impl Default for TokenParser {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Create a mock ParsedSseEvent for testing
    ///
    /// Note: SslEvent is large (~512KB), so we allocate it on the heap
    fn create_test_event(data: &str) -> ParsedSseEvent {
        use crate::probes::sslsniff::SslEvent;
        use std::rc::Rc;

        let ssl_event = Rc::new(SslEvent {
            source: 0,
            timestamp_ns: 1234567890,
            delta_ns: 0,
            pid: 1234,
            tid: 5678,
            uid: 0,
            len: data.len() as u32,
            rw: 0,
            comm: String::new(),
            buf: data.as_bytes().to_vec(),
            is_handshake: false,
            ssl_ptr: 0,
        });

        ParsedSseEvent::new(None, None, None, 0, data.len(), ssl_event)
    }

    /// DashScope/Bailian native protocol is identified by its **response
    /// envelope** (`output` object + `request_id`, no top-level `model`), not by
    /// the shape of `usage` — the native `usage` field names are
    /// indistinguishable from Anthropic-gateway traffic that also reports
    /// `total_tokens`.
    /// Envelope shape taken verbatim from a real non-streaming native response.
    #[test]
    fn test_parse_dashscope_native_envelope() {
        let parser = TokenParser::new();
        let data = r#"{
            "output": {"choices": [{
                "message": {"content": "收到了", "role": "assistant"},
                "finish_reason": "stop"
            }]},
            "usage": {"total_tokens": 26, "output_tokens": 1, "input_tokens": 25,
                      "prompt_tokens_details": {"cached_tokens": 0}},
            "request_id": "b82b259c-ed1d-95bd-8c0e-4f357868a622"
        }"#;
        let usage = parser
            .parse_data(data)
            .expect("native envelope must yield usage");
        assert_eq!(usage.provider, LLMProvider::DashScope);
        assert_eq!(usage.input_tokens, 25);
        assert_eq!(usage.output_tokens, 1);
    }

    /// `result_format: text` puts the answer on `output.text`.
    #[test]
    fn test_parse_dashscope_native_envelope_text_format() {
        let parser = TokenParser::new();
        let data = r#"{
            "output": {"finish_reason": "stop", "text": "hi"},
            "usage": {"input_tokens": 5, "output_tokens": 2, "total_tokens": 7},
            "request_id": "req-2"
        }"#;
        let usage = parser.parse_data(data).expect("usage");
        assert_eq!(usage.provider, LLMProvider::DashScope);
    }

    /// multimodal-generation itemises the input side; `image_tokens` is already
    /// counted inside `input_tokens` (real values: 1249 image + 12 text = 1261).
    #[test]
    fn test_parse_dashscope_native_envelope_multimodal_input_tokens() {
        let parser = TokenParser::new();
        let data = r#"{
            "output": {"choices": [{"finish_reason": "stop",
                "message": {"role": "assistant", "content": [{"text": "cat"}],
                            "reasoning_content": ""}}]},
            "usage": {
                "input_tokens_details": {"image_tokens": 1249, "text_tokens": 12},
                "prompt_tokens_details": {"cached_tokens": 0},
                "total_tokens": 1643,
                "output_tokens": 382,
                "input_tokens": 1261,
                "image_tokens": 1249
            },
            "request_id": "req-3"
        }"#;
        let usage = parser.parse_data(data).expect("usage");
        assert_eq!(usage.provider, LLMProvider::DashScope);
        assert_eq!(
            usage.input_tokens, 1261,
            "no double counting of image tokens"
        );
        assert_eq!(usage.output_tokens, 382);
        assert_eq!(usage.total_tokens(), 1643, "reconciles with total_tokens");
    }

    /// Gemini `streamGenerateContent` chunks carry usage under
    /// `usageMetadata` with camelCase counters — never under `usage` — and the
    /// model under `modelVersion`. Shape from generativelanguage.googleapis.com
    /// traffic; the endpoint detector already recognizes that host, but the
    /// parser never read the metadata, so every Gemini stream produced no
    /// token record at all.
    #[test]
    fn test_parse_gemini_usage_metadata() {
        let parser = TokenParser::new();
        let data = r#"{
            "candidates": [
                {"content": {"parts": [{"text": "Hello"}], "role": "model"},
                 "finishReason": "STOP", "index": 0}
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 5,
                              "totalTokenCount": 15, "cachedContentTokenCount": 3},
            "modelVersion": "gemini-2.0-flash-001"
        }"#;
        let usage = parser
            .parse_data(data)
            .expect("gemini usage must be parsed");
        assert_eq!(usage.provider, LLMProvider::Gemini);
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        assert_eq!(usage.cache_read_input_tokens, Some(3));
        assert_eq!(usage.model.as_deref(), Some("gemini-2.0-flash-001"));
    }

    /// Streaming chunks that only carry a text delta have no
    /// `usageMetadata`; they must not yield a usage record (the caller merges
    /// every parseable chunk and takes the max of each cumulative counter).
    #[test]
    fn test_parse_gemini_chunk_without_usage_metadata_yields_no_usage() {
        let parser = TokenParser::new();
        let data = r#"{
            "candidates": [
                {"content": {"parts": [{"text": "Hel"}], "role": "model"}, "index": 0}
            ]
        }"#;
        assert!(parser.parse_data(data).is_none());
    }

    /// The strict JSON path reads `usageMetadata` (tests above), but the
    /// continuation-buffer fallback scanned snake_case keys only: a Gemini
    /// chunk split across TLS records — the exact scenario
    /// `scan_partial_usage` exists for — recovered no tokens at all while
    /// every other provider still did.
    #[test]
    fn test_scan_partial_usage_gemini_camel_case_counters() {
        let data = r#"data:{"candidates":[{"content":{"parts":[{"text":"Hel"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":57,"candidatesTokenCount":3,"totalTokenCount":60,"cachedContentTokenCount":2"#;
        let parser = TokenParser::new();
        let usage = parser
            .parse_data(data)
            .expect("truncated gemini usage should still parse");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);
        assert_eq!(usage.cache_read_input_tokens, Some(2));
        // Gemini bills the cached prefix inside `promptTokenCount` (the
        // OpenAI formula), so the provider label must be Gemini — never
        // Anthropic, whose formula would add the cache on top a second time.
        assert_eq!(usage.provider, LLMProvider::Gemini);
    }

    /// Thinking models report their reasoning budget in `thoughtsTokenCount`,
    /// outside `candidatesTokenCount`, while `totalTokenCount` covers both (the
    /// reasoning budget is billed as output). Reading only the candidate
    /// counter under-reported such a call by orders of magnitude — 1 output
    /// token instead of 816 for the shape below — and broke the
    /// `input + output == total` reconciliation every other provider arm keeps
    /// (asserted for DashScope above).
    #[test]
    fn test_parse_gemini_thinking_usage_reconciles_with_total() {
        let parser = TokenParser::new();
        let data = r#"{
            "candidates": [
                {"content": {"parts": [{"text": "4"}], "role": "model"},
                 "finishReason": "STOP", "index": 0}
            ],
            "usageMetadata": {"promptTokenCount": 10, "candidatesTokenCount": 1,
                              "thoughtsTokenCount": 815, "totalTokenCount": 826},
            "modelVersion": "gemini-2.5-flash"
        }"#;
        let usage = parser
            .parse_data(data)
            .expect("gemini usage must be parsed");
        assert_eq!(
            usage.output_tokens, 816,
            "thinking tokens are billed as output tokens"
        );
        assert_eq!(usage.total_tokens(), 826, "reconciles with totalTokenCount");
    }

    /// Gemini's counters have two spellings: the wire format is camelCase
    /// (`promptTokenCount`/`candidatesTokenCount` under `usageMetadata`), and
    /// the snake_case spellings are the gateway fallback `extract_usage_object`
    /// keeps (the provider detector reads `prompt_token_count` as the Gemini
    /// signal too). The continuation recovery was taught the camelCase pair
    /// only, and its gate runs before the scan: a truncated chunk from such a
    /// gateway is turned away before anything reads it, so the same payload
    /// that parses cleanly one line above records no usage at all.
    #[test]
    fn test_scan_partial_usage_gemini_gateway_spellings() {
        let parser = TokenParser::new();
        // The same payload the gateway spells snake_case, complete.
        let complete = r#"{"candidates":[{"content":{"parts":[{"text":"Hel"}],"role":"model"},"index":0}],"usageMetadata":{"prompt_token_count":57,"candidates_token_count":3,"total_token_count":60},"modelVersion":"gemini-2.5-flash"}"#;
        let parsed = parser.parse_data(complete).expect("complete usage parses");
        assert_eq!(parsed.input_tokens, 57);
        assert_eq!(parsed.output_tokens, 3);
        assert_eq!(parsed.provider, LLMProvider::Gemini);

        // The last TLS record was cut, so the strict parse fails; the
        // continuation recovery must reach the same counters.
        let truncated = r#"data:{"candidates":[{"content":{"parts":[{"text":"Hel"}],"role":"model"},"index":0}],"usageMetadata":{"prompt_token_count":57,"candidates_token_count":3,"total_token_count":60"#;
        let usage = parser
            .parse_data(truncated)
            .expect("truncated gateway-spelled gemini usage must still parse");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);
        assert_eq!(
            usage.provider,
            LLMProvider::Gemini,
            "the gateway spelling still bills like Gemini (cache sits inside the prompt count)"
        );
    }

    /// The direct parse folds `thoughtsTokenCount` into the output count
    /// (`test_parse_gemini_thinking_usage_reconciles_with_total`); the
    /// continuation-buffer scan read only the candidate counter, so a
    /// thinking model's chunk split across TLS records recovered 1 output
    /// token where the same payload parsed whole reported 816 — the
    /// recovery is the only usage source for a call that ended there.
    #[test]
    fn test_scan_partial_usage_gemini_thinking_counts_reasoning_budget() {
        // Deliberately not valid JSON: the last TLS record was cut.
        let data = r#"data:{"candidates":[{"content":{"parts":[{"text":"4"}],"role":"model"},"index":0}],"usageMetadata":{"promptTokenCount":10,"candidatesTokenCount":1,"thoughtsTokenCount":815,"totalTokenCount":826"#;
        let parser = TokenParser::new();
        let usage = parser
            .parse_data(data)
            .expect("truncated thinking usage must still parse");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(
            usage.output_tokens, 816,
            "thinking tokens are billed as output tokens"
        );
        assert_eq!(usage.total_tokens(), 826, "reconciles with totalTokenCount");
        // Gemini bills the cached prefix inside the prompt count, so the
        // provider label must stay Gemini for the folded shape too.
        assert_eq!(usage.provider, LLMProvider::Gemini);
    }

    /// The gateway spellings (`prompt_token_count` and friends) are the
    /// other form the scan keeps; their thinking counter folds the same
    /// way, so a truncated gateway-spelled chunk answers what the same
    /// payload parsed whole answers.
    #[test]
    fn test_scan_partial_usage_gemini_gateway_thinking_spellings() {
        // Deliberately not valid JSON: the last TLS record was cut.
        let data = r#"data:{"candidates":[{"content":{"parts":[{"text":"4"}],"role":"model"},"index":0}],"usageMetadata":{"prompt_token_count":10,"candidates_token_count":1,"thoughts_token_count":815,"total_token_count":826"#;
        let parser = TokenParser::new();
        let usage = parser
            .parse_data(data)
            .expect("truncated gateway-spelled thinking usage must still parse");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(
            usage.output_tokens, 816,
            "the gateway spelling folds thinking tokens the same way"
        );
        assert_eq!(usage.provider, LLMProvider::Gemini);
    }

    /// Regression guard from **real captured traffic**: Anthropic-protocol
    /// gateway responses carry `total_tokens` in `usage` but have no `output`
    /// envelope, so they must stay Anthropic.
    #[test]
    fn test_parse_anthropic_gateway_usage_with_total_tokens_stays_anthropic() {
        let parser = TokenParser::new();
        let data = r#"{
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {
                "input_tokens": 1111,
                "output_tokens": 803,
                "total_tokens": 26490,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 24576
            }
        }"#;
        let usage = parser.parse_data(data).expect("usage");
        assert_eq!(usage.provider, LLMProvider::Anthropic);
        assert_eq!(usage.input_tokens, 1111);
        assert_eq!(usage.cache_read_input_tokens, Some(24576));
    }

    /// Same shape without the Anthropic `type` marker (non-streaming gateway
    /// response) must also stay Anthropic rather than flipping to DashScope.
    #[test]
    fn test_parse_bare_anthropic_usage_with_total_tokens_stays_anthropic() {
        let parser = TokenParser::new();
        let data = r#"{
            "id": "msg_01",
            "model": "claude-opus-5",
            "usage": {
                "input_tokens": 977,
                "output_tokens": 169,
                "total_tokens": 26746,
                "cache_creation_input_tokens": 0,
                "cache_read_input_tokens": 25600
            }
        }"#;
        let usage = parser.parse_data(data).expect("usage");
        assert_eq!(usage.provider, LLMProvider::Anthropic);
        assert_eq!(usage.model.as_deref(), Some("claude-opus-5"));
    }

    #[test]
    fn test_parse_openai_usage() {
        let parser = TokenParser::new();
        let data = r#"{
            "id": "chatcmpl-123",
            "model": "gpt-4",
            "usage": {
                "prompt_tokens": 100,
                "completion_tokens": 50,
                "total_tokens": 150
            }
        }"#;

        let event = create_test_event(data);
        let usage = parser.parse_event(&event);
        assert!(usage.is_some());

        let usage = usage.unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.total_tokens(), 150);
        assert_eq!(usage.provider, LLMProvider::OpenAI);
        assert_eq!(usage.model, Some("gpt-4".to_string()));
    }

    #[test]
    fn test_parse_anthropic_message_start() {
        let parser = TokenParser::new();
        let data = r#"{
            "type": "message_start",
            "message": {
                "id": "msg_123",
                "model": "claude-3-opus",
                "usage": {
                    "input_tokens": 100,
                    "output_tokens": 0
                }
            }
        }"#;

        let event = create_test_event(data);
        let usage = parser.parse_event(&event);
        assert!(usage.is_some());

        let usage = usage.unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 0);
        assert_eq!(usage.provider, LLMProvider::Anthropic);
        // The model rides inside `message.model`; no streamed Anthropic event
        // carries a top-level `model`, so reading only that key stored every
        // Claude stream with an empty model in the token record.
        assert_eq!(usage.model.as_deref(), Some("claude-3-opus"));
    }

    #[test]
    fn test_parse_anthropic_message_delta() {
        let parser = TokenParser::new();
        let data = r#"{
            "type": "message_delta",
            "delta": {},
            "usage": {
                "input_tokens": 100,
                "output_tokens": 50,
                "cache_creation_input_tokens": 10,
                "cache_read_input_tokens": 20
            }
        }"#;

        let event = create_test_event(data);
        let usage = parser.parse_event(&event);
        assert!(usage.is_some());

        let usage = usage.unwrap();
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_creation_input_tokens, Some(10));
        assert_eq!(usage.cache_read_input_tokens, Some(20));
        assert_eq!(usage.provider, LLMProvider::Anthropic);
    }

    #[test]
    fn test_parse_anthropic_message_delta_output_only() {
        // Official Anthropic message_delta carries only output_tokens; the
        // parser must still surface it (input defaults to 0) so the caller can
        // merge it with message_start rather than dropping the event.
        let parser = TokenParser::new();
        let data = r#"{
            "type": "message_delta",
            "delta": {"stop_reason": "end_turn"},
            "usage": {"output_tokens": 42}
        }"#;

        let event = create_test_event(data);
        let usage = parser
            .parse_event(&event)
            .expect("output-only should parse");
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 42);
        assert_eq!(usage.provider, LLMProvider::Anthropic);
    }

    #[test]
    fn test_parse_openai_sse_streaming() {
        let parser = TokenParser::new();
        let data = r#"{"choices":[],"object":"chat.completion.chunk","usage":{"prompt_tokens":61744,"completion_tokens":61,"total_tokens":61805},"created":1773640825,"model":"qwen3.5-plus","id":"chatcmpl-816f7538-0ac9-98c4-8259-9bade0c2cde7"}"#;

        let event = create_test_event(data);
        let usage = parser.parse_event(&event);
        assert!(
            usage.is_some(),
            "Should extract usage from SSE streaming data"
        );

        let usage = usage.unwrap();
        assert_eq!(usage.input_tokens, 61744);
        assert_eq!(usage.output_tokens, 61);
        assert_eq!(usage.total_tokens(), 61805);
        assert_eq!(usage.provider, LLMProvider::OpenAI);
        assert_eq!(usage.model, Some("qwen3.5-plus".to_string()));
    }

    #[test]
    fn test_parse_openai_usage_with_cached_tokens() {
        // OpenAI nests cache hits under prompt_tokens_details.cached_tokens.
        let parser = TokenParser::new();
        let data = r#"{
            "id": "chatcmpl-456",
            "model": "gpt-4o",
            "usage": {
                "prompt_tokens": 10000,
                "completion_tokens": 50,
                "total_tokens": 10050,
                "prompt_tokens_details": {
                    "cached_tokens": 8500
                }
            }
        }"#;

        let event = create_test_event(data);
        let usage = parser.parse_event(&event).expect("should parse");
        assert_eq!(usage.input_tokens, 10000);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_read_input_tokens, Some(8500));
        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(usage.provider, LLMProvider::OpenAI);
    }

    #[test]
    fn test_parse_dashscope_usage_with_cached_tokens() {
        // DashScope may surface cached_tokens at the top level of the usage
        // object instead of nesting under prompt_tokens_details.
        let parser = TokenParser::new();
        let data = r#"{
            "id": "chatcmpl-789",
            "model": "qwen3.5-plus",
            "usage": {
                "prompt_tokens": 9800,
                "completion_tokens": 120,
                "total_tokens": 9920,
                "cached_tokens": 9000
            }
        }"#;

        let event = create_test_event(data);
        let usage = parser.parse_event(&event).expect("should parse");
        assert_eq!(usage.input_tokens, 9800);
        assert_eq!(usage.output_tokens, 120);
        assert_eq!(usage.cache_read_input_tokens, Some(9000));
        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(usage.provider, LLMProvider::OpenAI);
    }

    #[test]
    fn test_parse_dashscope_prompt_tokens_details_cache_creation() {
        // DashScope nests cache_creation_input_tokens under prompt_tokens_details
        // rather than at the top level of the usage object.
        let parser = TokenParser::new();
        let data = r#"{
            "id": "chatcmpl-ds-001",
            "model": "qwen3.6-plus",
            "usage": {
                "prompt_tokens": 29719,
                "completion_tokens": 435,
                "total_tokens": 30154,
                "completion_tokens_details": {
                    "reasoning_tokens": 382,
                    "text_tokens": 435
                },
                "prompt_tokens_details": {
                    "text_tokens": 29719,
                    "cache_creation": {
                        "ephemeral_5m_input_tokens": 29713
                    },
                    "cache_creation_input_tokens": 29713,
                    "cache_type": "ephemeral",
                    "cached_tokens": 0
                }
            }
        }"#;

        let event = create_test_event(data);
        let usage = parser.parse_event(&event).expect("should parse");
        assert_eq!(usage.input_tokens, 29719);
        assert_eq!(usage.output_tokens, 435);
        assert_eq!(usage.cache_creation_input_tokens, Some(29713));
        assert_eq!(usage.cache_read_input_tokens, Some(0));
    }

    /// A truncated DashScope-compatible body (the continuation buffer left by
    /// a process that died mid-stream) must keep the OpenAI-compatible billing
    /// rule: `prompt_tokens` already contains the cache counters nested under
    /// `prompt_tokens_details`, so adding them again doubles the input.
    #[test]
    fn test_partial_dashscope_usage_keeps_openai_billing() {
        let parser = TokenParser::new();
        // Deliberately not valid JSON: the last TLS record was cut.
        let data = r#"{"id":"chatcmpl-ds-003","model":"qwen3.6-plus","usage":{"prompt_tokens":29719,"completion_tokens":435,"total_tokens":30154,"prompt_tokens_details":{"cache_creation_input_tokens":29713,"cached_tokens":0}"#;

        let usage = parser
            .parse_data(data)
            .expect("partial usage must be recovered");
        assert_eq!(usage.provider, LLMProvider::OpenAI);
        assert_eq!(usage.input_tokens, 29719);
        assert_eq!(usage.cache_creation_input_tokens, Some(29713));

        let record = crate::analyzer::token::record::TokenRecord::new(
            1,
            "qwen".to_string(),
            usage.provider.to_string(),
            usage.input_tokens,
            0,
        )
        .with_cache_tokens(
            usage.cache_creation_input_tokens.unwrap_or(0),
            usage.cache_read_input_tokens.unwrap_or(0),
        );
        assert_eq!(
            record.billed_input_tokens(),
            29719,
            "nested cache counters are already part of prompt_tokens"
        );
    }

    #[test]
    fn test_parse_dashscope_usage_without_prompt_tokens_details() {
        // DashScope response lacking prompt_tokens_details — cache fields
        // must remain None, preserving pre-fix behavior.
        let parser = TokenParser::new();
        let data = r#"{
            "id": "chatcmpl-ds-002",
            "model": "qwen3.6-plus",
            "usage": {
                "prompt_tokens": 500,
                "completion_tokens": 50,
                "total_tokens": 550
            }
        }"#;

        let event = create_test_event(data);
        let usage = parser.parse_event(&event).expect("should parse");
        assert_eq!(usage.input_tokens, 500);
        assert_eq!(usage.output_tokens, 50);
        assert_eq!(usage.cache_creation_input_tokens, None);
        assert_eq!(usage.cache_read_input_tokens, None);
    }

    #[test]
    fn test_skip_done_marker() {
        let parser = TokenParser::new();

        let event = create_test_event("[DONE]");
        assert!(parser.parse_event(&event).is_none());

        let event = create_test_event("[END]");
        assert!(parser.parse_event(&event).is_none());
    }

    #[test]
    fn test_parse_no_usage() {
        let parser = TokenParser::new();
        let data = r#"{"choices":[{"delta":{"content":"Hello"}}]}"#;

        let event = create_test_event(data);
        assert!(parser.parse_event(&event).is_none());
    }

    #[test]
    fn test_parse_data_directly() {
        let parser = TokenParser::new();
        let data = r#"{"usage":{"prompt_tokens":10,"completion_tokens":5},"model":"gpt-3.5"}"#;

        let usage = parser.parse_data(data);
        assert!(usage.is_some());

        let usage = usage.unwrap();
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn test_scan_partial_usage_dashscope_response_completed() {
        // Real dashscope `/compatible-mode/v1/responses` `response.completed`
        // payload, captured with curl. The strict JSON path should succeed
        // for this complete buffer, so this just confirms canonical fields.
        let data = r#"{"sequence_number":10,"type":"response.completed","response":{"top_logprobs":0,"instructions":"You are a helpful assistant.","metadata":{},"usage":{"total_tokens":60,"input_tokens_details":{"cached_tokens":0},"output_tokens":3,"input_tokens":57,"output_tokens_details":{"reasoning_tokens":0},"x_details":[{"total_tokens":60,"x_billing_type":"response_api","output_tokens":3,"input_tokens":57,"prompt_tokens_details":{"cached_tokens":0}}]},"created_at":1782287513,"model":"qwen3-coder-plus"}}"#;
        let parser = TokenParser::new();
        let usage = parser.parse_data(data).expect("usage should parse");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);
        // The nested counter must survive the strict JSON path too: the same
        // payload truncated recovers it through the scan fallback, so dropping
        // it here would make the complete buffer the less accurate of the two.
        assert_eq!(usage.cache_read_input_tokens, Some(0));
    }

    #[test]
    fn test_parse_responses_usage_with_nested_cached_tokens() {
        // The Responses API nests the cache-hit counter under
        // `input_tokens_details.cached_tokens`. Nothing read that key, so a
        // complete payload reported no cache reads at all even though the
        // counter was present.
        let data = r#"{"sequence_number":10,"type":"response.completed","response":{"usage":{"total_tokens":60,"input_tokens_details":{"cached_tokens":2},"output_tokens":3,"input_tokens":57},"model":"qwen3-coder-plus"}}"#;
        let parser = TokenParser::new();
        let usage = parser.parse_data(data).expect("usage should parse");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);
        assert_eq!(
            usage.cache_read_input_tokens,
            Some(2),
            "input_tokens_details.cached_tokens is the Responses-API cache counter"
        );
        // Guard the ordering the fallback already pinned for the truncated
        // variant: an OpenAI-style nested counter must not relabel the provider.
        assert_eq!(usage.provider, LLMProvider::OpenAI);
    }

    #[test]
    fn test_parse_responses_incomplete_usage() {
        // A capped Responses stream terminates with `response.incomplete`,
        // not `response.completed` — and the terminal event carries the
        // final usage either way. The strict path recognized `completed`
        // only, so everywhere TokenParser feeds usage (the drain
        // enrichment, the analyzer's SSE usage aggregation) a capped call
        // recorded no tokens at all, while the live message parser
        // recovered them (e534bec1b).
        let data = r#"{"sequence_number":7,"type":"response.incomplete","response":{"id":"resp_1","status":"incomplete","incomplete_details":{"reason":"max_output_tokens"},"usage":{"total_tokens":15,"input_tokens":10,"output_tokens":5},"model":"qwen3-coder-plus"}}"#;
        let parser = TokenParser::new();
        let usage = parser.parse_data(data).expect("usage should parse");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
        // OpenAI-style usage keys must keep the OpenAI provider label so
        // billed_input_tokens does not switch to the Anthropic formula.
        assert_eq!(usage.provider, LLMProvider::OpenAI);
        // The model rides inside `response.model`; the event has no top-level
        // `model`, so the token record kept an empty model for Responses calls.
        assert_eq!(usage.model.as_deref(), Some("qwen3-coder-plus"));
    }

    #[test]
    fn test_scan_partial_usage_truncated_buffer() {
        // Simulate a continuation buffer where only the leading bytes around
        // the `usage` block survived; trailing braces / brackets are missing
        // and strict JSON parsing fails. The fallback path should still
        // recover the integer values.
        let data = r#"event:response.completed
data:{"sequence_number":10,"type":"response.completed","response":{"usage":{"total_tokens":60,"input_tokens_details":{"cached_tokens":2},"output_tokens":3,"input_tokens":57"#;
        let parser = TokenParser::new();
        let usage = parser
            .parse_data(data)
            .expect("partial usage should still parse");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);
        assert_eq!(usage.cache_read_input_tokens, Some(2));
        // OpenAI-style cache (cached_tokens) stays inside input: provider
        // must not be mislabeled as Anthropic.
        assert_eq!(usage.provider, LLMProvider::OpenAI);
    }

    #[test]
    fn test_scan_partial_usage_anthropic_cache_fields_report_anthropic() {
        // Truncated Anthropic message_delta: Anthropic bills cache tokens on
        // top of input_tokens, so the scan must label the provider Anthropic —
        // otherwise billed_input_tokens silently drops the cache counters.
        let data = r#"{"type":"message_delta","usage":{"input_tokens":100,"cache_creation_input_tokens":10,"cache_read_input_tokens":20"#;
        let parser = TokenParser::new();
        let usage = parser
            .parse_data(data)
            .expect("partial usage should still parse");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.cache_creation_input_tokens, Some(10));
        assert_eq!(usage.cache_read_input_tokens, Some(20));
        assert_eq!(usage.provider, LLMProvider::Anthropic);
    }

    #[test]
    fn test_scan_partial_usage_skips_a_tool_schema_key() {
        // The Responses `response.completed` payload echoes the request's
        // tool definitions before `usage`. A tool parameter named like a
        // usage counter is an object schema, not a count; the scan used to
        // stop there and record input=0 for a truncated buffer.
        let data = r#"{"type":"response.completed","response":{"tools":[{"type":"function","function":{"name":"meter","parameters":{"type":"object","properties":{"input_tokens":{"type":"integer"},"output_tokens":{"type":"integer"}}}}}],"usage":{"input_tokens":57,"output_tokens":3"#;
        let parser = TokenParser::new();
        let usage = parser
            .parse_data(data)
            .expect("the real usage counters must still parse");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);
    }

    #[test]
    fn test_scan_partial_usage_returns_none_when_no_tokens() {
        let data = "event:response.output_text.delta\ndata:{\"delta\":\"hi\"}";
        let parser = TokenParser::new();
        assert!(parser.parse_data(data).is_none());
    }

    #[test]
    fn test_scan_partial_usage_only_input() {
        // Truncated JSON forces the regex-free string-scan fallback.
        let data = r#"{"input_tokens": 42"#;
        let parser = TokenParser::new();
        let usage = parser.parse_data(data).expect("should parse input only");
        assert_eq!(usage.input_tokens, 42);
        assert_eq!(usage.output_tokens, 0);
    }

    #[test]
    fn test_scan_partial_usage_only_output() {
        let data = r#"{"output_tokens": 7"#;
        let parser = TokenParser::new();
        let usage = parser.parse_data(data).expect("should parse output only");
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 7);
    }

    #[test]
    fn test_scan_partial_usage_prompt_completion_aliases() {
        let data = r#"{"prompt_tokens": 10, "completion_tokens": 5"#;
        let parser = TokenParser::new();
        let usage = parser.parse_data(data).expect("should parse aliases");
        assert_eq!(usage.input_tokens, 10);
        assert_eq!(usage.output_tokens, 5);
    }

    #[test]
    fn test_scan_partial_usage_invalid_value_none() {
        let data = r#"{"input_tokens": "not a number"}"#;
        let parser = TokenParser::new();
        assert!(parser.parse_data(data).is_none());
    }

    fn capture_truncated_responses_sse(
        truncated_terminal_event: &str,
    ) -> Vec<crate::analyzer::AnalysisResult> {
        use crate::aggregator::Aggregator;
        use crate::analyzer::Analyzer;
        use crate::parser::Parser;
        use crate::probes::sslsniff::SslEvent;
        use std::rc::Rc;

        let make_ssl = |buf: &[u8], rw: i32, ts: u64| {
            Rc::new(SslEvent {
                source: 0,
                timestamp_ns: ts,
                delta_ns: 0,
                pid: 4242,
                tid: 4242,
                uid: 0,
                len: buf.len() as u32,
                rw,
                comm: "agent-test".to_string(),
                buf: buf.to_vec(),
                is_handshake: false,
                ssl_ptr: 0x7000,
            })
        };

        let parser = Parser::new();
        let mut aggregator = Aggregator::new();

        let req_body = r#"{"model":"gpt-5","stream":true,"input":"hi"}"#;
        let req = format!(
            "POST /v1/responses HTTP/1.1\r\nHost: api.openai.com\r\nContent-Type: application/json\r\nContent-Length: {}\r\n\r\n{req_body}",
            req_body.len()
        );
        assert!(
            aggregator
                .process_result(parser.parse_ssl_event(make_ssl(req.as_bytes(), 1, 100)))
                .is_empty()
        );

        let resp_head = concat!(
            "HTTP/1.1 200 OK\r\n",
            "Content-Type: text/event-stream\r\n",
            "\r\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
        );
        assert!(
            aggregator
                .process_result(parser.parse_ssl_event(make_ssl(resp_head.as_bytes(), 0, 200)))
                .is_empty()
        );

        // Truncated terminal continuation chunk (no trailing `\n\n`, cut mid-JSON).
        assert!(
            aggregator
                .process_result(parser.parse_ssl_event(make_ssl(
                    truncated_terminal_event.as_bytes(),
                    0,
                    300
                )))
                .is_empty()
        );

        let drained = aggregator.drain_pending();
        assert_eq!(
            drained.len(),
            1,
            "drained capture should emit one truncated SSE exchange"
        );
        Analyzer::new().analyze_aggregated(&drained[0])
    }

    #[test]
    fn partial_usage_boundaries_ignore_schema_examples_in_captured_responses() {
        let truncated = concat!(
            "event: response.completed\n",
            "data: {",
            "\"sequence_number\":10,",
            "\"type\":\"response.completed\",",
            "\"response\":{",
            "\"id\":\"resp_1\",",
            "\"model\":\"gpt-5\",",
            "\"tools\":[{",
            "\"type\":\"function\",",
            "\"name\":\"meter\",",
            "\"parameters\":{",
            "\"type\":\"object\",",
            "\"default\":{\"input_tokens\":9000,\"output_tokens\":8000,\"cached_tokens\":7000},",
            "\"examples\":[{\"input_tokens\":9000,\"output_tokens\":8000,\"cached_tokens\":7000}],",
            "\"properties\":{",
            "\"input_tokens\":{\"type\":\"integer\",\"default\":9000},",
            "\"output_tokens\":{\"type\":\"integer\",\"default\":8000},",
            "\"cached_tokens\":{\"type\":\"integer\",\"default\":7000}",
            "}}",
            "}],",
            "\"usage\":{",
            "\"input_tokens\":57,",
            "\"output_tokens\":3,",
            "\"total_tokens\":60,",
            "\"input_tokens_details\":{\"cached_tokens\":2}",
            "},",
            "\"metadata\":{\"tail\":\"cut",
        );

        let results = capture_truncated_responses_sse(truncated);
        let token = results
            .iter()
            .find_map(|r| match r {
                crate::analyzer::AnalysisResult::Token(record) => Some(record),
                _ => None,
            })
            .expect("canonical response.usage must still produce a Token record");

        assert_eq!(token.input_tokens, 57);
        assert_eq!(token.output_tokens, 3);
        assert_eq!(token.cache_read_tokens, Some(2));
        assert_eq!(token.provider, "openai");
    }

    #[test]
    fn partial_usage_boundaries_do_not_invent_usage_before_the_usage_block_arrives() {
        let truncated_before_usage = concat!(
            "event: response.completed\n",
            "data: {",
            "\"sequence_number\":10,",
            "\"type\":\"response.completed\",",
            "\"response\":{",
            "\"id\":\"resp_2\",",
            "\"model\":\"gpt-5\",",
            "\"tools\":[{",
            "\"type\":\"function\",",
            "\"name\":\"meter\",",
            "\"parameters\":{",
            "\"type\":\"object\",",
            "\"default\":{\"usage\":{\"input_tokens\":9000,\"output_tokens\":8000,\"cached_tokens\":7000}},",
            "\"examples\":[{\"usage\":{\"input_tokens\":9000,\"output_tokens\":8000,\"cached_tokens\":7000}}]",
            "}}",
            "}],",
            "\"metadata\":{\"before_usage\":\"cut",
        );

        let results = capture_truncated_responses_sse(truncated_before_usage);
        let token = results.iter().find_map(|r| match r {
            crate::analyzer::AnalysisResult::Token(record) => Some(record),
            _ => None,
        });
        assert!(
            token.is_none(),
            "schema examples before real usage must not invent a Token record, got: {token:?}"
        );
    }

    #[test]
    fn test_scan_partial_usage_skips_numeric_tool_schema_defaults_and_examples() {
        let parser = TokenParser::new();
        let data = concat!(
            "{\"type\":\"response.completed\",\"response\":{",
            "\"tools\":[{\"type\":\"function\",\"function\":{\"name\":\"meter\",\"parameters\":{",
            "\"type\":\"object\",",
            "\"default\":{\"input_tokens\":9000,\"output_tokens\":8000,\"cached_tokens\":7000},",
            "\"examples\":[{\"usage\":{\"input_tokens\":9000,\"output_tokens\":8000,\"cached_tokens\":7000}}]",
            "}}}],",
            "\"usage\":{\"input_tokens\":57,\"output_tokens\":3,\"input_tokens_details\":{\"cached_tokens\":2}},",
            "\"metadata\":{\"tail\":\"cut",
        );
        let usage = parser
            .parse_data(data)
            .expect("canonical usage after numeric tool schema examples must parse");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);
        assert_eq!(usage.cache_read_input_tokens, Some(2));
        assert_eq!(usage.provider, LLMProvider::OpenAI);
    }

    #[test]
    fn test_scan_partial_usage_rejects_nested_usage_inside_schema_before_real_usage() {
        let parser = TokenParser::new();
        let data = concat!(
            "{\"type\":\"response.completed\",\"response\":{",
            "\"tools\":[{\"type\":\"function\",\"input_schema\":{",
            "\"type\":\"object\",",
            "\"properties\":{\"usage\":{\"type\":\"object\",\"default\":{\"input_tokens\":9000,\"output_tokens\":8000}}},",
            "\"$defs\":{\"UsageExample\":{\"default\":{\"usage\":{\"input_tokens\":9000,\"output_tokens\":8000}}}}",
            "}}],",
            "\"metadata\":{\"cut",
        );
        assert!(parser.parse_data(data).is_none());
    }

    #[test]
    fn test_scan_partial_usage_ignores_escaped_quotes_and_trailing_backslashes_in_strings() {
        let parser = TokenParser::new();
        // String literal contains fake usage with escaped quotes and terminates
        // immediately after an escaped backslash (`\\"`), followed by real usage.
        let with_real_usage = r#"{"type":"response.completed","response":{"instructions":"Example \"usage\":{\"input_tokens\":9000,\"output_tokens\":8000}\\","usage":{"input_tokens":57,"output_tokens":3"#;
        let usage = parser
            .parse_data(with_real_usage)
            .expect("escaped backslash before closing quote must not swallow real usage");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);

        // Truncated inside a string literal that embeds fake JSON usage.
        let truncated_in_string = r#"{"type":"response.output_text.delta","delta":"{\"usage\":{\"input_tokens\":9000,\"output_tokens\":8000}}"#;
        assert!(parser.parse_data(truncated_in_string).is_none());
    }

    #[test]
    fn test_scan_partial_usage_skips_non_protocol_containers_and_user_content() {
        let parser = TokenParser::new();
        for payload in [
            r#"{"type":"response.completed","response":{"output":[{"type":"function_call","arguments":{"usage":{"input_tokens":9000,"output_tokens":8000}}}],"tail":"cut"#,
            r#"{"type":"message_delta","content":{"usage":{"input_tokens":9000,"output_tokens":8000}},"tail":"cut"#,
            r#"{"type":"response.completed","response":{"metadata":{"usage":{"input_tokens":9000,"output_tokens":8000}}},"tail":"cut"#,
            r#"{"definitions":{"usage":{"input_tokens":9000,"output_tokens":8000}},"tail":"cut"#,
        ] {
            assert!(
                parser.parse_data(payload).is_none(),
                "non-protocol container must not yield usage: {payload}"
            );
        }
    }

    #[test]
    fn test_scan_partial_usage_accepts_standard_protocol_wrappers() {
        let parser = TokenParser::new();

        // Anthropic message_start truncated inside message.usage.
        let anthropic = r#"{"type":"message_start","message":{"id":"msg_1","usage":{"input_tokens":120,"output_tokens":1,"cache_creation_input_tokens":30,"cache_read_input_tokens":40"#;
        let usage = parser
            .parse_data(anthropic)
            .expect("message.usage must parse");
        assert_eq!(usage.provider, LLMProvider::Anthropic);
        assert_eq!(usage.input_tokens, 120);
        assert_eq!(usage.output_tokens, 1);
        assert_eq!(usage.cache_creation_input_tokens, Some(30));
        assert_eq!(usage.cache_read_input_tokens, Some(40));

        // Nested standard protocol wrappers (`data.response.usage`).
        let wrapped = r#"{"event":"response.completed","data":{"response":{"tools":[{"default":{"input_tokens":9000}}],"usage":{"input_tokens":12,"output_tokens":4"#;
        let usage = parser
            .parse_data(wrapped)
            .expect("data.response.usage must parse");
        assert_eq!(usage.input_tokens, 12);
        assert_eq!(usage.output_tokens, 4);
        assert_eq!(usage.provider, LLMProvider::OpenAI);

        // Gemini usageMetadata wrapped inside `result`.
        let gemini_wrapped = r#"{"result":{"candidates":[{"index":0}],"usageMetadata":{"promptTokenCount":20,"toolUsePromptTokenCount":5,"candidatesTokenCount":4,"thoughtsTokenCount":10,"cachedContentTokenCount":8"#;
        let usage = parser
            .parse_data(gemini_wrapped)
            .expect("result.usageMetadata must parse");
        assert_eq!(usage.provider, LLMProvider::Gemini);
        assert_eq!(usage.input_tokens, 25);
        assert_eq!(usage.output_tokens, 14);
        assert_eq!(usage.cache_read_input_tokens, Some(8));
    }

    #[test]
    fn test_scan_partial_usage_gemini_snake_case_tool_prompt_and_cache() {
        let parser = TokenParser::new();
        let data = r#"data:{"candidates":[{"index":0}],"usage_metadata":{"prompt_token_count":30,"tool_use_prompt_token_count":7,"candidates_token_count":2,"thoughts_token_count":18,"cached_content_token_count":11"#;
        let usage = parser
            .parse_data(data)
            .expect("snake_case usage_metadata must parse");
        assert_eq!(usage.provider, LLMProvider::Gemini);
        assert_eq!(usage.input_tokens, 37);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.cache_read_input_tokens, Some(11));
    }

    #[test]
    fn test_scan_partial_usage_dashscope_ignores_x_details_echo() {
        let parser = TokenParser::new();
        let data = r#"{"type":"response.completed","response":{"usage":{"x_details":[{"input_tokens":9999,"output_tokens":8888,"prompt_tokens_details":{"cached_tokens":7777}}],"input_tokens":57,"output_tokens":3,"input_tokens_details":{"cached_tokens":2}},"tail":"cut"#;
        let usage = parser
            .parse_data(data)
            .expect("canonical usage fields must ignore x_details echo");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);
        assert_eq!(usage.cache_read_input_tokens, Some(2));
        assert_eq!(usage.provider, LLMProvider::OpenAI);
    }

    #[test]
    fn test_scan_partial_usage_rejects_non_usage_roots_and_non_integer_values() {
        let parser = TokenParser::new();
        assert!(
            parser
                .parse_data(r#"{"name":"meter","input_tokens":9000"#)
                .is_none()
        );
        assert!(
            parser
                .parse_data(r#"{"input_tokens":9000,"description":"schema example""#)
                .is_none()
        );
        assert!(
            parser
                .parse_data(r#"{"usage":{"input_tokens":57.5,"output_tokens":-3"#)
                .is_none()
        );
        assert!(
            parser
                .parse_data(r#""input_tokens":9000,"output_tokens":8000},"properties":{"#)
                .is_none()
        );

        let mixed = parser
            .parse_data(r#"{"usage":{"input_tokens":57,"output_tokens":3.5"#)
            .expect("valid integer sibling should survive non-integer field");
        assert_eq!(mixed.input_tokens, 57);
        assert_eq!(mixed.output_tokens, 0);

        let braceless = parser
            .parse_data(r#""input_tokens": 42, "output_tokens": 7"#)
            .expect("brace-less top-level bare usage fragment should parse");
        assert_eq!(braceless.input_tokens, 42);
        assert_eq!(braceless.output_tokens, 7);
    }

    #[test]
    fn test_scan_partial_usage_multi_frame_sse_skips_null_usage_and_recovers_later_frame() {
        let parser = TokenParser::new();
        let data = concat!(
            "event: response.created\n",
            "data: {\"type\":\"response.created\",\"response\":{\"tools\":[{\"parameters\":{\"default\":{\"input_tokens\":9000,\"output_tokens\":8000}}}],\"usage\":null}}\n",
            "\n",
            "event: response.output_text.delta\n",
            "data: {\"type\":\"response.output_text.delta\",\"delta\":\"{\\\"usage\\\":{\\\"input_tokens\\\":9999}}\"}\n",
            "\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\"tools\":[{\"parameters\":{\"default\":{\"usage\":{\"input_tokens\":9000}}}}],\"usage\":{\"input_tokens\":57,\"output_tokens\":3,\"input_tokens_details\":{\"cached_tokens\":2}},\"metadata\":{\"tail\":\"cut",
        );
        let usage = parser
            .parse_data(data)
            .expect("later truncated frame's canonical usage must be recovered");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);
        assert_eq!(usage.cache_read_input_tokens, Some(2));
        assert_eq!(usage.provider, LLMProvider::OpenAI);
    }

    #[test]
    fn test_scan_partial_usage_leading_json_before_sse_frames_and_multiline_data() {
        let parser = TokenParser::new();

        // A TLS split right after `data: ` of an earlier delta leaves leading
        // JSON before later SSE `data:` frames in the same continuation buffer.
        let leading_json_then_sse = concat!(
            "{\"type\":\"response.output_text.delta\",\"delta\":\"ok\"}\n\n",
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{",
            "\"tools\":[{\"default\":{\"input_tokens\":9000}}],",
            "\"usage\":{\"input_tokens\":57,\"output_tokens\":3",
        );
        let usage = parser
            .parse_data(leading_json_then_sse)
            .expect("terminal SSE frame after leading JSON must be recovered");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);
        assert_eq!(usage.provider, LLMProvider::OpenAI);

        // Earlier zero-placeholder snapshot merged with later complete
        // Anthropic frame and trailing `[DONE]` marker.
        let zero_then_anthropic = concat!(
            "{\"type\":\"message_start\",\"message\":{\"usage\":{\"input_tokens\":0,\"output_tokens\":0}}}\n\n",
            "data: {\"type\":\"message_delta\",\"usage\":{\"input_tokens\":100,\"output_tokens\":42,\"cache_read_input_tokens\":20}}\n\n",
            "data: [DONE]\n\n",
        );
        let usage = parser
            .parse_data(zero_then_anthropic)
            .expect("zero placeholder must merge with later Anthropic usage");
        assert_eq!(usage.input_tokens, 100);
        assert_eq!(usage.output_tokens, 42);
        assert_eq!(usage.cache_read_input_tokens, Some(20));
        assert_eq!(usage.provider, LLMProvider::Anthropic);

        // Unterminated multiline SSE `data:` event.
        let multiline_sse = concat!(
            "event: response.completed\n",
            "data: {\"type\":\"response.completed\",\"response\":{\n",
            "data: \"usage\":{\"input_tokens\":57,\"output_tokens\":3",
        );
        let usage = parser
            .parse_data(multiline_sse)
            .expect("multiline unterminated SSE data must parse");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);

        // Consecutive raw JSON objects without SSE framing.
        let ndjson = concat!(
            "{\"type\":\"response.output_text.delta\",\"delta\":\"hi\"}\n",
            "{\"usage\":{\"input_tokens\":19,\"output_tokens\":6",
        );
        let usage = parser
            .parse_data(ndjson)
            .expect("second consecutive raw JSON object must parse");
        assert_eq!(usage.input_tokens, 19);
        assert_eq!(usage.output_tokens, 6);
    }

    #[test]
    fn test_scan_partial_usage_dashscope_multimodal_and_details_edge_cases() {
        let parser = TokenParser::new();

        // DashScope multimodal usage carries modality counters and top-level
        // `cached_tokens`; input_tokens already includes modality counts.
        let multimodal = r#"{"output":{"choices":[]},"usage":{"input_tokens":1261,"output_tokens":382,"image_tokens":1249,"cached_tokens":12"#;
        let usage = parser
            .parse_data(multimodal)
            .expect("truncated DashScope multimodal usage must parse");
        assert_eq!(usage.provider, LLMProvider::DashScope);
        assert_eq!(usage.input_tokens, 1261);
        assert_eq!(usage.output_tokens, 382);
        assert_eq!(usage.cache_read_input_tokens, Some(12));

        // Non-object details fields and `output_tokens_details` keep OpenAI
        // billing without aborting sibling counters.
        let non_object_details = r#"{"usage":{"input_tokens_details":null,"prompt_tokens_details":"n/a","output_tokens_details":{"reasoning_tokens":5},"input_tokens":21,"output_tokens":8"#;
        let usage = parser
            .parse_data(non_object_details)
            .expect("non-object details values must be skipped cleanly");
        assert_eq!(usage.provider, LLMProvider::OpenAI);
        assert_eq!(usage.input_tokens, 21);
        assert_eq!(usage.output_tokens, 8);

        // Unknown sibling keys and nested objects inside `prompt_tokens_details`.
        let nested_details = r#"{"usage":{"prompt_tokens":300,"completion_tokens":20,"prompt_tokens_details":{"text_tokens":300,"cache_creation":{"ephemeral_5m_input_tokens":50},"cached_tokens":9"#;
        let usage = parser
            .parse_data(nested_details)
            .expect("unknown keys inside prompt_tokens_details must be skipped");
        assert_eq!(usage.provider, LLMProvider::OpenAI);
        assert_eq!(usage.input_tokens, 300);
        assert_eq!(usage.output_tokens, 20);
        assert_eq!(usage.cache_read_input_tokens, Some(9));
    }

    #[test]
    fn test_scan_partial_usage_escaped_keys_malformed_and_braceless_edges() {
        let parser = TokenParser::new();

        // Unicode-escaped `"usage"` key and quote-escaped sibling key.
        let escaped_keys = r#"{"k\"ey":"val","us\u0061ge":{"input_tokens":57,"output_tokens":3"#;
        let usage = parser
            .parse_data(escaped_keys)
            .expect("escaped JSON object keys must decode cleanly");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 3);

        // Truncated escape sequence inside a trailing key still keeps earlier
        // counters.
        let truncated_key_escape = r#"{"usage":{"input_tokens":57,"out\"#;
        let usage = parser
            .parse_data(truncated_key_escape)
            .expect("truncated trailing key escape must preserve earlier counter");
        assert_eq!(usage.input_tokens, 57);
        assert_eq!(usage.output_tokens, 0);

        // RFC 8259 leading zero is skipped without blocking valid siblings.
        let leading_zero = r#"{"usage":{"input_tokens":0123,"output_tokens":4"#;
        let usage = parser
            .parse_data(leading_zero)
            .expect("valid sibling must parse after leading-zero number");
        assert_eq!(usage.input_tokens, 0);
        assert_eq!(usage.output_tokens, 4);

        // Malformed / truncated skipped subtrees and invalid braceless inputs
        // must fail closed with `None`. Each payload includes `"input_tokens"`
        // so it enters `scan_partial_usage` past `parse_data`'s fast gate.
        for rejected in [
            r#"{"tools":[{"a":1}},"usage":{"input_tokens":57"#,
            r#"{"tools":"cut\","usage":{"input_tokens":57"#,
            "{\"tools\":\"\\\"input_tokens\\\":9000\\",
            r#"{"tools":[{"input_tokens":9000"#,
            r#""foo": 1, "input_tokens": 9000"#,
            r#""input_tokens" 9000"#,
            r#"prefix "input_tokens": 9000"#,
            "data:   \n\ndata: [DONE]\n\n",
        ] {
            assert!(
                parser.parse_data(rejected).is_none(),
                "malformed or non-usage fragment must return None: {rejected}"
            );
            assert!(
                TokenParser::scan_partial_usage(rejected).is_none(),
                "scan_partial_usage must return None: {rejected}"
            );
        }
    }
}
