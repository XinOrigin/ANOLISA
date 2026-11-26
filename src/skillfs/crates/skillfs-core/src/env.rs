//! Environment detection for SkillFS compiler.
//!
//! Detects OS, available commands, and selected environment variables to enable
//! environment-aware skill compilation.

use std::collections::{HashMap, HashSet};

// ---------------------------------------------------------------------------
// OsKind
// ---------------------------------------------------------------------------

/// The operating system kind, used in `@if os == ...` expressions.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum OsKind {
    Darwin,
    Linux,
    Windows,
    Unknown(String),
}

impl OsKind {
    /// Detect the current operating system.
    pub fn current() -> Self {
        match std::env::consts::OS {
            "macos" => OsKind::Darwin,
            "linux" => OsKind::Linux,
            "windows" => OsKind::Windows,
            other => OsKind::Unknown(other.to_string()),
        }
    }

    /// Return the canonical string representation used in skill conditions.
    pub fn as_str(&self) -> &str {
        match self {
            OsKind::Darwin => "darwin",
            OsKind::Linux => "linux",
            OsKind::Windows => "windows",
            OsKind::Unknown(s) => s.as_str(),
        }
    }
}

// ---------------------------------------------------------------------------
// EnvironmentProfile
// ---------------------------------------------------------------------------

/// Commands to probe for availability.
const COMMAND_WHITELIST: &[&str] = &[
    "uv",
    "pip",
    "pip3",
    "python",
    "python3",
    "npm",
    "yarn",
    "pnpm",
    "node",
    "docker",
    "docker-compose",
    "kubectl",
    "cargo",
    "rustc",
    "go",
    "git",
    "sh",
    "bash",
    "zsh",
    "fish",
    "jq",
    "curl",
    "wget",
];

/// Environment variables to capture (selective whitelist).
const ENV_VAR_WHITELIST: &[&str] = &[
    "HOME",
    "PATH",
    "SHELL",
    "VIRTUAL_ENV",
    "PIP_USER",
    "CONDA_DEFAULT_ENV",
    "NODE_ENV",
    "GOPATH",
    "CARGO_HOME",
    "UV_CACHE_DIR",
    "PYENV_ROOT",
];

/// Complete runtime environment snapshot used for skill compilation and filtering.
#[derive(Debug, Clone)]
pub struct EnvironmentProfile {
    pub os: OsKind,
    /// Commands available on PATH (from `COMMAND_WHITELIST`).
    pub available_commands: HashSet<String>,
    /// Selected environment variables (from `ENV_VAR_WHITELIST`).
    pub env_vars: HashMap<String, String>,
}

impl EnvironmentProfile {
    pub fn detect() -> Self {
        let os = OsKind::current();
        let available_commands = probe_commands();
        let env_vars = capture_env_vars();
        Self {
            os,
            available_commands,
            env_vars,
        }
    }

    /// Returns `true` if `cmd` was found on PATH.
    pub fn has_command(&self, cmd: &str) -> bool {
        self.available_commands.contains(cmd)
    }

    /// Returns `true` if environment variable `var` is set.
    pub fn has_env(&self, var: &str) -> bool {
        self.env_vars.contains_key(var)
    }
}

// ---------------------------------------------------------------------------
// Probe helpers
// ---------------------------------------------------------------------------

fn probe_commands() -> HashSet<String> {
    let mut found = HashSet::new();
    for &cmd in COMMAND_WHITELIST {
        if command_exists(cmd) {
            found.insert(cmd.to_string());
        }
    }
    found
}

fn command_exists(cmd: &str) -> bool {
    #[cfg(unix)]
    {
        use std::ffi::CString;
        use std::os::unix::ffi::OsStrExt;

        let Some(path) = std::env::var_os("PATH").or_else(default_search_path) else {
            return false;
        };
        // Probe the executable itself: a restricted PATH need not contain
        // `which`. Do not run any of the commands while detecting them.
        std::env::split_paths(&path).any(|directory| {
            let candidate = directory.join(cmd);
            if !candidate.is_file() {
                return false;
            }
            let Ok(candidate) = CString::new(candidate.as_os_str().as_bytes()) else {
                return false;
            };
            // SAFETY: candidate owns a NUL-terminated path for the call.
            // AT_EACCESS checks the effective identity used to execute it.
            unsafe {
                libc::faccessat(
                    libc::AT_FDCWD,
                    candidate.as_ptr(),
                    libc::X_OK,
                    libc::AT_EACCESS,
                ) == 0
            }
        })
    }
    #[cfg(windows)]
    {
        std::process::Command::new("where")
            .arg(cmd)
            .stdout(std::process::Stdio::null())
            .stderr(std::process::Stdio::null())
            .status()
            .map(|s| s.success())
            .unwrap_or(false)
    }
    #[cfg(not(any(unix, windows)))]
    {
        false
    }
}

// Match the platform's default executable search path when PATH is absent.
#[cfg(unix)]
fn default_search_path() -> Option<std::ffi::OsString> {
    use std::os::unix::ffi::OsStringExt;

    // SAFETY: a null buffer with zero length queries the required capacity.
    let size = unsafe { libc::confstr(libc::_CS_PATH, std::ptr::null_mut(), 0) };
    if size == 0 {
        return None;
    }
    let mut path = vec![0u8; size];
    // SAFETY: path has size writable bytes; confstr writes a terminating NUL.
    let written = unsafe { libc::confstr(libc::_CS_PATH, path.as_mut_ptr().cast(), size) };
    if written == 0 || written > size {
        return None;
    }
    path.truncate(written - 1);
    Some(std::ffi::OsString::from_vec(path))
}

fn capture_env_vars() -> HashMap<String, String> {
    ENV_VAR_WHITELIST
        .iter()
        .filter_map(|&key| {
            let val = std::env::var_os(key)?;
            let encoded = match val.into_string() {
                Ok(utf8) => utf8,
                Err(raw) => encode_raw_os_str(&raw),
            };
            Some((key.to_string(), encoded))
        })
        .collect()
}

// Keep valid UTF-8 values verbatim while prefixing escaped raw OS payloads
// with `\0raw:` (impossible in NUL-terminated OS env vars) so raw bytes never
// collide with literal UTF-8 escapes in `TransformPipeline::identity`.
fn encode_raw_os_str(raw: &std::ffi::OsStr) -> String {
    #[cfg(unix)]
    {
        use std::fmt::Write;
        use std::os::unix::ffi::OsStrExt;

        let mut out = String::from("\0raw:");
        for &byte in raw.as_bytes() {
            match byte {
                b'\\' => out.push_str("\\\\"),
                0x20..=0x7e => out.push(char::from(byte)),
                _ => {
                    let _ = write!(out, "\\x{byte:02x}");
                }
            }
        }
        out
    }
    #[cfg(windows)]
    {
        use std::fmt::Write;
        use std::os::windows::ffi::OsStrExt;

        let mut out = String::from("\0raw:");
        for unit in raw.encode_wide() {
            match unit {
                0x005c => out.push_str("\\\\"),
                0x0020..=0x007e => out.push(char::from(unit as u8)),
                _ => {
                    let _ = write!(out, "\\u{unit:04x}");
                }
            }
        }
        out
    }
    #[cfg(not(any(unix, windows)))]
    {
        format!("\0raw:{}", raw.to_string_lossy())
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_os_kind_current_is_valid() {
        let os = OsKind::current();
        let s = os.as_str();
        assert!(!s.is_empty());
    }

    #[test]
    fn test_os_kind_as_str() {
        assert_eq!(OsKind::Darwin.as_str(), "darwin");
        assert_eq!(OsKind::Linux.as_str(), "linux");
        assert_eq!(OsKind::Windows.as_str(), "windows");
        assert_eq!(OsKind::Unknown("freebsd".to_string()).as_str(), "freebsd");
    }

    #[test]
    fn test_capture_pip_user() {
        assert!(ENV_VAR_WHITELIST.contains(&"PIP_USER"));
        assert_eq!(
            capture_env_vars().get("PIP_USER"),
            std::env::var("PIP_USER").ok().as_ref()
        );
    }

    #[test]
    fn test_has_command_nonexistent() {
        let profile = EnvironmentProfile::detect();
        assert!(!profile.has_command("this-command-does-not-exist-99999"));
    }

    #[test]
    #[cfg(unix)]
    fn test_has_command_sh_exists() {
        let profile = EnvironmentProfile::detect();
        assert!(
            profile.has_command("sh"),
            "sh should always be available on Unix"
        );
    }

    #[cfg(unix)]
    fn run_raw_env_child(bin_dir: &std::path::Path, case: &str, vars: &[(&str, &[u8])]) -> String {
        use std::ffi::OsStr;
        use std::os::unix::{ffi::OsStrExt, fs::PermissionsExt};

        let uv = bin_dir.join("uv");
        if !uv.exists() {
            std::fs::write(&uv, "#!/bin/sh\nexit 0\n").unwrap();
            std::fs::set_permissions(&uv, std::fs::Permissions::from_mode(0o755)).unwrap();
        }
        let mut path_bytes = bin_dir.as_os_str().as_bytes().to_vec();
        path_bytes.extend_from_slice(b":/missing-\xff");
        let mut cmd = std::process::Command::new(std::env::current_exe().unwrap());
        cmd.env_clear()
            .env("SKILLFS_RAW_ENV_CASE", case)
            .env("PATH", OsStr::from_bytes(&path_bytes));
        for (k, v) in vars {
            cmd.env(k, OsStr::from_bytes(v));
        }
        let out = cmd.args(["--exact", case, "--nocapture"]).output().unwrap();
        let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
        assert!(
            out.status.success(),
            "{stdout}\n{}",
            String::from_utf8_lossy(&out.stderr)
        );
        stdout
    }

    #[test]
    #[cfg(unix)]
    fn test_detect_captures_raw_byte_whitelisted_env_vars() {
        const CASE: &str = "env::tests::test_detect_captures_raw_byte_whitelisted_env_vars";
        if std::env::var("SKILLFS_RAW_ENV_CASE").as_deref() != Ok(CASE) {
            let tmp = tempfile::tempdir().unwrap();
            let extra: &[(&str, &[u8])] = &[
                ("VIRTUAL_ENV", b"/tmp/venv-\xff"),
                ("HOME", b"/home/user-\x80"),
                ("UV_CACHE_DIR", b"/tmp/uv-cache-\xfe"),
                ("UNLISTED_RAW_ENV", b"ignored-\xff"),
            ];
            assert!(run_raw_env_child(tmp.path(), CASE, extra).contains("raw-env-captured"));
            return;
        }
        let profile = EnvironmentProfile::detect();
        assert!(profile.has_command("uv"));
        for key in ["PATH", "VIRTUAL_ENV", "HOME", "UV_CACHE_DIR"] {
            assert!(profile.has_env(key), "{key} should be captured");
        }
        assert!(!profile.has_env("UNLISTED_RAW_ENV"));
        let pipeline = crate::transform::TransformPipeline::directive_only(profile);
        let doc = "<!-- @if has_env(\"VIRTUAL_ENV\") -->\nactive\n<!-- @else -->\ninactive\n<!-- @endif -->\n";
        assert_eq!(pipeline.run(doc), "active\n");
        println!("raw-env-captured");
    }

    #[test]
    #[cfg(unix)]
    fn test_detect_raw_byte_pip_user_preserves_pip_install() {
        const CASE: &str = "env::tests::test_detect_raw_byte_pip_user_preserves_pip_install";
        if std::env::var("SKILLFS_RAW_ENV_CASE").as_deref() != Ok(CASE) {
            let tmp = tempfile::tempdir().unwrap();
            let out = run_raw_env_child(tmp.path(), CASE, &[("PIP_USER", b"\xff")]);
            assert!(out.contains("raw-pip-user-preserved"));
            return;
        }
        let profile = EnvironmentProfile::detect();
        assert!(profile.has_command("uv") && profile.has_env("PIP_USER"));
        let pipeline = crate::transform::TransformPipeline::directive_only(profile);
        assert_eq!(
            pipeline.run("pip install requests\n"),
            "pip install requests\n"
        );
        println!("raw-pip-user-preserved");
    }

    #[test]
    #[cfg(unix)]
    fn test_detect_distinguishes_raw_byte_env_identities() {
        const CASE: &str = "env::tests::test_detect_distinguishes_raw_byte_env_identities";
        if std::env::var("SKILLFS_RAW_ENV_CASE").as_deref() != Ok(CASE) {
            let tmp = tempfile::tempdir().unwrap();
            let variants: [&[(&str, &[u8])]; 5] = [
                &[],
                &[("VIRTUAL_ENV", b"/venv-\x80")],
                &[("VIRTUAL_ENV", b"/venv-\xff")],
                &[("VIRTUAL_ENV", b"/venv-\\xff")],
                &[("VIRTUAL_ENV", b"/venv-\\x80\xff")],
            ];
            let mut digests = HashSet::new();
            for variant in variants {
                let stdout = run_raw_env_child(tmp.path(), CASE, variant);
                let hex = stdout.lines().find_map(|l| l.strip_prefix("identity="));
                assert!(digests.insert(hex.unwrap().to_string()));
            }
            return;
        }
        let id = crate::transform::TransformPipeline::directive_only(EnvironmentProfile::detect())
            .identity();
        let hex: String = id.iter().map(|b| format!("{b:02x}")).collect();
        println!("identity={hex}");
    }
}
