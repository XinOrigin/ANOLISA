//! Seccomp-bpf syscall filtering for the ws-ckpt daemon.
//!
//! Applies a **deny-list** filter: all syscalls are allowed by default,
//! and a curated list of dangerous / irrelevant syscalls is blocked with EPERM.
//!
//! This approach is chosen because the daemon shells out to external commands
//! (btrfs, mount, rsync, systemctl) whose full syscall set is impractical to
//! enumerate in an allow-list.

use std::collections::BTreeMap;

use seccompiler::{
    apply_filter_all_threads, BpfProgram, SeccompAction, SeccompCmpArgLen, SeccompCmpOp,
    SeccompCondition, SeccompFilter, SeccompRule, TargetArch,
};

/// Match the build target_arch to seccompiler's TargetArch so the BPF
/// audit_arch check passes at runtime. Returns Err on unsupported arches
/// (e.g. loongarch64) so the daemon refuses to install a no-op filter.
fn target_arch() -> anyhow::Result<TargetArch> {
    #[cfg(target_arch = "x86_64")]
    return Ok(TargetArch::x86_64);
    #[cfg(target_arch = "aarch64")]
    return Ok(TargetArch::aarch64);
    #[cfg(target_arch = "riscv64")]
    return Ok(TargetArch::riscv64);
    #[cfg(not(any(
        target_arch = "x86_64",
        target_arch = "aarch64",
        target_arch = "riscv64"
    )))]
    anyhow::bail!(
        "seccomp filter not supported on target arch '{}'; refusing to install no-op filter",
        std::env::consts::ARCH
    );
}

/// Blocked syscalls — operations the daemon should never perform.
///
/// Excluded on purpose: new mount API (fsopen/fsmount/fsconfig/move_mount/
/// open_tree/mount_setattr), io_uring, and landlock. mount(8) and various
/// libraries probe these and only fall back on ENOSYS; returning EPERM would
/// break the daemon's own mount path or future tokio io_uring opt-in.
///
/// Categories:
/// - Kernel/module manipulation
/// - System state changes (reboot, hostname, clock)
/// - Debugging / tracing
/// - Swap management
/// - Accounting / profiling
/// - Key management
/// - Virtualization / namespaces
/// - pidfd / cross-process memory (daemon never inspects other processes)
fn blocked_syscalls() -> Vec<i64> {
    vec![
        // ── Kernel module loading ──
        libc::SYS_init_module,
        libc::SYS_finit_module,
        libc::SYS_delete_module,
        // ── System state ──
        libc::SYS_reboot,
        libc::SYS_kexec_load,
        libc::SYS_kexec_file_load,
        libc::SYS_sethostname,
        libc::SYS_setdomainname,
        libc::SYS_settimeofday,
        libc::SYS_adjtimex,
        libc::SYS_clock_adjtime,
        libc::SYS_pivot_root,
        // ── Debugging / tracing ──
        libc::SYS_ptrace,
        libc::SYS_process_vm_readv,
        libc::SYS_process_vm_writev,
        libc::SYS_kcmp,
        // ── Swap management ──
        libc::SYS_swapon,
        libc::SYS_swapoff,
        // ── Accounting / profiling ──
        libc::SYS_acct,
        libc::SYS_perf_event_open,
        // ── Key management ──
        libc::SYS_add_key,
        libc::SYS_request_key,
        libc::SYS_keyctl,
        // ── Virtualization / namespaces (daemon must not create new namespaces) ──
        libc::SYS_unshare,
        libc::SYS_setns,
        // ── Misc dangerous ──
        libc::SYS_bpf, // prevent loading arbitrary BPF programs
        libc::SYS_userfaultfd,
        libc::SYS_move_pages,
        // ── pidfd / cross-process memory (5.x) ──
        libc::SYS_pidfd_open,
        libc::SYS_pidfd_send_signal,
        libc::SYS_pidfd_getfd,
        libc::SYS_process_madvise,
    ]
}

/// Create a rule that matches unconditionally (arg0 as u64 >= 0 is always true).
fn unconditional_rule() -> Result<SeccompRule, seccompiler::Error> {
    Ok(SeccompRule::new(vec![SeccompCondition::new(
        0,
        SeccompCmpArgLen::Qword,
        SeccompCmpOp::Ge,
        0,
    )?])?)
}

/// Apply the seccomp deny-list filter to all threads in the process (and future children).
///
/// Uses `SECCOMP_FILTER_FLAG_TSYNC` so pre-existing Tokio worker and blocking
/// threads receive the filter as well. Must be called **after** bootstrap
/// (image creation, loop mount) completes, but **before** the listener loop starts.
pub fn apply_seccomp_filter() -> anyhow::Result<()> {
    let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();

    for syscall in blocked_syscalls() {
        rules.insert(syscall, vec![unconditional_rule()?]);
    }

    let filter = SeccompFilter::new(
        rules,
        SeccompAction::Allow,                     // default: allow everything
        SeccompAction::Errno(libc::EPERM as u32), // blocked: return EPERM
        target_arch()?,
    )?;

    let bpf: BpfProgram = filter.try_into()?;
    apply_filter_all_threads(&bpf)?;

    tracing::info!(
        "Seccomp filter applied to all threads: {} dangerous syscalls blocked",
        blocked_syscalls().len()
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn blocked_syscalls_not_empty() {
        assert!(!blocked_syscalls().is_empty());
    }

    #[test]
    fn blocked_syscalls_no_duplicates() {
        let list = blocked_syscalls();
        let mut sorted = list.clone();
        sorted.sort();
        sorted.dedup();
        assert_eq!(list.len(), sorted.len(), "duplicate syscall numbers found");
    }

    #[test]
    fn filter_builds_for_target_arch() {
        let arch = target_arch().expect("test host arch should be supported");
        let mut rules: BTreeMap<i64, Vec<SeccompRule>> = BTreeMap::new();
        for syscall in blocked_syscalls() {
            rules.insert(syscall, vec![unconditional_rule().unwrap()]);
        }
        let filter = SeccompFilter::new(
            rules,
            SeccompAction::Allow,
            SeccompAction::Errno(libc::EPERM as u32),
            arch,
        );
        assert!(filter.is_ok(), "seccomp filter should build without errors");
    }

    const TSYNC_TEST_HELPER_ENV: &str = "WS_CKPT_TEST_SECCOMP_TSYNC_HELPER";
    const TSYNC_TEST_NAME: &str = "seccomp::tests::apply_seccomp_filter_applies_to_all_threads";

    fn current_tid() -> libc::pid_t {
        // SAFETY: `gettid(2)` takes no arguments and always succeeds on Linux.
        unsafe { libc::syscall(libc::SYS_gettid) as libc::pid_t }
    }

    fn read_thread_status(tid: libc::pid_t) -> String {
        let path = format!("/proc/self/task/{tid}/status");
        std::fs::read_to_string(&path).unwrap_or_else(|e| panic!("failed to read {path}: {e}"))
    }

    fn parse_status_field<'a>(status: &'a str, key: &str) -> Option<&'a str> {
        status
            .lines()
            .find_map(|line| line.strip_prefix(key).map(str::trim))
    }

    #[test]
    fn apply_seccomp_filter_applies_to_all_threads() {
        if std::env::var_os(TSYNC_TEST_HELPER_ENV).is_none() {
            let exe = std::env::current_exe().expect("resolve current test executable");
            let output = std::process::Command::new(exe)
                .arg("--exact")
                .arg(TSYNC_TEST_NAME)
                .arg("--test-threads=1")
                .arg("--nocapture")
                .env(TSYNC_TEST_HELPER_ENV, "1")
                .output()
                .expect("spawn seccomp subprocess helper");
            let stdout = String::from_utf8_lossy(&output.stdout);
            let stderr = String::from_utf8_lossy(&output.stderr);
            assert!(
                output.status.success(),
                "seccomp subprocess helper failed (status: {:?})\nstdout:\n{stdout}\nstderr:\n{stderr}",
                output.status
            );
            assert!(
                stdout.contains("1 passed"),
                "seccomp subprocess helper did not execute target test\nstdout:\n{stdout}\nstderr:\n{stderr}"
            );
            return;
        }

        let ready = std::sync::Arc::new(std::sync::Barrier::new(2));
        let applied = std::sync::Arc::new(std::sync::Barrier::new(2));
        let ready_sibling = std::sync::Arc::clone(&ready);
        let applied_sibling = std::sync::Arc::clone(&applied);

        let sibling = std::thread::spawn(move || {
            let tid = current_tid();
            let before = read_thread_status(tid);
            let before_filters = parse_status_field(&before, "Seccomp_filters:\t")
                .and_then(|v| v.parse::<u64>().ok());

            // Ensure the sibling thread is live before apply_seccomp_filter()
            // runs, then inspect /proc status after installation completes.
            ready_sibling.wait();
            applied_sibling.wait();

            let after = read_thread_status(tid);
            (tid, before_filters, after)
        });

        let caller_tid = current_tid();
        let caller_before = read_thread_status(caller_tid);
        let caller_before_filters = parse_status_field(&caller_before, "Seccomp_filters:\t")
            .and_then(|v| v.parse::<u64>().ok());

        ready.wait();
        let apply_res = apply_seccomp_filter();
        applied.wait();
        apply_res.expect("apply_seccomp_filter should succeed");

        let caller_after = read_thread_status(caller_tid);
        let (sibling_tid, sibling_before_filters, sibling_after) =
            sibling.join().expect("sibling thread panicked");

        assert_ne!(
            caller_tid, sibling_tid,
            "sibling must be a distinct OS thread"
        );

        let caller_seccomp = parse_status_field(&caller_after, "Seccomp:\t");
        let sibling_seccomp = parse_status_field(&sibling_after, "Seccomp:\t");
        assert_eq!(
            caller_seccomp,
            Some("2"),
            "calling thread (tid {caller_tid}) should have Seccomp:\\t2, got {caller_seccomp:?}"
        );
        assert_eq!(
            sibling_seccomp,
            Some("2"),
            "pre-existing sibling thread (tid {sibling_tid}) should have Seccomp:\\t2, got {sibling_seccomp:?}"
        );

        let caller_nnp = parse_status_field(&caller_after, "NoNewPrivs:\t");
        let sibling_nnp = parse_status_field(&sibling_after, "NoNewPrivs:\t");
        assert_eq!(
            caller_nnp,
            Some("1"),
            "calling thread (tid {caller_tid}) should have NoNewPrivs:\\t1, got {caller_nnp:?}"
        );
        assert_eq!(
            sibling_nnp,
            Some("1"),
            "pre-existing sibling thread (tid {sibling_tid}) should have NoNewPrivs:\\t1, got {sibling_nnp:?}"
        );

        if let Some(before) = caller_before_filters {
            let after = parse_status_field(&caller_after, "Seccomp_filters:\t")
                .and_then(|v| v.parse::<u64>().ok());
            assert_eq!(
                after,
                Some(before + 1),
                "calling thread (tid {caller_tid}) should gain one seccomp filter"
            );
        }
        if let Some(before) = sibling_before_filters {
            let after = parse_status_field(&sibling_after, "Seccomp_filters:\t")
                .and_then(|v| v.parse::<u64>().ok());
            assert_eq!(
                after,
                Some(before + 1),
                "pre-existing sibling thread (tid {sibling_tid}) should gain one seccomp filter"
            );
        }
    }
}
