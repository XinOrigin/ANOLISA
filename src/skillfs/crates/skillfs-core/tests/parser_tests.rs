//! Integration tests for the parser module using fixture files.

use skillfs_core::parser;

mod common;
use common::load_fixture;

#[test]
fn test_parse_valid_full_fixture() {
    let content = load_fixture("valid_full.md");
    let entry = parser::parse_skill_md(&content, "web-search");

    assert_eq!(entry.metadata.name, "web-search");
    assert_eq!(
        entry.metadata.description,
        "Search the web for current information on any topic"
    );
    assert_eq!(entry.metadata.version, "1.2.0");
    assert_eq!(entry.metadata.tags, vec!["search", "web", "information"]);
    assert!(entry.metadata.enabled);
    assert_eq!(entry.parameters.len(), 3);
    assert_eq!(entry.returns.len(), 2);
    assert!(entry.parse_status.is_ok());
}

#[test]
fn test_parse_valid_minimal_fixture() {
    let content = load_fixture("valid_minimal.md");
    let entry = parser::parse_skill_md(&content, "hello-world");

    assert_eq!(entry.metadata.name, "hello-world");
    assert_eq!(
        entry.metadata.description,
        "A minimal skill that greets the user"
    );
    assert_eq!(entry.metadata.version, "0.0.0"); // default
    assert!(entry.metadata.tags.is_empty());
    assert!(entry.metadata.enabled);
    assert!(entry.parameters.is_empty());
    assert!(entry.returns.is_empty());
    assert!(entry.parse_status.is_ok());
}

#[test]
fn test_parse_valid_no_params_fixture() {
    let content = load_fixture("valid_no_params.md");
    let entry = parser::parse_skill_md(&content, "no-params-skill");

    assert!(entry.parameters.is_empty());
    assert!(entry.parse_status.is_ok());
}

#[test]
fn test_parse_missing_frontmatter_fixture() {
    let content = load_fixture("missing_frontmatter.md");
    let entry = parser::parse_skill_md(&content, "fallback-name");

    assert_eq!(entry.metadata.name, "fallback-name"); // from dir_name
    assert!(entry.parse_status.is_degraded());
}

#[test]
fn test_parse_empty_frontmatter_fixture() {
    let content = load_fixture("empty_frontmatter.md");
    let entry = parser::parse_skill_md(&content, "fallback-name");

    assert_eq!(entry.metadata.name, "fallback-name"); // from dir_name
    assert!(entry.parse_status.is_degraded());
}

#[test]
fn test_parse_invalid_yaml_fixture() {
    let content = load_fixture("invalid_yaml.md");
    let entry = parser::parse_skill_md(&content, "fallback-name");

    assert_eq!(entry.metadata.name, "fallback-name");
    assert!(entry.parse_status.is_error());
}

#[test]
fn test_parse_missing_name_fixture() {
    let content = load_fixture("missing_name.md");
    let entry = parser::parse_skill_md(&content, "fallback-name");

    assert_eq!(entry.metadata.name, "fallback-name"); // from dir_name
    assert!(entry.parse_status.is_degraded());
}

#[test]
fn test_parse_missing_description_fixture() {
    let content = load_fixture("missing_description.md");
    let entry = parser::parse_skill_md(&content, "no-desc");

    assert_eq!(entry.metadata.name, "no-desc");
    // Description should be extracted from body
    assert!(!entry.metadata.description.is_empty());
    assert!(entry.parse_status.is_degraded());
}

#[test]
fn test_parse_long_description_fixture() {
    let content = load_fixture("long_description.md");
    let entry = parser::parse_skill_md(&content, "long-desc");

    assert!(entry.parse_status.is_ok());
}

#[test]
fn test_parse_mixed_params_fixture() {
    let content = load_fixture("mixed_params.md");
    let entry = parser::parse_skill_md(&content, "mixed-params");

    // Should parse some valid params and mark as degraded
    assert!(!entry.parameters.is_empty());
    assert!(entry.parse_status.is_degraded());
}

#[test]
fn test_parse_empty_fixture() {
    let content = load_fixture("empty.md");
    let entry = parser::parse_skill_md(&content, "empty-skill");

    assert!(entry.parse_status.is_error());
}

#[test]
fn test_parse_anthropic_compatible_fixture() {
    let content = load_fixture("anthropic_compatible.md");
    let entry = parser::parse_skill_md(&content, "anthropic-skill");

    // Should parse successfully (Anthropic format is a subset of our format)
    assert!(!entry.metadata.name.is_empty());
    assert!(!entry.metadata.description.is_empty());
}

#[test]
fn test_parse_skill_file_directly() {
    let fixture_path = common::fixture_path("valid_full.md");
    let entry = parser::parse_skill_file(&fixture_path).expect("should parse file");

    assert_eq!(entry.metadata.name, "web-search");
    assert!(entry.source_path.ends_with("valid_full.md"));
}

#[test]
fn test_parse_file_exceeding_default_limit_is_rejected() {
    // large_file.md (1,060,479 bytes) was checked in specifically to exceed
    // the 1 MiB default, but nothing referenced it: the FileTooLarge guard
    // had no coverage. The rejection must happen before reading the body.
    let fixture_path = common::fixture_path("large_file.md");
    let err =
        parser::parse_skill_file(&fixture_path).expect_err("oversized skill file must be rejected");
    match err {
        parser::ParseError::FileTooLarge { size, max } => {
            assert_eq!(max, 1_048_576, "default limit must be 1 MiB");
            assert!(size > max, "fixture must exceed the limit, got {size}");
        }
        other => panic!("expected FileTooLarge, got {other:?}"),
    }
}

#[test]
fn test_parse_large_file_under_raised_limit_succeeds() {
    // The same fixture parses once the configured limit admits it: the
    // guard is size-driven, not content-driven (store.rs passes
    // config.max_skill_size through parse_skill_file_with_limit).
    let fixture_path = common::fixture_path("large_file.md");
    let entry = parser::parse_skill_file_with_limit(&fixture_path, 2 * 1_048_576)
        .expect("fixture must parse under a raised limit");

    assert_eq!(entry.metadata.name, "large-skill");
    assert_eq!(entry.metadata.description, "A very large skill file");
    assert!(
        entry.parse_status.is_ok(),
        "valid frontmatter must not degrade: {:?}",
        entry.parse_status
    );
}

#[test]
fn test_size_guard_fires_before_reading_the_body() {
    // A file that is over the limit AND not valid UTF-8 can only answer
    // FileTooLarge when the guard runs on metadata before any read: a
    // read-first implementation would surface a decode/IO error instead,
    // and a huge skill file would be fully allocated before rejection.
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("SKILL.md");
    let mut body = vec![0xff, 0xfe];
    body.resize(64, b'x');
    std::fs::write(&path, &body).expect("write oversized invalid-utf8 file");

    let err = parser::parse_skill_file_with_limit(&path, 32)
        .expect_err("oversized file must be rejected before reading");
    assert!(
        matches!(err, parser::ParseError::FileTooLarge { size: 64, max: 32 }),
        "guard must fire on metadata first, got {err:?}"
    );
}

#[test]
fn test_limit_boundary_is_exclusive() {
    // The guard is `size > max`: a file exactly at the limit is legal and
    // must parse. Pinning the boundary keeps a `>=` regression from
    // silently rejecting legitimate at-limit skill files.
    let fixture_path = common::fixture_path("large_file.md");
    let exact = std::fs::metadata(&fixture_path)
        .expect("fixture metadata")
        .len() as usize;
    let entry = parser::parse_skill_file_with_limit(&fixture_path, exact)
        .expect("a file exactly at the limit must parse");
    assert_eq!(entry.metadata.name, "large-skill");
}

#[cfg(unix)]
#[test]
fn test_parse_fifo_fails_fast_without_blocking() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::OpenOptionsExt;
    use std::time::Duration;

    let dir = tempfile::tempdir().expect("tempdir");
    let fifo_path = dir.path().join("SKILL.md");
    let c_path = CString::new(fifo_path.as_os_str().as_bytes()).expect("fifo CString");
    assert_eq!(
        unsafe { libc::mkfifo(c_path.as_ptr(), 0o644) },
        0,
        "mkfifo failed: {}",
        std::io::Error::last_os_error()
    );

    let thread_path = fifo_path.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let res = parser::parse_skill_file_with_limit(&thread_path, 1024);
        let _ = tx.send(res);
    });

    let res = match rx.recv_timeout(Duration::from_secs(3)) {
        Ok(res) => res,
        Err(err) => {
            // Unblock a pre-fix blocking reader so the test process exits cleanly.
            let _ = std::fs::OpenOptions::new()
                .write(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(&fifo_path);
            panic!("parse_skill_file_with_limit stalled on a FIFO: {err}");
        }
    };

    match res {
        Err(parser::ParseError::IoError(err)) => {
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::InvalidInput,
                "FIFO must be rejected as InvalidInput, got {err:?}"
            );
        }
        other => panic!("expected IoError(InvalidInput) for FIFO, got {other:?}"),
    }
}

#[cfg(unix)]
#[test]
fn test_parse_symlink_to_fifo_fails_fast_without_blocking() {
    use std::ffi::CString;
    use std::os::unix::ffi::OsStrExt;
    use std::os::unix::fs::{OpenOptionsExt, symlink};
    use std::time::Duration;

    let dir = tempfile::tempdir().expect("tempdir");
    let fifo_path = dir.path().join("pipe");
    let link_path = dir.path().join("SKILL.md");
    let c_path = CString::new(fifo_path.as_os_str().as_bytes()).expect("fifo CString");
    assert_eq!(
        unsafe { libc::mkfifo(c_path.as_ptr(), 0o644) },
        0,
        "mkfifo failed: {}",
        std::io::Error::last_os_error()
    );
    symlink(&fifo_path, &link_path).expect("symlink to fifo");

    let thread_path = link_path.clone();
    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let res = parser::parse_skill_file_with_limit(&thread_path, 1024);
        let _ = tx.send(res);
    });

    let res = match rx.recv_timeout(Duration::from_secs(3)) {
        Ok(res) => res,
        Err(err) => {
            let _ = std::fs::OpenOptions::new()
                .write(true)
                .custom_flags(libc::O_NONBLOCK)
                .open(&fifo_path);
            panic!("parse_skill_file_with_limit stalled on a symlinked FIFO: {err}");
        }
    };

    match res {
        Err(parser::ParseError::IoError(err)) => {
            assert_eq!(
                err.kind(),
                std::io::ErrorKind::InvalidInput,
                "symlinked FIFO must be rejected as InvalidInput, got {err:?}"
            );
        }
        other => panic!("expected IoError(InvalidInput) for symlinked FIFO, got {other:?}"),
    }
}

#[cfg(unix)]
#[test]
fn test_parse_character_device_is_rejected() {
    let err = parser::parse_skill_file_with_limit(std::path::Path::new("/dev/null"), 64)
        .expect_err("/dev/null must be rejected as non-regular file");
    match err {
        parser::ParseError::IoError(io_err) => {
            assert_eq!(
                io_err.kind(),
                std::io::ErrorKind::InvalidInput,
                "character device must be rejected as InvalidInput, got {io_err:?}"
            );
        }
        other => panic!("expected IoError(InvalidInput) for /dev/null, got {other:?}"),
    }
}

#[cfg(unix)]
#[test]
fn test_parse_dev_zero_is_rejected_without_reading() {
    use std::time::Duration;

    let (tx, rx) = std::sync::mpsc::channel();
    std::thread::spawn(move || {
        let res = parser::parse_skill_file_with_limit(std::path::Path::new("/dev/zero"), 64);
        let _ = tx.send(res);
    });

    let res = rx
        .recv_timeout(Duration::from_secs(3))
        .expect("parse_skill_file_with_limit stalled reading /dev/zero");
    match res {
        Err(parser::ParseError::IoError(io_err)) => {
            assert_eq!(
                io_err.kind(),
                std::io::ErrorKind::InvalidInput,
                "/dev/zero must be rejected as InvalidInput, got {io_err:?}"
            );
        }
        other => panic!("expected IoError(InvalidInput) for /dev/zero, got {other:?}"),
    }
}

#[test]
fn test_parse_directory_is_rejected_as_invalid_input() {
    let dir = tempfile::tempdir().expect("tempdir");
    let err = parser::parse_skill_file_with_limit(dir.path(), 1024)
        .expect_err("directory must be rejected as non-regular file");
    match err {
        parser::ParseError::IoError(io_err) => {
            assert_eq!(
                io_err.kind(),
                std::io::ErrorKind::InvalidInput,
                "directory must be rejected as InvalidInput, got {io_err:?}"
            );
        }
        other => panic!("expected IoError(InvalidInput) for directory, got {other:?}"),
    }
}

#[cfg(unix)]
#[test]
fn test_parse_valid_symlink_follows_target() {
    use std::os::unix::fs::symlink;

    let dir = tempfile::tempdir().expect("tempdir");
    let target_dir = dir.path().join("real-skill");
    let link_dir = dir.path().join("linked-skill");
    std::fs::create_dir_all(&target_dir).expect("create target dir");
    std::fs::create_dir_all(&link_dir).expect("create link dir");

    let target_path = target_dir.join("SKILL.md");
    let link_path = link_dir.join("SKILL.md");
    std::fs::write(
        &target_path,
        "---\nname: linked-skill\ndescription: Follows valid symlink\n---\n\nBody.\n",
    )
    .expect("write target skill");
    symlink(&target_path, &link_path).expect("create symlink");

    let entry = parser::parse_skill_file_with_limit(&link_path, 1024)
        .expect("valid symlink to regular SKILL.md must be followed");
    assert_eq!(entry.metadata.name, "linked-skill");
    assert_eq!(entry.metadata.description, "Follows valid symlink");
    assert!(entry.parse_status.is_ok());
    assert_eq!(entry.source_path, link_path);
}

#[test]
fn test_parse_exact_limit_multibyte_and_usize_max() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("SKILL.md");
    let content = "---\nname: exact-skill\ndescription: Caf\u{00e9}\n---\n\nBody.\n";
    std::fs::write(&path, content).expect("write skill");
    let exact = content.len();

    let entry = parser::parse_skill_file_with_limit(&path, exact)
        .expect("file exactly at max_size must parse");
    assert_eq!(entry.metadata.name, "exact-skill");
    assert_eq!(entry.metadata.description, "Caf\u{00e9}");

    let err = parser::parse_skill_file_with_limit(&path, exact - 1)
        .expect_err("file 1 byte over max_size must fail");
    assert!(
        matches!(
            err,
            parser::ParseError::FileTooLarge { size, max }
                if size == exact && max == exact - 1
        ),
        "expected FileTooLarge {{ size: {exact}, max: {} }}, got {err:?}",
        exact - 1
    );

    let entry_max = parser::parse_skill_file_with_limit(&path, usize::MAX)
        .expect("usize::MAX limit must not overflow read cap");
    assert_eq!(entry_max.metadata.name, "exact-skill");
}

#[test]
fn test_parse_invalid_utf8_within_limit_returns_invalid_data() {
    let dir = tempfile::tempdir().expect("tempdir");
    let path = dir.path().join("SKILL.md");
    std::fs::write(&path, [0xff, 0xfe, b'a', b'b']).expect("write invalid utf8");

    let err = parser::parse_skill_file_with_limit(&path, 64)
        .expect_err("invalid UTF-8 within limit must return IoError");
    match err {
        parser::ParseError::IoError(io_err) => {
            assert_eq!(
                io_err.kind(),
                std::io::ErrorKind::InvalidData,
                "invalid UTF-8 must map to InvalidData, got {io_err:?}"
            );
        }
        other => panic!("expected IoError(InvalidData), got {other:?}"),
    }
}

#[cfg(target_os = "linux")]
#[test]
fn test_parse_proc_virtual_file_bounded_by_stream_cap() {
    // On Linux, `/proc/self/status` reports S_IFREG with st_size == 0 while
    // yielding hundreds of bytes on read, exercising the post-fstat read cap.
    let err = parser::parse_skill_file_with_limit(std::path::Path::new("/proc/self/status"), 16)
        .expect_err("zero-size /proc file exceeding cap must return FileTooLarge");
    assert!(
        matches!(err, parser::ParseError::FileTooLarge { size: 17, max: 16 }),
        "expected FileTooLarge {{ size: 17, max: 16 }}, got {err:?}"
    );
}
