use std::path::{Path, PathBuf};

use ws_ckpt_common::{
    load_config_file, save_config_file, CleanupRetention, FileConfig, WsCkptError,
};

struct CwdGuard(PathBuf);

impl Drop for CwdGuard {
    fn drop(&mut self) {
        let _ = std::env::set_current_dir(&self.0);
    }
}

#[test]
fn save_config_file_supports_bare_relative_filename_atomically() {
    let original_cwd = std::env::current_dir().unwrap();
    let _guard = CwdGuard(original_cwd);
    let dir = tempfile::tempdir().unwrap();
    std::env::set_current_dir(dir.path()).unwrap();

    let rel = Path::new("config.toml");
    let fc1 = FileConfig {
        auto_cleanup_keep: Some(CleanupRetention::Count(5)),
        ..Default::default()
    };
    save_config_file(rel, &fc1).unwrap();
    assert!(!dir.path().join("config.toml.tmp").exists());
    assert_eq!(load_config_file(rel).unwrap(), fc1);

    #[cfg(unix)]
    let backup = {
        let backup = dir.path().join("config.toml.hardlink");
        std::fs::hard_link(rel, &backup).unwrap();
        backup
    };

    let fc2 = FileConfig {
        auto_cleanup_keep: Some(CleanupRetention::Count(9)),
        health_check_interval_secs: Some(45),
        ..Default::default()
    };
    save_config_file(rel, &fc2).unwrap();
    assert!(!dir.path().join("config.toml.tmp").exists());
    assert_eq!(load_config_file(rel).unwrap(), fc2);
    #[cfg(unix)]
    assert_eq!(load_config_file(&backup).unwrap(), fc1);

    for invalid in [Path::new("/"), Path::new(""), Path::new(".")] {
        let err = save_config_file(invalid, &fc2).unwrap_err();
        assert!(
            matches!(err, WsCkptError::Config(_)),
            "expected Config error for {invalid:?}, got {err:?}"
        );
    }
}
