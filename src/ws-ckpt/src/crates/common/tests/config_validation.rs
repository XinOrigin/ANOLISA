use ws_ckpt_common::{
    load_config_file, load_config_file_unvalidated, save_config_file, FileConfig, WsCkptError,
};

#[test]
fn load_config_file_accepts_known_and_default_backend_types() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");

    // Missing file uses default "auto".
    let missing = load_config_file(&path).unwrap();
    assert_eq!(missing.backend.r#type, "auto");

    // Empty file and [backend] without explicit `type` default to "auto".
    for content in ["", "[backend]\n"] {
        std::fs::write(&path, content).unwrap();
        let config = load_config_file(&path).unwrap();
        assert_eq!(config.backend.r#type, "auto");
    }

    for valid in ["auto", "btrfs-base", "btrfs-loop"] {
        std::fs::write(&path, format!("[backend]\ntype = {:?}\n", valid)).unwrap();
        let config = load_config_file(&path).unwrap();
        assert_eq!(config.backend.r#type, valid);
    }
}

#[test]
fn load_config_file_rejects_unknown_backend_type() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");

    for invalid in [
        "btrfs_base",
        "btrfs_loop",
        "overlay",
        "",
        " ",
        " auto",
        "btrfs-base ",
        "AUTO",
        "Btrfs-Base",
        "BTRFS-LOOP",
    ] {
        std::fs::write(&path, format!("[backend]\ntype = {:?}\n", invalid)).unwrap();
        match load_config_file(&path) {
            Err(WsCkptError::Config(msg)) => {
                assert!(
                    msg.contains("backend.type"),
                    "error should name backend.type, got: {msg}"
                );
                assert!(
                    msg.contains(&path.display().to_string()),
                    "error should include config path, got: {msg}"
                );
                assert!(
                    msg.contains(&format!("{invalid:?}")),
                    "error should include offending value {invalid:?}, got: {msg}"
                );
            }
            other => panic!("expected WsCkptError::Config for {invalid:?}, got {other:?}"),
        }
    }
}

#[test]
fn save_config_file_rejects_unknown_backend_type_without_writing_or_creating_dirs() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    let original = "auto_cleanup = true\n";
    std::fs::write(&path, original).unwrap();

    for invalid in ["btrfs_base", "overlay", "", " auto"] {
        let mut fc = FileConfig::default();
        fc.backend.r#type = invalid.to_string();
        match save_config_file(&path, &fc) {
            Err(WsCkptError::Config(msg)) => {
                assert!(
                    msg.contains("backend.type"),
                    "error should name backend.type for {invalid:?}, got: {msg}"
                );
            }
            other => panic!("save_config_file should reject {invalid:?}, got {other:?}"),
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), original);
    }

    let nested_dir = temp.path().join("nested");
    let nested_path = nested_dir.join("config.toml");
    let mut fc = FileConfig::default();
    fc.backend.r#type = "btrfs_base".to_string();
    let err = save_config_file(&nested_path, &fc).unwrap_err();
    assert!(matches!(err, WsCkptError::Config(ref msg) if msg.contains("backend.type")));
    assert!(!nested_dir.exists());
}

#[test]
fn invalid_backend_type_on_disk_can_be_repaired_via_unvalidated_load() {
    let temp = tempfile::tempdir().unwrap();
    let path = temp.path().join("config.toml");
    std::fs::write(
        &path,
        "auto_cleanup = true\n[backend]\ntype = \"btrfs_base\"\n",
    )
    .unwrap();

    assert!(matches!(
        load_config_file(&path),
        Err(WsCkptError::Config(_))
    ));

    let mut fc = load_config_file_unvalidated(&path).unwrap();
    assert_eq!(fc.backend.r#type, "btrfs_base");
    fc.backend.r#type = "btrfs-base".to_string();
    save_config_file(&path, &fc).unwrap();

    let repaired = load_config_file(&path).unwrap();
    assert_eq!(repaired, fc);
    assert_eq!(repaired.auto_cleanup, Some(true));
    assert_eq!(repaired.backend.r#type, "btrfs-base");
}
