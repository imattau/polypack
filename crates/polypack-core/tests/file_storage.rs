use polypack_core::storage::{FileStorage, Storage};
use std::time::{SystemTime, UNIX_EPOCH};

fn temp_dir() -> std::path::PathBuf {
    std::env::temp_dir().join(format!("polypack-file-storage-{}-{}", std::process::id(), SystemTime::now().duration_since(UNIX_EPOCH).unwrap().as_nanos()))
}

#[test]
fn exclusive_writer_lock_and_read_only_mode_are_enforced() {
    let dir = temp_dir();
    let mut writer = FileStorage::open(&dir, false).unwrap();
    assert!(!writer.read_only());
    assert!(!writer.capabilities().concurrent_writers);
    assert!(FileStorage::open(&dir, false).is_err());
    let mut reader = FileStorage::open(&dir, true).unwrap();
    assert!(reader.read_only());
    assert!(reader.write("value", b"nope").is_err());
    writer.write("value", b"ok").unwrap();
    assert_eq!(reader.read("value").unwrap(), Some(b"ok".to_vec()));
    drop(reader);
    drop(writer);
    std::fs::remove_dir_all(dir).unwrap();
}
