use polypack_core::storage::{migrate_storage, FormatArtifact, FormatMigrationRegistry, InMemoryStorage, Storage};

#[test]
fn physical_format_migrations_are_explicit_and_dry_run_is_non_destructive() {
    let mut source = InMemoryStorage::new();
    source.write("snapshot.msgpack", b"legacy-snapshot").unwrap();
    source.write("wal.msgpack", b"legacy-wal").unwrap();
    let mut registry = FormatMigrationRegistry::new();
    registry.register(FormatArtifact::Snapshot, 0, 1, |bytes| {
        assert_eq!(bytes, b"legacy-snapshot");
        Ok(b"current-snapshot".to_vec())
    }).unwrap();
    registry.register(FormatArtifact::Wal, 0, 1, |bytes| {
        assert_eq!(bytes, b"legacy-wal");
        Ok(b"current-wal".to_vec())
    }).unwrap();
    let mut destination = InMemoryStorage::new();
    let dry = migrate_storage(&source, &mut destination, &registry, 0, 0, 1, true).unwrap();
    assert!(dry.dry_run && dry.snapshot_migrated && dry.wal_migrated);
    assert!(!destination.exists("snapshot.msgpack").unwrap());
    let report = migrate_storage(&source, &mut destination, &registry, 0, 0, 1, false).unwrap();
    assert_eq!(report.bytes_written, 27);
    assert_eq!(destination.read("snapshot.msgpack").unwrap(), Some(b"current-snapshot".to_vec()));
    assert_eq!(destination.read("wal.msgpack").unwrap(), Some(b"current-wal".to_vec()));
}

#[test]
fn missing_migration_steps_are_rejected() {
    let registry = FormatMigrationRegistry::new();
    let error = registry.migrate(FormatArtifact::Snapshot, 0, 1, b"bytes").unwrap_err();
    assert!(error.to_string().contains("no contiguous format migration"));
}
