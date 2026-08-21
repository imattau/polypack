//! Explicit physical snapshot/WAL format migrations.
//!
//! Normal store opening remains strict: unsupported bytes are rejected. This
//! module provides an administrative path for applications that deliberately
//! register a versioned byte migration before restoring the result.

use super::store::{Storage, INDEXES_FILE, MUTATION_LOG_FILE, SCHEMAS_FILE, SNAPSHOT_FILE, WAL_FILE};
use crate::error::{PolypackError, Result};
use std::sync::Arc;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum FormatArtifact { Snapshot, Wal }

type MigrationFn = Arc<dyn Fn(&[u8]) -> Result<Vec<u8>> + Send + Sync>;

struct FormatStep { artifact: FormatArtifact, from: u64, to: u64, migrate: MigrationFn }

/// Registry of explicit, contiguous physical-format migrations.
#[derive(Default)]
pub struct FormatMigrationRegistry { steps: Vec<FormatStep> }

impl FormatMigrationRegistry {
    pub fn new() -> Self { Self::default() }

    pub fn register<F>(&mut self, artifact: FormatArtifact, from: u64, to: u64, migrate: F) -> Result<()>
    where F: Fn(&[u8]) -> Result<Vec<u8>> + Send + Sync + 'static {
        if to <= from { return Err(PolypackError::InvalidArgument("format migration target must be greater than source".into())); }
        if self.steps.iter().any(|step| step.artifact == artifact && step.from == from) {
            return Err(PolypackError::InvalidArgument(format!("format migration from {from} is already registered")));
        }
        self.steps.push(FormatStep { artifact, from, to, migrate: Arc::new(migrate) });
        Ok(())
    }

    pub fn migrate(&self, artifact: FormatArtifact, from: u64, to: u64, bytes: &[u8]) -> Result<Vec<u8>> {
        if to < from { return Err(PolypackError::InvalidArgument("format migration target must not precede source".into())); }
        let mut version = from;
        let mut output = bytes.to_vec();
        while version < to {
            let step = self.steps.iter().find(|step| step.artifact == artifact && step.from == version)
                .ok_or_else(|| PolypackError::InvalidArgument(format!("no contiguous format migration from {version} to {to}")))?;
            if step.to > to { return Err(PolypackError::InvalidArgument(format!("format migration from {version} overshoots target {to}"))); }
            output = (step.migrate)(&output)?;
            version = step.to;
        }
        Ok(output)
    }
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct FormatMigrationReport {
    pub snapshot_migrated: bool,
    pub wal_migrated: bool,
    pub bytes_read: usize,
    pub bytes_written: usize,
    pub dry_run: bool,
}

/// Migrate physical snapshot/WAL bytes into another storage adapter.
///
/// Auxiliary logical-mutation, index, and schema metadata are copied without
/// transformation. With `dry_run`, validation and callbacks run but the
/// destination is not modified.
pub fn migrate_storage(
    source: &dyn Storage,
    destination: &mut dyn Storage,
    registry: &FormatMigrationRegistry,
    snapshot_from: u64,
    wal_from: u64,
    target_version: u64,
    dry_run: bool,
) -> Result<FormatMigrationReport> {
    let mut report = FormatMigrationReport { dry_run, ..Default::default() };
    for (name, artifact, from) in [(SNAPSHOT_FILE, FormatArtifact::Snapshot, snapshot_from), (WAL_FILE, FormatArtifact::Wal, wal_from)] {
        let Some(bytes) = source.read(name)? else { continue };
        report.bytes_read += bytes.len();
        let migrated = registry.migrate(artifact, from, target_version, &bytes)?;
        report.bytes_written += migrated.len();
        if artifact == FormatArtifact::Snapshot { report.snapshot_migrated = from != target_version; }
        if artifact == FormatArtifact::Wal { report.wal_migrated = from != target_version; }
        if !dry_run { destination.write(name, &migrated)?; }
    }
    for name in [MUTATION_LOG_FILE, INDEXES_FILE, SCHEMAS_FILE] {
        if let Some(bytes) = source.read(name)? {
            report.bytes_read += bytes.len();
            report.bytes_written += bytes.len();
            if !dry_run { destination.write(name, &bytes)?; }
        }
    }
    Ok(report)
}
