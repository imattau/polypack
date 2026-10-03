//! Explicit offline vector-precision migration for filesystem-backed Rust stores.

use super::file::FileStorage;
use super::store::{Store, StoreConfig, INDEXES_FILE, SCHEMAS_FILE};
use super::VectorPrecision;
use crate::error::{PolypackError, Result};
use crate::model::{ChangeBatch, VectorEntry};
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VectorPrecisionMigration {
    pub store_dir: PathBuf,
    pub backup_dir: PathBuf,
    pub from: VectorPrecision,
    pub to: VectorPrecision,
}

/// Rebuild a filesystem store at another precision and retain the source as a backup.
/// Stop all processes using the store before calling this function.
pub fn migrate_vector_precision(
    store_dir: impl AsRef<Path>,
    precision: VectorPrecision,
    backup_dir: Option<&Path>,
) -> Result<VectorPrecisionMigration> {
    let store_dir = absolute_path(store_dir.as_ref())?;
    let backup_dir = match backup_dir {
        Some(path) => absolute_path(path)?,
        None => {
            let stamp = SystemTime::now().duration_since(UNIX_EPOCH).unwrap_or_default().as_nanos();
            store_dir.with_file_name(format!("{}.backup-{stamp}", store_dir.file_name().and_then(|name| name.to_str()).unwrap_or("store")))
        }
    };
    if store_dir == backup_dir || store_dir.starts_with(&backup_dir) || backup_dir.starts_with(&store_dir) {
        return Err(PolypackError::InvalidArgument("backup directory must be separate from the store directory".into()));
    }
    if backup_dir.exists() { return Err(PolypackError::Storage(format!("backup directory already exists: {}", backup_dir.display()))); }
    if !store_dir.is_dir() { return Err(PolypackError::Storage(format!("store directory does not exist: {}", store_dir.display()))); }
    let temp_dir = store_dir.with_file_name(format!(".{}.migrate-{}", store_dir.file_name().and_then(|name| name.to_str()).unwrap_or("store"), std::process::id()));
    if temp_dir.exists() { return Err(PolypackError::Storage(format!("temporary directory already exists: {}", temp_dir.display()))); }

    let (from, nodes, edges, vectors, mutations, auxiliary) = {
        let mut source = Store::new(Box::new(FileStorage::open(&store_dir, false)?), StoreConfig::default());
        let from = source.vector_precision()?;
        source.compact()?;
        let ids = source.node_ids()?;
        let nodes = ids.iter().map(|id| source.get_node(id)).collect::<Result<Vec<_>>>()?.into_iter().flatten().collect::<Vec<_>>();
        let edges = source.edges_snapshot()?.into_iter().map(|(_, edge)| edge).collect::<Vec<_>>();
        let vectors = source.vectors_snapshot()?.into_iter().map(|(id, vector)| VectorEntry { id, vector }).collect::<Vec<_>>();
        let mutations = source.mutation_log()?;
        let mut auxiliary = Vec::new();
        for name in [INDEXES_FILE, SCHEMAS_FILE] {
            if let Some(bytes) = source.read_auxiliary(name)? { auxiliary.push((name, bytes)); }
        }
        source.close()?;
        (from, nodes, edges, vectors, mutations, auxiliary)
    };

    let result = (|| {
        let mut destination = Store::new(Box::new(FileStorage::open(&temp_dir, false)?), StoreConfig {
            vector_precision: Some(precision),
            compact_threshold: usize::MAX,
            ..Default::default()
        });
        for (name, bytes) in &auxiliary { destination.write_auxiliary(name, bytes)?; }
        destination.apply(&ChangeBatch { put_nodes: nodes.clone(), put_edges: edges.clone(), put_vectors: vectors.clone(), ..Default::default() })?;
        destination.replace_mutation_log(&mutations)?;
        destination.compact()?;
        let report = destination.verify()?;
        if !report.ok || report.node_count != nodes.len() || report.edge_count != edges.len() || report.vector_count != vectors.len() || destination.vector_precision()? != precision || destination.latest_mutation_sequence()? != mutations.last().map(|record| record.sequence).unwrap_or(0) {
            return Err(PolypackError::CorruptData(format!("migrated store verification failed: {:?}", report.errors)));
        }
        destination.close()?;
        Ok(())
    })();
    if let Err(error) = result {
        let _ = fs::remove_dir_all(&temp_dir);
        return Err(error);
    }
    fs::rename(&store_dir, &backup_dir).map_err(|error| { let _ = fs::remove_dir_all(&temp_dir); PolypackError::Storage(error.to_string()) })?;
    if let Err(error) = fs::rename(&temp_dir, &store_dir) {
        let restored = fs::rename(&backup_dir, &store_dir);
        let _ = fs::remove_dir_all(&temp_dir);
        if let Err(restore_error) = restored { return Err(PolypackError::Storage(format!("replacement failed ({error}); backup restore failed ({restore_error}); backup remains at {}", backup_dir.display()))); }
        return Err(PolypackError::Storage(error.to_string()));
    }
    Ok(VectorPrecisionMigration { store_dir, backup_dir, from, to: precision })
}

fn absolute_path(path: &Path) -> Result<PathBuf> {
    if path.is_absolute() { Ok(path.to_path_buf()) }
    else { std::env::current_dir().map(|cwd| cwd.join(path)).map_err(|error| PolypackError::Storage(error.to_string())) }
}
