//! Single-writer filesystem storage for direct Rust users.
//!
//! The lock is deliberately exclusive for writers. Read-only handles do not
//! acquire it and must be used only when the store is not being mutated. This
//! adapter does not claim multi-process writer support.

use super::store::{AdapterCapabilities, Storage, VectorSearchCapability};
use crate::error::{PolypackError, Result};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

const LOCK_FILE: &str = "store.lock";
const STALE_LOCK_MILLIS: u128 = 24 * 60 * 60 * 1000;

/// Filesystem-backed byte storage with an exclusive writer lock.
pub struct FileStorage {
    dir: PathBuf,
    read_only: bool,
    lock_token: Option<String>,
    lock_file: Option<File>,
}

impl FileStorage {
    /// Open a store directory. Writers acquire `store.lock`; read-only handles
    /// reject all mutations and do not remove a writer's lock on drop.
    pub fn open(path: impl AsRef<Path>, read_only: bool) -> Result<Self> {
        let dir = path.as_ref().to_path_buf();
        fs::create_dir_all(&dir).map_err(|error| PolypackError::Storage(error.to_string()))?;
        if read_only {
            return Ok(Self { dir, read_only, lock_token: None, lock_file: None });
        }

        let lock_path = dir.join(LOCK_FILE);
        let started_at = now_millis();
        let token = format!("{}-{started_at}", std::process::id());
        let metadata = format!(r#"{{"pid":{},"startedAt":{},"token":"{}"}}"#, std::process::id(), started_at, token);
        for attempt in 0..2 {
            match OpenOptions::new().read(true).write(true).create_new(true).open(&lock_path) {
                Ok(mut file) => {
                    file.write_all(metadata.as_bytes()).map_err(|error| PolypackError::Storage(error.to_string()))?;
                    file.sync_all().map_err(|error| PolypackError::Storage(error.to_string()))?;
                    return Ok(Self { dir, read_only, lock_token: Some(token), lock_file: Some(file) });
                }
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists && attempt == 0 => {
                    if stale_lock(&lock_path, started_at) {
                        fs::remove_file(&lock_path).map_err(|remove_error| PolypackError::Storage(remove_error.to_string()))?;
                        continue;
                    }
                    return Err(PolypackError::Storage(format!("store is already locked: {}", lock_path.display())));
                }
                Err(error) => return Err(PolypackError::Storage(error.to_string())),
            }
        }
        Err(PolypackError::Storage(format!("store is already locked: {}", lock_path.display())))
    }

    pub fn read_only(&self) -> bool { self.read_only }

    fn writable(&self) -> Result<()> {
        if self.read_only { Err(PolypackError::Storage("store was opened read-only".into())) } else { Ok(()) }
    }

    fn path(&self, name: &str) -> PathBuf { self.dir.join(name) }
}

fn now_millis() -> u128 {
    SystemTime::now().duration_since(UNIX_EPOCH).map(|duration| duration.as_millis()).unwrap_or(0)
}

fn stale_lock(path: &Path, now: u128) -> bool {
    let Ok(contents) = fs::read_to_string(path) else { return false };
    let Ok(value) = serde_json::from_str::<serde_json::Value>(&contents) else { return false };
    value.get("startedAt").and_then(serde_json::Value::as_u64).map(|started| now.saturating_sub(started as u128) > STALE_LOCK_MILLIS).unwrap_or(false)
}

impl Drop for FileStorage {
    fn drop(&mut self) {
        let Some(mut file) = self.lock_file.take() else { return };
        let mut contents = String::new();
        let _ = file.seek(SeekFrom::Start(0));
        let _ = file.read_to_string(&mut contents);
        let owns_lock = self.lock_token.as_ref().is_some_and(|token| contents.contains(&format!(r#""token":"{}""#, token)));
        drop(file);
        if owns_lock { let _ = fs::remove_file(self.path(LOCK_FILE)); }
    }
}

impl Storage for FileStorage {
    fn read(&self, name: &str) -> Result<Option<Vec<u8>>> {
        match fs::read(self.path(name)) {
            Ok(bytes) => Ok(Some(bytes)),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(None),
            Err(error) => Err(PolypackError::Storage(error.to_string())),
        }
    }

    fn write(&mut self, name: &str, data: &[u8]) -> Result<()> {
        self.writable()?;
        let target = self.path(name);
        let temp = self.path(&format!(".{name}.tmp-{}", self.lock_token.as_deref().unwrap_or("readonly")));
        let mut file = File::create(&temp).map_err(|error| PolypackError::Storage(error.to_string()))?;
        file.write_all(data).map_err(|error| PolypackError::Storage(error.to_string()))?;
        // No `sync_all()` here: durability is `Store`'s call, via the explicit
        // `sync()`/`sync_dir()` it issues only under `Durability::Fsync`.
        // Syncing unconditionally on every write silently upgraded
        // `Durability::Process` ("written to the OS, not fsynced") into an
        // always-fsync adapter, defeating the whole point of the distinction.
        drop(file);
        if let Err(error) = fs::rename(&temp, &target) {
            // On Windows, rename fails with AlreadyExists when the target is present;
            // only that specific failure warrants clearing the target before retrying.
            // Any other error (disk full, permissions, AV lock, etc.) must not delete
            // the existing durable file, or a failed retry would lose it permanently.
            if error.kind() == std::io::ErrorKind::AlreadyExists {
                fs::remove_file(&target).map_err(|remove_error| PolypackError::Storage(remove_error.to_string()))?;
                fs::rename(&temp, &target).map_err(|rename_error| PolypackError::Storage(rename_error.to_string()))?;
            } else {
                let _ = fs::remove_file(&temp);
                return Err(PolypackError::Storage(error.to_string()));
            }
        }
        Ok(())
    }

    fn append(&mut self, name: &str, data: &[u8]) -> Result<()> {
        self.writable()?;
        let mut file = OpenOptions::new().create(true).append(true).open(self.path(name)).map_err(|error| PolypackError::Storage(error.to_string()))?;
        file.write_all(data).map_err(|error| PolypackError::Storage(error.to_string()))?;
        // See `write()`: fsync is `Store`'s call via the explicit `sync()`
        // it issues under `Durability::Fsync`, not this adapter's default.
        Ok(())
    }

    fn delete(&mut self, name: &str) -> Result<()> {
        self.writable()?;
        match fs::remove_file(self.path(name)) {
            Ok(()) => Ok(()),
            Err(error) if error.kind() == std::io::ErrorKind::NotFound => Ok(()),
            Err(error) => Err(PolypackError::Storage(error.to_string())),
        }
    }

    fn exists(&self, name: &str) -> Result<bool> { Ok(self.path(name).exists()) }

    fn sync(&self, name: &str) -> Result<()> {
        let file = File::open(self.path(name)).map_err(|error| PolypackError::Storage(error.to_string()))?;
        file.sync_all().map_err(|error| PolypackError::Storage(error.to_string()))
    }

    fn sync_dir(&self) -> Result<()> { Ok(()) }

    fn capabilities(&self) -> AdapterCapabilities {
        AdapterCapabilities { atomic_batches: true, transactions: true, fsync: true, secondary_indexes: true, snapshots: true, change_feed: true, concurrent_writers: false, vector_search: VectorSearchCapability::Exact }
    }
}
