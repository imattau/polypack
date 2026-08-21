//! Error taxonomy mirroring `specification/errors.md`.

use std::fmt;

/// Stable cross-language error categories.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolypackError {
    InvalidArgument(String),
    DimensionMismatch { expected: usize, got: usize },
    RangeOutOfBounds(String),
    Conflict { id: String, expected: u64, actual: u64 },
    ResourceLimit { name: String, limit: usize },
    Closed,
    FormatVersion(u64),
    CorruptData(String),
    Storage(String),
    NotImplemented(String),
}

impl fmt::Display for PolypackError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            PolypackError::InvalidArgument(m) => write!(f, "invalid_argument: {m}"),
            PolypackError::DimensionMismatch { expected, got } => {
                write!(f, "dimension_mismatch: expected {expected} dimensions, got {got}")
            }
            PolypackError::RangeOutOfBounds(m) => write!(f, "range_out_of_bounds: {m}"),
            PolypackError::Conflict { id, expected, actual } => {
                write!(f, "conflict: record {id} has revision {actual}, expected {expected}")
            }
            PolypackError::ResourceLimit { name, limit } => write!(f, "resource_limit: {name} exceeded limit {limit}"),
            PolypackError::Closed => write!(f, "closed: operation on a closed store"),
            PolypackError::FormatVersion(v) => write!(f, "format_version: unsupported version {v}"),
            PolypackError::CorruptData(m) => write!(f, "corrupt_data: {m}"),
            PolypackError::Storage(m) => write!(f, "storage: {m}"),
            PolypackError::NotImplemented(m) => write!(f, "not_implemented: {m}"),
        }
    }
}

impl std::error::Error for PolypackError {}

/// Stable machine-readable code for each category.
impl PolypackError {
    pub fn code(&self) -> &'static str {
        match self {
            PolypackError::InvalidArgument(_) => "invalid_argument",
            PolypackError::DimensionMismatch { .. } => "dimension_mismatch",
            PolypackError::RangeOutOfBounds(_) => "range_out_of_bounds",
            PolypackError::Conflict { .. } => "conflict",
            PolypackError::ResourceLimit { .. } => "resource_limit",
            PolypackError::Closed => "closed",
            PolypackError::FormatVersion(_) => "format_version",
            PolypackError::CorruptData(_) => "corrupt_data",
            PolypackError::Storage(_) => "storage",
            PolypackError::NotImplemented(_) => "not_implemented",
        }
    }
}

pub type Result<T> = std::result::Result<T, PolypackError>;
