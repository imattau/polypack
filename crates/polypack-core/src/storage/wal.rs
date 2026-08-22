//! WAL entry types mirroring `specification/persistence.md` and the
//! TypeScript `WalEntry`.

use crate::model::{Edge, Node};

// `Node` carries several optional provenance/activation fields, making it far
// larger than the other variants. Boxing it would ripple through every WAL
// call site for no real benefit here — entries are appended one at a time,
// not batched in a way where the size difference matters.
#[derive(Debug, Clone, PartialEq)]
#[allow(clippy::large_enum_variant)]
pub enum WalEntry {
    PutNode(Node),
    DeleteNode(String),
    PutEdge(Edge),
    DeleteEdge(String),
    PutVector { id: String, vector: Vec<f64> },
    DeleteVector(String),
    ClearAll,
}
