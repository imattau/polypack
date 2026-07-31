//! WAL entry types mirroring `specification/persistence.md` and the
//! TypeScript `WalEntry`.

use crate::model::{Edge, Node};

#[derive(Debug, Clone, PartialEq)]
pub enum WalEntry {
    PutNode(Node),
    DeleteNode(String),
    PutEdge(Edge),
    DeleteEdge(String),
    PutVector { id: String, vector: Vec<f64> },
    DeleteVector(String),
    ClearAll,
}
