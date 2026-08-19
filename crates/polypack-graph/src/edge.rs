//! Edge representation, mirrors the `EdgeOwnership` union and the
//! `EdgeEntry`/`EdgeIndex` shapes in `src/graph.ts`.

/// Ownership semantics for a directed edge, controlling cascade-delete and
/// orphan-detection behavior in [`crate::Graph::remove_node`] /
/// [`crate::Graph::remove_edges`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum EdgeOwnership {
    /// Target is deleted when this is the last owning source (cascade delete).
    Owned,
    /// Target is not deleted, but `on_orphan` fires if it becomes disconnected.
    Shared,
    /// No cascade behavior.
    Reference,
}

/// One outgoing edge from a source node. The adjacency map uses a private
/// lookup key, while `id` is the independent durable identity.
#[derive(Clone, Debug)]
pub struct EdgeEntry {
    pub id: String,
    pub revision: u64,
    pub target: String,
    pub edge_type: String,
    pub data: Option<serde_json::Map<String, serde_json::Value>>,
    pub ownership: EdgeOwnership,
}

impl EdgeOwnership {
    fn as_str(self) -> &'static str {
        match self {
            EdgeOwnership::Owned => "owned",
            EdgeOwnership::Shared => "shared",
            EdgeOwnership::Reference => "reference",
        }
    }
}

/// `polypack_core::Edge` has no dedicated ownership column — matching
/// `graph.ts`, ownership is embedded in `data` under this reserved key so it
/// survives a `flush`/`warm` round trip through `Store`. `get_edges` and
/// friends never see this key; [`encode_ownership`]/[`decode_ownership`] are
/// only used at the persistence boundary (`Graph::flush`,
/// `Graph::rebuild_edge_index`).
const OWNERSHIP_KEY: &str = "__ownership";

/// Embed `ownership` into `data` for persistence. Always writes the key
/// (even for the default `Reference`) so restoration never has to guess.
pub(crate) fn encode_ownership(
    mut data: Option<serde_json::Map<String, serde_json::Value>>,
    ownership: EdgeOwnership,
) -> Option<serde_json::Map<String, serde_json::Value>> {
    data.get_or_insert_with(serde_json::Map::new)
        .insert(OWNERSHIP_KEY.to_string(), serde_json::Value::String(ownership.as_str().to_string()));
    data
}

/// Inverse of [`encode_ownership`]: split a persisted edge's `data` back
/// into `(ownership, data-without-the-reserved-key)`. Missing/unrecognized
/// markers default to `Reference`.
pub(crate) fn decode_ownership(
    mut data: Option<serde_json::Map<String, serde_json::Value>>,
) -> (EdgeOwnership, Option<serde_json::Map<String, serde_json::Value>>) {
    let ownership = match data.as_ref().and_then(|d| d.get(OWNERSHIP_KEY)).and_then(|v| v.as_str()) {
        Some("owned") => EdgeOwnership::Owned,
        Some("shared") => EdgeOwnership::Shared,
        _ => EdgeOwnership::Reference,
    };
    if let Some(map) = data.as_mut() {
        map.remove(OWNERSHIP_KEY);
        if map.is_empty() {
            data = None;
        }
    }
    (ownership, data)
}
