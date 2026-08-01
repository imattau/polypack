//! Core data envelopes matching `specification/data-model.md` and
//! `specification/change-batch.schema.json`.

use crate::error::{PolypackError, Result};
use serde::{Deserialize, Serialize};

fn is_finite_vec(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// Durable activation state for a node (see `crate::activation`). All fields
/// are persisted and replicated; transient, runtime-only attention is held by
/// the `ActivationEngine` in `polypack-graph` and never serialized. Decay is a
/// pure function of elapsed time anchored at `last_meaningful_activation`, so
/// replicas with the same stored state compute identical current scores.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct NodeActivation {
    /// Current learned activation, decay-corrected on read. Clamped to [0, 1].
    pub score: f64,
    /// Long-term relevance that decays far slower than `score` (or never). Clamped to [0, 1].
    pub importance: f64,
    /// How many times this node has been meaningfully reinforced.
    pub reinforcement_count: u64,
    /// Epoch-ms anchor for decay. Both `score` and `importance` decay from here.
    pub last_meaningful_activation: i64,
}

/// A typed property-graph node. Serializes as camelCase JSON matching the
/// TypeScript `PolyNode`/`SerializedNode` shape (`type` on the wire, `node_type` in Rust).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Node {
    pub id: String,
    #[serde(rename = "type")]
    pub node_type: String,
    #[serde(default)]
    pub data: serde_json::Map<String, serde_json::Value>,
    #[serde(default)]
    pub vector: Option<Vec<f64>>,
    pub inserted_at: i64,
    pub updated_at: i64,
    /// Durable, syncable activation. Optional — absent until first reinforced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation: Option<NodeActivation>,
}

/// A directed property-graph edge. `id` is expected to be [`edge_id`] of
/// `(source, edge_type, target)`; `validate_edge` checks this invariant.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct Edge {
    pub id: String,
    pub source: String,
    pub target: String,
    #[serde(rename = "type")]
    pub edge_type: String,
    #[serde(default)]
    pub data: Option<serde_json::Map<String, serde_json::Value>>,
    pub created_at: i64,
}

/// A single stored vector, keyed by node id.
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct VectorEntry {
    pub id: String,
    pub vector: Vec<f64>,
}

/// One logical commit: puts and deletes across nodes, edges, and vectors,
/// applied atomically by [`crate::storage::Store::apply`]. Validate with
/// [`validate_batch`] before applying — all-or-nothing, like the TypeScript
/// `PersistenceChanges` contract.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ChangeBatch {
    #[serde(default)]
    pub put_nodes: Vec<Node>,
    #[serde(default)]
    pub delete_node_ids: Vec<String>,
    #[serde(default)]
    pub put_edges: Vec<Edge>,
    #[serde(default)]
    pub delete_edge_ids: Vec<String>,
    #[serde(default)]
    pub put_vectors: Vec<VectorEntry>,
    #[serde(default)]
    pub delete_vector_ids: Vec<String>,
}

/// Validate a single node envelope.
pub fn validate_node(n: &Node) -> Result<()> {
    if n.id.is_empty() {
        return Err(PolypackError::InvalidArgument("node id must not be empty".into()));
    }
    if n.node_type.is_empty() {
        return Err(PolypackError::InvalidArgument("node type must not be empty".into()));
    }
    if n.inserted_at < 0 || n.updated_at < 0 {
        return Err(PolypackError::RangeOutOfBounds(
            "node timestamps must be finite non-negative numbers".into(),
        ));
    }
    if let Some(v) = &n.vector {
        if !is_finite_vec(v) {
            return Err(PolypackError::InvalidArgument("vector must contain finite values".into()));
        }
    }
    if let Some(a) = &n.activation {
        validate_activation(a)?;
    }
    Ok(())
}

/// Validate a durable activation record: score/importance in [0, 1], finite,
/// and a non-negative anchor.
pub fn validate_activation(a: &NodeActivation) -> Result<()> {
    if !a.score.is_finite() || !(0.0..=1.0).contains(&a.score) {
        return Err(PolypackError::RangeOutOfBounds(
            "activation.score must be a finite number in [0, 1]".into(),
        ));
    }
    if !a.importance.is_finite() || !(0.0..=1.0).contains(&a.importance) {
        return Err(PolypackError::RangeOutOfBounds(
            "activation.importance must be a finite number in [0, 1]".into(),
        ));
    }
    if a.last_meaningful_activation < 0 {
        return Err(PolypackError::RangeOutOfBounds(
            "activation.lastMeaningfulActivation must be a finite non-negative number".into(),
        ));
    }
    Ok(())
}

/// Validate a single edge envelope.
pub fn validate_edge(e: &Edge) -> Result<()> {
    if e.source.is_empty() || e.edge_type.is_empty() || e.target.is_empty() {
        return Err(PolypackError::InvalidArgument(
            "edge source, type, and target must not be empty".into(),
        ));
    }
    if e.source.contains("::") || e.edge_type.contains("::") {
        return Err(PolypackError::RangeOutOfBounds(
            "edge source and type must not contain \"::\"".into(),
        ));
    }
    if e.created_at < 0 {
        return Err(PolypackError::RangeOutOfBounds(
            "edge createdAt must be finite non-negative".into(),
        ));
    }
    Ok(())
}

/// Canonical edge id: `source::type::target`.
pub fn edge_id(source: &str, edge_type: &str, target: &str) -> String {
    format!("{source}::{edge_type}::{target}")
}

/// Validate every element of a batch; reject the whole batch on any error.
pub fn validate_batch(batch: &ChangeBatch) -> Result<()> {
    for n in &batch.put_nodes {
        validate_node(n)?;
    }
    for e in &batch.put_edges {
        validate_edge(e)?;
    }
    for v in &batch.put_vectors {
        if v.id.is_empty() {
            return Err(PolypackError::InvalidArgument("vector id must not be empty".into()));
        }
        if !is_finite_vec(&v.vector) {
            return Err(PolypackError::InvalidArgument("vector must contain finite values".into()));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn node(id: &str) -> Node {
        Node {
            id: id.into(),
            node_type: "t".into(),
            data: serde_json::Map::new(),
            vector: None,
            inserted_at: 1,
            updated_at: 1,
            activation: None,
        }
    }

    #[test]
    fn rejects_empty_id() {
        assert_eq!(
            validate_node(&node("")),
            Err(PolypackError::InvalidArgument("node id must not be empty".into()))
        );
    }

    #[test]
    fn rejects_negative_timestamp() {
        let mut n = node("a");
        n.inserted_at = -1;
        assert!(matches!(validate_node(&n), Err(PolypackError::RangeOutOfBounds(_))));
    }

    #[test]
    fn rejects_non_finite_vector() {
        let mut n = node("a");
        n.vector = Some(vec![0.0, f64::NAN]);
        assert!(matches!(validate_node(&n), Err(PolypackError::InvalidArgument(_))));
    }

    #[test]
    fn rejects_invalid_activation() {
        let mut n = node("a");
        n.activation = Some(NodeActivation {
            score: 2.0,
            importance: 0.0,
            reinforcement_count: 0,
            last_meaningful_activation: 0,
        });
        assert!(matches!(validate_node(&n), Err(PolypackError::RangeOutOfBounds(_))));

        let mut n = node("a");
        n.activation = Some(NodeActivation {
            score: 0.0,
            importance: 0.0,
            reinforcement_count: 0,
            last_meaningful_activation: -1,
        });
        assert!(matches!(validate_node(&n), Err(PolypackError::RangeOutOfBounds(_))));
    }

    #[test]
    fn edge_identity_and_reserved_separator() {
        assert_eq!(edge_id("a", "REL", "b"), "a::REL::b");
        let mut e = Edge {
            id: edge_id("a::x", "REL", "b"),
            source: "a::x".into(),
            target: "b".into(),
            edge_type: "REL".into(),
            data: None,
            created_at: 1,
        };
        assert!(matches!(validate_edge(&e), Err(PolypackError::RangeOutOfBounds(_))));
        e.source = "a".into();
        e.id = edge_id("a", "REL", "b");
        assert!(validate_edge(&e).is_ok());
    }

    #[test]
    fn serde_round_trip_camel_case() {
        let n = Node {
            id: "a".into(),
            node_type: "doc".into(),
            data: serde_json::Map::new(),
            vector: Some(vec![0.1, 0.2]),
            inserted_at: 7,
            updated_at: 8,
            activation: None,
        };
        let s = serde_json::to_string(&n).unwrap();
        let back: Node = serde_json::from_str(&s).unwrap();
        assert_eq!(n, back);
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed["type"], json!("doc"));
        assert_eq!(parsed["insertedAt"], json!(7));
    }

    #[test]
    fn batch_validation_is_all_or_reject() {
        let mut batch = ChangeBatch {
            put_nodes: vec![node("ok")],
            ..Default::default()
        };
        assert!(validate_batch(&batch).is_ok());
        batch.put_nodes.push(node(""));
        assert!(matches!(validate_batch(&batch), Err(PolypackError::InvalidArgument(_))));
    }
}
