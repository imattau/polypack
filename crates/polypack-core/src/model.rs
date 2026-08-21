//! Core data envelopes matching `specification/data-model.md` and
//! `specification/change-batch.schema.json`.

use crate::error::{PolypackError, Result};
use serde::{Deserialize, Serialize};

fn is_finite_vec(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

/// A node's decayed relevance within one named context (e.g. a project, user, or task).
#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct ContextActivation {
    /// Context-scoped activation. Clamped to [0, 1].
    pub score: f64,
    /// Epoch-ms anchor for this context's decay.
    pub last_meaningful_activation: i64,
}

/// Durable activation state for a node (see `crate::activation`). All fields
/// are persisted and replicated; transient, runtime-only attention is held by
/// the `ActivationEngine` in `polypack-graph` and never serialized. Decay is a
/// pure function of elapsed time anchored at `last_meaningful_activation`, so
/// replicas with the same stored state compute identical current scores.
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
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
    /// Suppression, subtracted from `score` at read time only (never inside
    /// relational spreading). Clamped to [0, 1]. Absent is equivalent to 0.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inhibition: Option<f64>,
    /// Epoch-ms anchor for `inhibition`'s decay. Absent iff `inhibition` is absent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub last_inhibited_at: Option<i64>,
    /// Per-context activation, additional to (not a replacement for) the
    /// global `score` above.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub context: Option<std::collections::HashMap<String, ContextActivation>>,
}

/// A typed property-graph node. Serializes as camelCase JSON matching the
/// TypeScript `PolyNode`/`SerializedNode` shape (`type` on the wire, `node_type` in Rust).
#[derive(Serialize, Deserialize, Clone, Debug, Default, PartialEq)]
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
    /// Monotonically increasing record revision used for optimistic writes.
    /// Legacy snapshots that do not carry a revision decode as zero.
    #[serde(default)]
    pub revision: u64,
    /// Durable, syncable activation. Optional — absent until first reinforced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub activation: Option<NodeActivation>,
    /// Overrides `NodeTypeDefinition.memory_class` for this node only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub memory_class: Option<MemoryClass>,
    /// Confidence this node's content is currently believed true. Clamped to [0, 1]. Absent means "not tracked."
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub confidence: Option<f64>,
    /// Where this node's content came from (free-form provenance label).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source: Option<String>,
    /// When the underlying fact was actually observed/asserted. Epoch ms.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed_at: Option<i64>,
    /// Node ids this node was derived/consolidated from. Soft references.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub derived_from: Option<Vec<String>>,
    /// Node id this node supersedes (contradiction axis). A soft reference.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub supersedes: Option<String>,
    /// Node ids this node is in direct conflict with. Soft references.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub contradicts: Option<Vec<String>>,
}

/// Memory class: which decay curve a node's activation follows (see
/// `crate::activation`'s class-half-life resolution). Episodic memories decay
/// fastest by default, entity facts slowest.
#[derive(Serialize, Deserialize, Clone, Copy, Debug, PartialEq, Eq, Hash)]
#[serde(rename_all = "lowercase")]
pub enum MemoryClass {
    Episodic,
    Semantic,
    Procedural,
    Entity,
}

/// A directed property-graph edge. `id` is an independent durable identity;
/// [`edge_id`] remains available as a deterministic legacy helper for callers
/// that want one ID per source/type/target triple.
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
    /// Monotonically increasing record revision used for optimistic writes.
    /// Legacy snapshots that do not carry a revision decode as zero.
    #[serde(default)]
    pub revision: u64,
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
    validate_provenance(n)?;
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
    if let Some(inhibition) = a.inhibition {
        if !inhibition.is_finite() || !(0.0..=1.0).contains(&inhibition) {
            return Err(PolypackError::RangeOutOfBounds(
                "activation.inhibition must be a finite number in [0, 1]".into(),
            ));
        }
        if a.last_inhibited_at.map(|t| t < 0).unwrap_or(true) {
            return Err(PolypackError::RangeOutOfBounds(
                "activation.lastInhibitedAt must be a finite non-negative number when inhibition is set".into(),
            ));
        }
    }
    if let Some(context) = &a.context {
        for (key, entry) in context {
            if !entry.score.is_finite() || !(0.0..=1.0).contains(&entry.score) {
                return Err(PolypackError::RangeOutOfBounds(format!(
                    "activation.context[{key}].score must be a finite number in [0, 1]"
                )));
            }
            if entry.last_meaningful_activation < 0 {
                return Err(PolypackError::RangeOutOfBounds(format!(
                    "activation.context[{key}].lastMeaningfulActivation must be a finite non-negative number"
                )));
            }
        }
    }
    Ok(())
}

/// Validate a node's memory-class and confidence/provenance fields. These are
/// node-level metadata (not activation state), so this is a sibling to
/// `validate_activation`, not an extension of it. `derived_from`/`supersedes`/
/// `contradicts` are soft references — the referenced ids are not required to
/// exist.
pub fn validate_provenance(n: &Node) -> Result<()> {
    if let Some(confidence) = n.confidence {
        if !confidence.is_finite() || !(0.0..=1.0).contains(&confidence) {
            return Err(PolypackError::RangeOutOfBounds(
                "confidence must be a finite number in [0, 1]".into(),
            ));
        }
    }
    if let Some(observed_at) = n.observed_at {
        if observed_at < 0 {
            return Err(PolypackError::RangeOutOfBounds(
                "observedAt must be a finite non-negative number".into(),
            ));
        }
    }
    if let Some(source) = &n.source {
        if source.is_empty() {
            return Err(PolypackError::InvalidArgument("source must not be empty".into()));
        }
    }
    if let Some(supersedes) = &n.supersedes {
        if supersedes.is_empty() {
            return Err(PolypackError::InvalidArgument("supersedes must not be empty".into()));
        }
    }
    for (field, values) in [("derivedFrom", &n.derived_from), ("contradicts", &n.contradicts)] {
        if let Some(values) = values {
            if values.iter().any(|id| id.is_empty()) {
                return Err(PolypackError::InvalidArgument(format!(
                    "{field} must contain only non-empty strings"
                )));
            }
        }
    }
    Ok(())
}

/// Validate a single edge envelope. Edge identity is independent of its
/// source/type/target tuple, so parallel edges are valid.
pub fn validate_edge(e: &Edge) -> Result<()> {
    if e.source.is_empty() || e.edge_type.is_empty() || e.target.is_empty() {
        return Err(PolypackError::InvalidArgument(
            "edge source, type, and target must not be empty".into(),
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
            revision: 0,
            activation: None,
            ..Default::default()
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
            ..Default::default()
        });
        assert!(matches!(validate_node(&n), Err(PolypackError::RangeOutOfBounds(_))));

        let mut n = node("a");
        n.activation = Some(NodeActivation {
            score: 0.0,
            importance: 0.0,
            reinforcement_count: 0,
            last_meaningful_activation: -1,
            ..Default::default()
        });
        assert!(matches!(validate_node(&n), Err(PolypackError::RangeOutOfBounds(_))));
    }

    #[test]
    fn edge_identity_and_reserved_separator() {
        assert_eq!(edge_id("a", "REL", "b"), "a::REL::b");
        let e = Edge {
            id: edge_id("a::x", "REL", "b"),
            source: "a::x".into(),
            target: "b".into(),
            edge_type: "REL".into(),
            data: None,
            created_at: 1,
            revision: 0,
        };
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
            revision: 12,
            activation: None,
            ..Default::default()
        };
        let s = serde_json::to_string(&n).unwrap();
        let back: Node = serde_json::from_str(&s).unwrap();
        assert_eq!(n, back);
        let parsed: serde_json::Value = serde_json::from_str(&s).unwrap();
        assert_eq!(parsed["type"], json!("doc"));
        assert_eq!(parsed["insertedAt"], json!(7));
        assert_eq!(parsed["revision"], json!(12));
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

    #[test]
    fn rejects_invalid_provenance_fields() {
        let mut n = node("a");
        n.confidence = Some(1.5);
        assert!(matches!(validate_node(&n), Err(PolypackError::RangeOutOfBounds(_))));

        let mut n = node("a");
        n.observed_at = Some(-1);
        assert!(matches!(validate_node(&n), Err(PolypackError::RangeOutOfBounds(_))));

        let mut n = node("a");
        n.source = Some(String::new());
        assert!(matches!(validate_node(&n), Err(PolypackError::InvalidArgument(_))));

        let mut n = node("a");
        n.supersedes = Some(String::new());
        assert!(matches!(validate_node(&n), Err(PolypackError::InvalidArgument(_))));

        let mut n = node("a");
        n.derived_from = Some(vec!["b".into(), String::new()]);
        assert!(matches!(validate_node(&n), Err(PolypackError::InvalidArgument(_))));
    }

    #[test]
    fn accepts_valid_provenance_fields() {
        let mut n = node("a");
        n.memory_class = Some(MemoryClass::Semantic);
        n.confidence = Some(0.9);
        n.source = Some("user".into());
        n.observed_at = Some(500);
        n.derived_from = Some(vec!["e1".into(), "e2".into()]);
        n.supersedes = Some("a-old".into());
        n.contradicts = Some(vec!["a-conflicting".into()]);
        assert!(validate_node(&n).is_ok());
        let json = serde_json::to_value(&n).unwrap();
        assert_eq!(json["memoryClass"], json!("semantic"));
        let back: Node = serde_json::from_value(json).unwrap();
        assert_eq!(back, n);
    }
}
