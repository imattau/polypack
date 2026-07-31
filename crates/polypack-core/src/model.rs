//! Core data envelopes matching `specification/data-model.md` and
//! `specification/change-batch.schema.json`.

use crate::error::{PolypackError, Result};
use serde::{Deserialize, Serialize};

fn is_finite_vec(v: &[f64]) -> bool {
    v.iter().all(|x| x.is_finite())
}

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
}

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

#[derive(Serialize, Deserialize, Clone, Debug, PartialEq)]
#[serde(rename_all = "camelCase")]
pub struct VectorEntry {
    pub id: String,
    pub vector: Vec<f64>,
}

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
