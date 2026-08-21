//! Snapshot and WAL codecs, byte-compatible with the TypeScript
//! `src/persistence/binary-format.ts` (version 1).
//!
//! Snapshot: `{ version: 1, nodes: [[id, node]...], edges: [[id, edge]...],
//! vectors: [[id, vector]...] }`.
//!
//! WAL: a sequence of frames, each a 4-byte big-endian length followed by a
//! MessagePack entry `{ kind, ...payload }`. Decoding stops at the first
//! truncated or invalid frame.

use crate::error::{PolypackError, Result};
use crate::model::{Edge, Node};
use crate::storage::msgpack::{decode, encode, Msg};
use crate::storage::wal::WalEntry;

const SNAPSHOT_VERSION: i64 = 1;

fn json_to_msg(value: &serde_json::Value) -> Msg {
    match value {
        serde_json::Value::Null => Msg::Nil,
        serde_json::Value::Bool(b) => Msg::Bool(*b),
        serde_json::Value::Number(n) => {
            if let Some(i) = n.as_i64() {
                Msg::Int(i)
            } else if let Some(u) = n.as_u64() {
                Msg::Int(u as i64)
            } else {
                Msg::Float(n.as_f64().unwrap_or(0.0))
            }
        }
        serde_json::Value::String(s) => Msg::Str(s.clone()),
        serde_json::Value::Array(items) => Msg::Array(items.iter().map(json_to_msg).collect()),
        serde_json::Value::Object(map) => {
            Msg::Map(map.iter().map(|(k, v)| (Msg::Str(k.clone()), json_to_msg(v))).collect())
        }
    }
}

fn msg_to_json(msg: &Msg) -> serde_json::Value {
    match msg {
        Msg::Nil => serde_json::Value::Null,
        Msg::Bool(b) => serde_json::Value::Bool(*b),
        Msg::Int(i) => serde_json::Value::Number((*i).into()),
        Msg::Float(f) => serde_json::Value::Number(serde_json::Number::from_f64(*f).unwrap_or(serde_json::Number::from(0))),
        Msg::Str(s) => serde_json::Value::String(s.clone()),
        Msg::Array(items) => serde_json::Value::Array(items.iter().map(msg_to_json).collect()),
        Msg::Map(entries) => {
            let mut map = serde_json::Map::new();
            for (k, v) in entries {
                if let Msg::Str(s) = k {
                    map.insert(s.clone(), msg_to_json(v));
                }
            }
            serde_json::Value::Object(map)
        }
    }
}

fn activation_to_msg(activation: &Option<crate::model::NodeActivation>) -> Msg {
    match activation {
        Some(a) => {
            let mut entries = vec![
                ("score", Msg::Float(a.score)),
                ("importance", Msg::Float(a.importance)),
                ("reinforcementCount", Msg::Int(a.reinforcement_count as i64)),
                ("lastMeaningfulActivation", Msg::Int(a.last_meaningful_activation)),
            ];
            if let Some(inhibition) = a.inhibition {
                entries.push(("inhibition", Msg::Float(inhibition)));
                entries.push(("lastInhibitedAt", Msg::Int(a.last_inhibited_at.unwrap_or(a.last_meaningful_activation))));
            }
            if let Some(context) = &a.context {
                let context_entries = context
                    .iter()
                    .map(|(key, entry)| {
                        (
                            Msg::Str(key.clone()),
                            Msg::map(vec![
                                ("score", Msg::Float(entry.score)),
                                ("lastMeaningfulActivation", Msg::Int(entry.last_meaningful_activation)),
                            ]),
                        )
                    })
                    .collect();
                entries.push(("context", Msg::Map(context_entries)));
            }
            Msg::map(entries)
        }
        // Match the TypeScript reference, which encodes `activation: undefined`
        // as an explicit nil key, so bytes stay identical across languages.
        None => Msg::Nil,
    }
}

fn msg_to_activation(msg: &Msg) -> Result<Option<crate::model::NodeActivation>> {
    let entries = match msg.get("activation") {
        Some(Msg::Map(map)) => map,
        Some(Msg::Nil) | None => return Ok(None),
        _ => return Err(PolypackError::CorruptData("activation must be a map or null".into())),
    };
    let find = |key: &str| {
        entries.iter().find(|(k, _)| matches!(k, Msg::Str(s) if s == key)).map(|(_, v)| v)
    };
    let get_f64 = |key: &str| -> Result<f64> {
        match find(key) {
            Some(Msg::Float(f)) => Ok(*f),
            Some(Msg::Int(i)) => Ok(*i as f64),
            _ => Err(PolypackError::CorruptData(format!("activation missing or invalid {key}"))),
        }
    };
    let score = get_f64("score")?;
    let importance = get_f64("importance")?;
    let reinforcement_count = match find("reinforcementCount") {
        Some(Msg::Int(i)) if *i >= 0 => *i as u64,
        _ => return Err(PolypackError::CorruptData("activation missing or invalid reinforcementCount".into())),
    };
    let last_meaningful_activation = match find("lastMeaningfulActivation") {
        Some(Msg::Int(i)) => *i,
        _ => return Err(PolypackError::CorruptData("activation missing or invalid lastMeaningfulActivation".into())),
    };
    let inhibition = match find("inhibition") {
        Some(Msg::Float(f)) => Some(*f),
        Some(Msg::Int(i)) => Some(*i as f64),
        _ => None,
    };
    let last_inhibited_at = match find("lastInhibitedAt") {
        Some(Msg::Int(i)) => Some(*i),
        _ => None,
    };
    let context = match find("context") {
        Some(Msg::Map(entries)) => {
            let mut context = std::collections::HashMap::new();
            for (k, v) in entries {
                let Msg::Str(key) = k else { continue };
                let Msg::Map(fields) = v else { continue };
                let find_field = |name: &str| fields.iter().find(|(k, _)| matches!(k, Msg::Str(s) if s == name)).map(|(_, v)| v);
                let score = match find_field("score") {
                    Some(Msg::Float(f)) => *f,
                    Some(Msg::Int(i)) => *i as f64,
                    _ => continue,
                };
                let anchor = match find_field("lastMeaningfulActivation") {
                    Some(Msg::Int(i)) => *i,
                    _ => continue,
                };
                context.insert(key.clone(), crate::model::ContextActivation { score, last_meaningful_activation: anchor });
            }
            Some(context)
        }
        _ => None,
    };
    Ok(Some(crate::model::NodeActivation {
        score,
        importance,
        reinforcement_count,
        last_meaningful_activation,
        inhibition,
        last_inhibited_at,
        context,
    }))
}

fn node_to_msg(node: &Node) -> Msg {
    let vector = node
        .vector
        .as_ref()
        .map(|v| Msg::Array(v.iter().map(|x| Msg::Float(*x)).collect()))
        .unwrap_or(Msg::Nil);
    let mut fields = vec![
        ("id", Msg::Str(node.id.clone())),
        ("type", Msg::Str(node.node_type.clone())),
        ("data", json_to_msg(&serde_json::Value::Object(node.data.clone()))),
        ("vector", vector),
        ("insertedAt", Msg::Int(node.inserted_at)),
        ("updatedAt", Msg::Int(node.updated_at)),
        ("activation", activation_to_msg(&node.activation)),
    ];
    if node.revision != 0 {
        fields.push(("revision", Msg::Int(node.revision as i64)));
    }
    Msg::map(fields)
}

fn msg_to_node(msg: &Msg) -> Result<Node> {
    let id = msg
        .get_str("id")
        .ok_or_else(|| PolypackError::CorruptData("node missing id".into()))?
        .to_string();
    let node_type = msg
        .get_str("type")
        .ok_or_else(|| PolypackError::CorruptData("node missing type".into()))?
        .to_string();
    let data = match msg.get("data") {
        Some(Msg::Map(_)) | Some(Msg::Nil) | None => {
            msg_to_json(msg.get("data").unwrap_or(&Msg::Nil)).as_object().cloned().unwrap_or_default()
        }
        _ => return Err(PolypackError::CorruptData("node data must be a map".into())),
    };
    let vector = match msg.get("vector") {
        Some(Msg::Array(items)) => Some(msg_vec_f64(items)?),
        Some(Msg::Nil) | None => None,
        _ => return Err(PolypackError::CorruptData("node vector must be an array or null".into())),
    };
    let inserted_at = msg_int_field(msg, "insertedAt")?;
    let updated_at = msg_int_field(msg, "updatedAt")?;
    let revision = msg_u64_field_default(msg, "revision", 0)?;
    let activation = msg_to_activation(msg)?;
    Ok(Node {
        id,
        node_type,
        data,
        vector,
        inserted_at,
        updated_at,
        revision,
        activation,
    })
}

fn msg_vec_f64(items: &[Msg]) -> Result<Vec<f64>> {
    items
        .iter()
        .map(|m| match m {
            Msg::Int(i) => Ok(*i as f64),
            Msg::Float(f) => Ok(*f),
            _ => Err(PolypackError::CorruptData("vector element must be numeric".into())),
        })
        .collect()
}

fn msg_int_field(msg: &Msg, key: &str) -> Result<i64> {
    match msg.get(key) {
        Some(Msg::Int(i)) => Ok(*i),
        Some(Msg::Float(f)) if f.fract() == 0.0 => Ok(*f as i64),
        Some(_) | None => Err(PolypackError::CorruptData(format!("missing or invalid {key}"))),
    }
}

fn msg_u64_field_default(msg: &Msg, key: &str, default: u64) -> Result<u64> {
    match msg.get(key) {
        None | Some(Msg::Nil) => Ok(default),
        Some(Msg::Int(i)) if *i >= 0 => Ok(*i as u64),
        Some(Msg::Float(f)) if f.is_finite() && *f >= 0.0 && f.fract() == 0.0 => Ok(*f as u64),
        Some(_) => Err(PolypackError::CorruptData(format!("invalid {key}"))),
    }
}

fn edge_to_msg(edge: &Edge) -> Msg {
    let data = edge
        .data
        .as_ref()
        .map(|d| json_to_msg(&serde_json::Value::Object(d.clone())))
        .unwrap_or(Msg::Nil);
    let mut fields = vec![
        ("id", Msg::Str(edge.id.clone())),
        ("source", Msg::Str(edge.source.clone())),
        ("target", Msg::Str(edge.target.clone())),
        ("type", Msg::Str(edge.edge_type.clone())),
        ("data", data),
        ("createdAt", Msg::Int(edge.created_at)),
    ];
    if edge.revision != 0 {
        fields.push(("revision", Msg::Int(edge.revision as i64)));
    }
    Msg::map(fields)
}

fn msg_to_edge(msg: &Msg) -> Result<Edge> {
    let id = msg
        .get_str("id")
        .ok_or_else(|| PolypackError::CorruptData("edge missing id".into()))?
        .to_string();
    let source = msg
        .get_str("source")
        .ok_or_else(|| PolypackError::CorruptData("edge missing source".into()))?
        .to_string();
    let target = msg
        .get_str("target")
        .ok_or_else(|| PolypackError::CorruptData("edge missing target".into()))?
        .to_string();
    let edge_type = msg
        .get_str("type")
        .ok_or_else(|| PolypackError::CorruptData("edge missing type".into()))?
        .to_string();
    let data = match msg.get("data") {
        Some(Msg::Map(_)) | Some(Msg::Nil) | None => {
            msg_to_json(msg.get("data").unwrap_or(&Msg::Nil)).as_object().cloned()
        }
        _ => return Err(PolypackError::CorruptData("edge data must be a map or null".into())),
    };
    let created_at = msg_int_field(msg, "createdAt")?;
    let revision = msg_u64_field_default(msg, "revision", 0)?;
    Ok(Edge {
        id,
        source,
        target,
        edge_type,
        data,
        created_at,
        revision,
    })
}

pub struct SnapshotData {
    pub nodes: Vec<(String, Node)>,
    pub edges: Vec<(String, Edge)>,
    pub vectors: Vec<(String, Vec<f64>)>,
}

pub fn encode_snapshot(
    nodes: &[(String, Node)],
    edges: &[(String, Edge)],
    vectors: &[(String, Vec<f64>)],
) -> Vec<u8> {
    let nodes_msg = Msg::Array(
        nodes
            .iter()
            .map(|(id, n)| Msg::Array(vec![Msg::Str(id.clone()), node_to_msg(n)]))
            .collect(),
    );
    let edges_msg = Msg::Array(
        edges
            .iter()
            .map(|(id, e)| Msg::Array(vec![Msg::Str(id.clone()), edge_to_msg(e)]))
            .collect(),
    );
    let vectors_msg = Msg::Array(
        vectors
            .iter()
            .map(|(id, v)| {
                Msg::Array(vec![
                    Msg::Str(id.clone()),
                    Msg::Array(v.iter().map(|x| Msg::Float(*x)).collect()),
                ])
            })
            .collect(),
    );
    let snap = Msg::map(vec![
        ("version", Msg::Int(SNAPSHOT_VERSION)),
        ("nodes", nodes_msg),
        ("edges", edges_msg),
        ("vectors", vectors_msg),
    ]);
    let mut out = Vec::new();
    encode(&snap, &mut out);
    out
}

pub fn decode_snapshot(data: &[u8]) -> Result<SnapshotData> {
    let msg = decode(data)?;
    let version = msg_int_field(&msg, "version")?;
    if version != SNAPSHOT_VERSION {
        return Err(PolypackError::FormatVersion(version as u64));
    }
    let nodes = match msg.get("nodes") {
        Some(Msg::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Msg::Array(pair) if pair.len() == 2 => {
                        let id = msg_str(&pair[0])?;
                        out.push((id, msg_to_node(&pair[1])?));
                    }
                    _ => return Err(PolypackError::CorruptData("node entry must be [id, node]".into())),
                }
            }
            out
        }
        _ => Vec::new(),
    };
    let edges = match msg.get("edges") {
        Some(Msg::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Msg::Array(pair) if pair.len() == 2 => {
                        let id = msg_str(&pair[0])?;
                        out.push((id, msg_to_edge(&pair[1])?));
                    }
                    _ => return Err(PolypackError::CorruptData("edge entry must be [id, edge]".into())),
                }
            }
            out
        }
        _ => Vec::new(),
    };
    let vectors = match msg.get("vectors") {
        Some(Msg::Array(items)) => {
            let mut out = Vec::with_capacity(items.len());
            for item in items {
                match item {
                    Msg::Array(pair) if pair.len() == 2 => {
                        let id = msg_str(&pair[0])?;
                        let vector = match &pair[1] {
                            Msg::Array(v) => msg_vec_f64(v)?,
                            _ => return Err(PolypackError::CorruptData("vector entry must be [id, array]".into())),
                        };
                        out.push((id, vector));
                    }
                    _ => return Err(PolypackError::CorruptData("vector entry must be [id, array]".into())),
                }
            }
            out
        }
        _ => Vec::new(),
    };
    Ok(SnapshotData { nodes, edges, vectors })
}

fn msg_str(msg: &Msg) -> Result<String> {
    match msg {
        Msg::Str(s) => Ok(s.clone()),
        _ => Err(PolypackError::CorruptData("expected a string".into())),
    }
}

pub fn encode_wal(entries: &[WalEntry]) -> Vec<u8> {
    let mut out = Vec::new();
    for entry in entries {
        let body = entry_to_msg(entry);
        let mut frame = Vec::new();
        encode(&body, &mut frame);
        out.extend_from_slice(&(frame.len() as u32).to_be_bytes());
        out.extend_from_slice(&frame);
    }
    out
}

fn entry_to_msg(entry: &WalEntry) -> Msg {
    match entry {
        WalEntry::PutNode(node) => Msg::map(vec![("kind", Msg::str("putNode")), ("node", node_to_msg(node))]),
        WalEntry::DeleteNode(id) => Msg::map(vec![("kind", Msg::str("deleteNode")), ("id", Msg::Str(id.clone()))]),
        WalEntry::PutEdge(edge) => Msg::map(vec![("kind", Msg::str("putEdge")), ("edge", edge_to_msg(edge))]),
        WalEntry::DeleteEdge(id) => Msg::map(vec![("kind", Msg::str("deleteEdge")), ("id", Msg::Str(id.clone()))]),
        WalEntry::PutVector { id, vector } => Msg::map(vec![
            ("kind", Msg::str("putVector")),
            ("id", Msg::Str(id.clone())),
            ("vector", Msg::Array(vector.iter().map(|x| Msg::Float(*x)).collect())),
        ]),
        WalEntry::DeleteVector(id) => Msg::map(vec![("kind", Msg::str("deleteVector")), ("id", Msg::Str(id.clone()))]),
        WalEntry::ClearAll => Msg::map(vec![("kind", Msg::str("clearAll"))]),
    }
}

/// Decode complete WAL frames, stopping at the first truncated or invalid
/// frame. The trailing partial frame (crash mid-append) is discarded.
pub fn decode_wal(data: &[u8]) -> Vec<WalEntry> {
    let mut entries = Vec::new();
    let mut offset = 0;
    while offset + 4 <= data.len() {
        let len = u32::from_be_bytes(data[offset..offset + 4].try_into().unwrap()) as usize;
        offset += 4;
        if offset + len > data.len() {
            break;
        }
        match decode(&data[offset..offset + len]) {
            Ok(msg) => match msg_to_entry(&msg) {
                Ok(entry) => entries.push(entry),
                Err(_) => break,
            },
            Err(_) => break,
        }
        offset += len;
    }
    entries
}

fn msg_to_entry(msg: &Msg) -> Result<WalEntry> {
    let kind = msg
        .get_str("kind")
        .ok_or_else(|| PolypackError::CorruptData("wal entry missing kind".into()))?;
    match kind {
        "putNode" => {
            let node = msg
                .get("node")
                .ok_or_else(|| PolypackError::CorruptData("putNode missing node".into()))?;
            Ok(WalEntry::PutNode(msg_to_node(node)?))
        }
        "deleteNode" => Ok(WalEntry::DeleteNode(msg_str_field(msg, "id")?)),
        "putEdge" => {
            let edge = msg
                .get("edge")
                .ok_or_else(|| PolypackError::CorruptData("putEdge missing edge".into()))?;
            Ok(WalEntry::PutEdge(msg_to_edge(edge)?))
        }
        "deleteEdge" => Ok(WalEntry::DeleteEdge(msg_str_field(msg, "id")?)),
        "putVector" => {
            let id = msg_str_field(msg, "id")?;
            let vector = match msg.get("vector") {
                Some(Msg::Array(items)) => msg_vec_f64(items)?,
                _ => return Err(PolypackError::CorruptData("putVector missing vector".into())),
            };
            Ok(WalEntry::PutVector { id, vector })
        }
        "deleteVector" => Ok(WalEntry::DeleteVector(msg_str_field(msg, "id")?)),
        "clearAll" => Ok(WalEntry::ClearAll),
        other => Err(PolypackError::CorruptData(format!("unknown wal kind {other}"))),
    }
}

fn msg_str_field(msg: &Msg, key: &str) -> Result<String> {
    match msg.get(key) {
        Some(Msg::Str(s)) => Ok(s.clone()),
        _ => Err(PolypackError::CorruptData(format!("wal entry missing {key}"))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::storage::wal::WalEntry;

    fn node(id: &str) -> Node {
        Node {
            id: id.into(),
            node_type: "doc".into(),
            data: serde_json::Map::new(),
            vector: Some(vec![0.1, 0.2, 0.3]),
            inserted_at: 7,
            updated_at: 8,
            revision: 0,
            activation: None,
        }
    }

    #[test]
    fn snapshot_bytes_match_javascript_reference() {
        // Captured from TS encodeSnapshot() on the same data.
        let expected = "84a776657273696f6e01a56e6f6465739192a26e3187a26964a26e31a474797065a3646f63a46461746182a57469746c65a548656c6c6fa573636f726503a6766563746f7293cb3fb999999999999acb3fc999999999999acb3fd3333333333333aa696e736572746564417407a975706461746564417408aa61637469766174696f6ec0a565646765739192a9613a3a52454c3a3a6286a26964a9613a3a52454c3a3a62a6736f75726365a161a6746172676574a162a474797065a352454ca464617461c0a963726561746564417409a7766563746f72739192a2763192cb3fe0000000000000cb3fd0000000000000";
        let mut n = node("n1");
        n.data = serde_json::json!({ "title": "Hello", "score": 3 }).as_object().unwrap().clone();
        let edge = Edge {
            id: "a::REL::b".into(),
            source: "a".into(),
            target: "b".into(),
            edge_type: "REL".into(),
            data: None,
            created_at: 9,
            revision: 0,
        };
        let bytes = encode_snapshot(
            &[("n1".to_string(), n)],
            &[("a::REL::b".to_string(), edge)],
            &[("v1".to_string(), vec![0.5, 0.25])],
        );
        assert_eq!(hex(&bytes), expected);
    }

    #[test]
    fn snapshot_with_activation_bytes_match_javascript_reference() {
        // Captured from TS encodeSnapshot() with an activation payload.
        let expected = "84a776657273696f6e01a56e6f6465739192a26e3187a26964a26e31a474797065a3646f63a46461746180a6766563746f72c0aa696e736572746564417401a975706461746564417401aa61637469766174696f6e84a573636f7265cb3fe0000000000000aa696d706f7274616e6365cb3fb999999999999ab27265696e666f7263656d656e74436f756e7402b86c6173744d65616e696e6766756c41637469766174696f6e01a5656467657390a7766563746f72739192a2763192cb3fe0000000000000cb3fd0000000000000";
        let mut n = node("n1");
        n.data = serde_json::Map::new();
        n.vector = None;
        n.inserted_at = 1;
        n.updated_at = 1;
        n.activation = Some(crate::model::NodeActivation {
            score: 0.5,
            importance: 0.1,
            reinforcement_count: 2,
            last_meaningful_activation: 1,
            ..Default::default()
        });
        let bytes = encode_snapshot(&[("n1".to_string(), n)], &[], &[("v1".to_string(), vec![0.5, 0.25])]);
        assert_eq!(hex(&bytes), expected);
    }

    #[test]
    fn wal_bytes_match_javascript_reference() {
        let expected = "0000005482a46b696e64a77075744e6f6465a46e6f646587a26964a26e31a474797065a3646f63a46461746180a6766563746f72c0aa696e736572746564417401a975706461746564417401aa61637469766174696f6ec00000001782a46b696e64aa64656c6574654e6f6465a26964a26e32";
        let mut n = node("n1");
        n.vector = None;
        n.data = serde_json::Map::new();
        n.inserted_at = 1;
        n.updated_at = 1;
        let bytes = encode_wal(&[
            WalEntry::PutNode(n),
            WalEntry::DeleteNode("n2".into()),
        ]);
        assert_eq!(hex(&bytes), expected);
    }

    #[test]
    fn snapshot_round_trips() {
        let mut n = node("n1");
        n.data = serde_json::json!({ "title": "Hello", "score": 3 }).as_object().unwrap().clone();
        let bytes = encode_snapshot(
            &[("n1".to_string(), n.clone())],
            &[],
            &[("v1".to_string(), vec![0.5])],
        );
        let decoded = decode_snapshot(&bytes).unwrap();
        assert_eq!(decoded.nodes[0].1, n);
        assert_eq!(decoded.vectors[0].1, vec![0.5]);
    }

    #[test]
    fn activation_snapshot_round_trips_and_absent_defaults_to_none() {
        let mut n = node("n1");
        n.vector = None;
        n.activation = Some(crate::model::NodeActivation {
            score: 0.7,
            importance: 0.2,
            reinforcement_count: 3,
            last_meaningful_activation: 42,
            ..Default::default()
        });
        let bytes = encode_snapshot(&[("n1".to_string(), n.clone())], &[], &[]);
        let decoded = decode_snapshot(&bytes).unwrap();
        assert_eq!(decoded.nodes[0].1, n);

        // A legacy snapshot without the activation key decodes to None.
        let legacy_bytes = encode_snapshot(&[("n2".to_string(), node("n2"))], &[], &[]);
        let decoded = decode_snapshot(&legacy_bytes).unwrap();
        assert_eq!(decoded.nodes[0].1.activation, None);
    }

    #[test]
    fn wal_decode_stops_at_truncated_tail() {
        let mut n = node("n1");
        n.vector = None;
        let full = encode_wal(&[WalEntry::PutNode(n.clone()), WalEntry::PutNode(n)]);
        let truncated = &full[..full.len() - 3];
        let entries = decode_wal(truncated);
        assert_eq!(entries.len(), 1);
        assert!(matches!(entries[0], WalEntry::PutNode(_)));
    }

    #[test]
    fn snapshot_rejects_unknown_version() {
        let bytes = encode_snapshot(&[], &[], &[]);
        let mut msg = decode(&bytes).unwrap();
        if let Msg::Map(entries) = &mut msg {
            for (k, v) in entries {
                if matches!(k, Msg::Str(s) if s == "version") {
                    *v = Msg::Int(99);
                }
            }
        }
        let mut out = Vec::new();
        encode(&msg, &mut out);
        assert!(matches!(decode_snapshot(&out), Err(PolypackError::FormatVersion(99))));
    }

    fn hex(b: &[u8]) -> String {
        b.iter().map(|x| format!("{:02x}", x)).collect()
    }
}
