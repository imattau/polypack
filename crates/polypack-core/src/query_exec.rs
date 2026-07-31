//! Query-plan executor.
//!
//! Runs a serialisable [`QueryPlan`] over a snapshot of nodes and edges,
//! mirroring the TypeScript `GraphQuery` pipeline order: source selection ->
//! attribute/edge filters -> joins -> traversal -> order -> similarity ->
//! offset/limit. Similarity is exact cosine by default; `engine: "hnsw"`
//! ranks via a caller-supplied approximate index.

use crate::error::{PolypackError, Result};
use crate::hnsw::HnswIndex;
use crate::model::{Edge, Node};
use crate::query::{AttributeFilter, Direction, OrderDirection, QueryPlan, SimilarityEngine};
use crate::vector::cosine;
use serde_json::Value;
use std::collections::{HashMap, HashSet};

pub struct GraphSnapshot {
    pub nodes: Vec<Node>,
    pub edges: Vec<Edge>,
    by_id: HashMap<String, usize>,
    outgoing: HashMap<String, Vec<usize>>,
    incoming: HashMap<String, Vec<usize>>,
}

impl GraphSnapshot {
    pub fn new(nodes: Vec<Node>, edges: Vec<Edge>) -> Self {
        let mut by_id = HashMap::with_capacity(nodes.len());
        for (i, n) in nodes.iter().enumerate() {
            by_id.insert(n.id.clone(), i);
        }
        let mut outgoing: HashMap<String, Vec<usize>> = HashMap::new();
        let mut incoming: HashMap<String, Vec<usize>> = HashMap::new();
        for (i, e) in edges.iter().enumerate() {
            outgoing.entry(e.source.clone()).or_default().push(i);
            incoming.entry(e.target.clone()).or_default().push(i);
        }
        GraphSnapshot {
            nodes,
            edges,
            by_id,
            outgoing,
            incoming,
        }
    }

    fn get(&self, id: &str) -> Option<&Node> {
        self.by_id.get(id).map(|&i| &self.nodes[i])
    }

    fn edge_of(&self, e: &Edge, edge_type: &str) -> bool {
        e.edge_type == edge_type
    }
}

fn json_eq(a: &Value, b: &Value) -> bool {
    match (a, b) {
        (Value::Number(x), Value::Number(y)) => match (x.as_f64(), y.as_f64()) {
            (Some(x), Some(y)) => x == y,
            _ => false,
        },
        _ => a == b,
    }
}

fn attr_num(node: &Node, field: &str) -> f64 {
    node.data
        .get(field)
        .and_then(|v| v.as_f64())
        .unwrap_or(0.0)
}

/// Source selection, mirroring `GraphQuery.getSourceNodes`.
fn source_ids(snap: &GraphSnapshot, plan: &QueryPlan) -> Vec<String> {
    if let Some(ef) = &plan.edge_filter {
        if let Some(source) = &ef.source {
            if let Some(edge_ids) = snap.outgoing.get(source) {
                let mut seen = HashSet::new();
                let mut ids = Vec::new();
                for &ei in edge_ids {
                    let e = &snap.edges[ei];
                    if snap.edge_of(e, &ef.edge_type) && seen.insert(e.target.clone()) {
                        ids.push(e.target.clone());
                    }
                }
                return ids;
            }
            return Vec::new();
        }
        if let Some(target) = &ef.target {
            if let Some(edge_ids) = snap.incoming.get(target) {
                let mut seen = HashSet::new();
                let mut ids = Vec::new();
                for &ei in edge_ids {
                    let e = &snap.edges[ei];
                    if snap.edge_of(e, &ef.edge_type) && seen.insert(e.source.clone()) {
                        ids.push(e.source.clone());
                    }
                }
                return ids;
            }
            return Vec::new();
        }
    }
    match &plan.node_types {
        Some(types) => snap
            .nodes
            .iter()
            .filter(|n| types.contains(&n.node_type))
            .map(|n| n.id.clone())
            .collect(),
        None => snap.nodes.iter().map(|n| n.id.clone()).collect(),
    }
}

fn match_node(snap: &GraphSnapshot, node: &Node, plan: &QueryPlan) -> bool {
    if let Some(types) = &plan.node_types {
        if !types.contains(&node.node_type) {
            return false;
        }
    }
    if let Some(attrs) = &plan.attributes {
        for attr in attrs {
            match attr {
                AttributeFilter::Eq { field, value } => {
                    let actual = if field == "type" {
                        Value::String(node.node_type.clone())
                    } else {
                        node.data.get(field).cloned().unwrap_or(Value::Null)
                    };
                    if !json_eq(&actual, value) {
                        return false;
                    }
                }
                AttributeFilter::Range { field, above, below } => {
                    let val = match node.data.get(field).and_then(|v| v.as_f64()) {
                        Some(v) => v,
                        None => return false,
                    };
                    if let Some(above) = above {
                        if val <= *above {
                            return false;
                        }
                    }
                    if let Some(below) = below {
                        if val >= *below {
                            return false;
                        }
                    }
                }
            }
        }
    }
    if let Some(ef) = &plan.edge_filter {
        if let Some(edge_ids) = snap.outgoing.get(&node.id) {
            let mut matched = false;
            for &ei in edge_ids {
                let e = &snap.edges[ei];
                if snap.edge_of(e, &ef.edge_type)
                    && (ef.target.as_deref().map(|t| t == e.target).unwrap_or(true))
                {
                    matched = true;
                    break;
                }
            }
            if !matched {
                return false;
            }
        } else if ef.source.is_none() {
            return false;
        }
        if let Some(source) = &ef.source {
            if let Some(edge_ids) = snap.outgoing.get(source) {
                let mut matched = false;
                for &ei in edge_ids {
                    let e = &snap.edges[ei];
                    if snap.edge_of(e, &ef.edge_type) && e.target == node.id {
                        matched = true;
                        break;
                    }
                }
                if !matched {
                    return false;
                }
            }
        }
    }
    if let Some(joins) = &plan.joins {
        for j in joins {
            if !joined(snap, node, &j.edge_type, j.direction) {
                return false;
            }
        }
    }
    true
}

fn joined(snap: &GraphSnapshot, node: &Node, edge_type: &str, direction: Direction) -> bool {
    match direction {
        Direction::Out => snap
            .outgoing
            .get(&node.id)
            .map(|ids| ids.iter().any(|&ei| snap.edge_of(&snap.edges[ei], edge_type)))
            .unwrap_or(false),
        Direction::In => snap
            .incoming
            .get(&node.id)
            .map(|ids| ids.iter().any(|&ei| snap.edge_of(&snap.edges[ei], edge_type)))
            .unwrap_or(false),
    }
}

/// BFS expansion from seeds; returns ids in discovery order, seeds first.
fn bfs(snap: &GraphSnapshot, seeds: &[String], edge_type: &str, depth: usize, direction: Direction) -> Vec<String> {
    let mut visited: Vec<String> = seeds.to_vec();
    let mut seen: HashSet<String> = seeds.iter().cloned().collect();
    let mut frontier: Vec<String> = seeds.to_vec();
    for _ in 0..depth {
        if frontier.is_empty() {
            break;
        }
        let mut next = Vec::new();
        for id in &frontier {
            let edge_ids: &Vec<usize> = match direction {
                Direction::Out => match snap.outgoing.get(id) {
                    Some(e) => e,
                    None => continue,
                },
                Direction::In => match snap.incoming.get(id) {
                    Some(e) => e,
                    None => continue,
                },
            };
            for &ei in edge_ids {
                let e = &snap.edges[ei];
                if !snap.edge_of(e, edge_type) {
                    continue;
                }
                let neighbor = match direction {
                    Direction::Out => &e.target,
                    Direction::In => &e.source,
                };
                if seen.insert(neighbor.clone()) {
                    visited.push(neighbor.clone());
                    next.push(neighbor.clone());
                }
            }
        }
        frontier = next;
    }
    visited
}

/// Execute a query plan, returning the ordered ids of the matching nodes.
pub fn execute(snap: &GraphSnapshot, plan: &QueryPlan, hnsw: Option<&HnswIndex>) -> Result<Vec<String>> {
    let mut ids = source_ids(snap, plan);
    ids.retain(|id| match snap.get(id) {
        Some(node) => match_node(snap, node, plan),
        None => false,
    });
    if ids.is_empty() {
        return Ok(Vec::new());
    }

    // Traversal: sequential BFS steps, discovery order, seeds included.
    for step in plan.traversal.as_deref().unwrap_or(&[]) {
        ids = bfs(snap, &ids, &step.edge_type, step.depth, step.direction);
    }

    // Ordering (stable, numeric; missing values sort as 0).
    if let Some(order) = &plan.order {
        let field = order.field.clone();
        let desc = order.direction == OrderDirection::Desc;
        ids.sort_by(|a, b| {
            let av = snap.get(a).map(|n| attr_num(n, &field)).unwrap_or(0.0);
            let bv = snap.get(b).map(|n| attr_num(n, &field)).unwrap_or(0.0);
            let ord = av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal);
            if desc {
                ord.reverse()
            } else {
                ord
            }
        });
    }

    // Similarity.
    if let Some(sim) = &plan.similarity {
        let top_k = sim.top_k.unwrap_or(ids.len());
        let engine = sim.engine.unwrap_or(SimilarityEngine::Exact);
        match engine {
            SimilarityEngine::Exact => {
                let mut scored: Vec<(f64, String)> = Vec::new();
                for id in &ids {
                    let node = snap.get(id).unwrap();
                    let Some(vector) = &node.vector else { continue };
                    let score = cosine(&sim.vector, vector)?;
                    if score >= sim.threshold {
                        scored.push((score, id.clone()));
                    }
                }
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                scored.truncate(top_k);
                ids = scored.into_iter().map(|(_, id)| id).collect();
            }
            SimilarityEngine::Hnsw => {
                let index = hnsw.ok_or_else(|| PolypackError::InvalidArgument(
                    "similarity.engine=hnsw requires an hnsw index".into(),
                ))?;
                let ann = index.query(&sim.vector, top_k, sim.threshold)?;
                let scores: HashMap<String, f64> = ann.into_iter().map(|s| (s.id, s.score)).collect();
                let mut scored: Vec<(f64, String)> = ids
                    .into_iter()
                    .filter_map(|id| scores.get(&id).map(|s| (*s, id)))
                    .collect();
                scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
                ids = scored.into_iter().map(|(_, id)| id).collect();
            }
        }
    }

    if let Some(offset) = plan.offset {
        ids = ids.into_iter().skip(offset).collect();
    }
    if let Some(limit) = plan.limit {
        ids.truncate(limit);
    }
    Ok(ids)
}

/// Aggregate a numeric field across the nodes matched by `plan`.
pub fn aggregate(snap: &GraphSnapshot, plan: &QueryPlan, field: &str, op: &str) -> Result<(f64, usize)> {
    let ids = execute(snap, plan, None)?;
    let values: Vec<f64> = ids
        .iter()
        .filter_map(|id| snap.get(id))
        .filter_map(|n| n.data.get(field).and_then(|v| v.as_f64()))
        .filter(|v| v.is_finite())
        .collect();
    let count = values.len();
    if count == 0 {
        return Ok((0.0, 0));
    }
    let value = match op {
        "sum" => values.iter().sum(),
        "avg" => values.iter().sum::<f64>() / count as f64,
        "min" => values.iter().cloned().fold(f64::INFINITY, f64::min),
        "max" => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        "count" => count as f64,
        other => return Err(PolypackError::InvalidArgument(format!("unknown aggregate op {other}"))),
    };
    Ok((value, count))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::query::{Direction, Order, OrderDirection, QueryPlan, Similarity, SimilarityEngine, TraversalStep};
    use serde_json::json;

    fn node(id: &str, node_type: &str, data: Vec<(&str, serde_json::Value)>) -> Node {
        let mut map = serde_json::Map::new();
        for (k, v) in data {
            map.insert(k.to_string(), v);
        }
        Node {
            id: id.into(),
            node_type: node_type.into(),
            data: map,
            vector: None,
            inserted_at: 1,
            updated_at: 1,
        }
    }

    fn edge(source: &str, edge_type: &str, target: &str) -> Edge {
        Edge {
            id: format!("{source}::{edge_type}::{target}"),
            source: source.into(),
            target: target.into(),
            edge_type: edge_type.into(),
            data: None,
            created_at: 1,
        }
    }

    fn ids(snap: &GraphSnapshot, plan: &QueryPlan) -> Vec<String> {
        execute(snap, plan, None).unwrap()
    }

    #[test]
    fn filter_and_order() {
        let snap = GraphSnapshot::new(
            vec![
                node("d1", "document", vec![("category", json!("science")), ("score", json!(0.9))]),
                node("d2", "document", vec![("category", json!("science")), ("score", json!(0.5))]),
                node("d3", "document", vec![("category", json!("food")), ("score", json!(0.8))]),
                node("d4", "document", vec![("category", json!("science")), ("score", json!(0.7))]),
                node("d5", "document", vec![("category", json!("food")), ("score", json!(0.1))]),
            ],
            vec![],
        );
        let plan = QueryPlan {
            node_types: Some(vec!["document".into()]),
            attributes: Some(vec![AttributeFilter::Eq {
                field: "category".into(),
                value: json!("science"),
            }]),
            order: Some(Order { field: "score".into(), direction: OrderDirection::Desc }),
            ..Default::default()
        };
        assert_eq!(ids(&snap, &plan), vec!["d1", "d4", "d2"]);
    }

    #[test]
    fn range_and_order_asc() {
        let snap = GraphSnapshot::new(
            vec![
                node("d1", "document", vec![("score", json!(0.9))]),
                node("d2", "document", vec![("score", json!(0.5))]),
                node("d3", "document", vec![("score", json!(0.8))]),
                node("d4", "document", vec![("score", json!(0.7))]),
            ],
            vec![],
        );
        let plan = QueryPlan {
            attributes: Some(vec![AttributeFilter::Range {
                field: "score".into(),
                above: Some(0.6),
                below: None,
            }]),
            order: Some(Order { field: "score".into(), direction: OrderDirection::Asc }),
            ..Default::default()
        };
        assert_eq!(ids(&snap, &plan), vec!["d4", "d3", "d1"]);
    }

    #[test]
    fn traversal_out_includes_seeds() {
        let snap = GraphSnapshot::new(
            vec![
                node("a", "node", vec![("seed", json!("a"))]),
                node("b", "node", vec![("seed", json!("b"))]),
                node("c", "node", vec![("seed", json!("c"))]),
                node("d", "node", vec![("seed", json!("d"))]),
            ],
            vec![edge("a", "REFERENCES", "b"), edge("b", "REFERENCES", "c"), edge("c", "REFERENCES", "d")],
        );
        let plan = QueryPlan {
            attributes: Some(vec![AttributeFilter::Eq { field: "seed".into(), value: json!("a") }]),
            traversal: Some(vec![TraversalStep {
                edge_type: "REFERENCES".into(),
                direction: Direction::Out,
                depth: 2,
            }]),
            ..Default::default()
        };
        assert_eq!(ids(&snap, &plan), vec!["a", "b", "c"]);
    }

    #[test]
    fn traversal_in_direction() {
        let snap = GraphSnapshot::new(
            vec![
                node("a", "node", vec![("seed", json!("a"))]),
                node("b", "node", vec![("seed", json!("b"))]),
                node("c", "node", vec![("seed", json!("c"))]),
            ],
            vec![edge("b", "REFERENCES", "c"), edge("a", "REFERENCES", "b")],
        );
        let plan = QueryPlan {
            attributes: Some(vec![AttributeFilter::Eq { field: "seed".into(), value: json!("c") }]),
            traversal: Some(vec![TraversalStep {
                edge_type: "REFERENCES".into(),
                direction: Direction::In,
                depth: 2,
            }]),
            ..Default::default()
        };
        assert_eq!(ids(&snap, &plan), vec!["c", "b", "a"]);
    }

    #[test]
    fn join_connectivity() {
        let snap = GraphSnapshot::new(
            vec![
                node("u1", "user", vec![]),
                node("u2", "user", vec![]),
                node("u3", "user", vec![]),
                node("b1", "book", vec![]),
                node("b2", "book", vec![]),
            ],
            vec![edge("u1", "RATED", "b1"), edge("u2", "RATED", "b2"), edge("u2", "RATED", "b1")],
        );
        let plan = QueryPlan {
            node_types: Some(vec!["user".into()]),
            joins: Some(vec![crate::query::Join { edge_type: "RATED".into(), direction: Direction::Out }]),
            ..Default::default()
        };
        assert_eq!(ids(&snap, &plan), vec!["u1", "u2"]);
    }

    #[test]
    fn pagination_and_aggregate() {
        let snap = GraphSnapshot::new(
            vec![
                node("p1", "item", vec![("score", json!(1))]),
                node("p2", "item", vec![("score", json!(2))]),
                node("p3", "item", vec![("score", json!(3))]),
                node("p4", "item", vec![("score", json!(4))]),
                node("p5", "item", vec![("score", json!(5))]),
            ],
            vec![],
        );
        let plan = QueryPlan {
            order: Some(Order { field: "score".into(), direction: OrderDirection::Desc }),
            offset: Some(1),
            limit: Some(2),
            ..Default::default()
        };
        assert_eq!(ids(&snap, &plan), vec!["p4", "p3"]);
        let (value, count) = aggregate(&snap, &QueryPlan::default(), "score", "sum").unwrap();
        assert_eq!((value, count), (15.0, 5));
    }

    #[test]
    fn similarity_exact() {
        let mut n1 = node("a", "vec", vec![]);
        n1.vector = Some(vec![1.0, 0.0, 0.0]);
        let mut n2 = node("b", "vec", vec![]);
        n2.vector = Some(vec![0.8, 0.6, 0.0]);
        let mut n3 = node("c", "vec", vec![]);
        n3.vector = Some(vec![0.0, 1.0, 0.0]);
        let snap = GraphSnapshot::new(vec![n1, n2, n3], vec![]);
        let plan = QueryPlan {
            similarity: Some(Similarity {
                vector: vec![1.0, 0.0, 0.0],
                threshold: 0.5,
                top_k: Some(2),
                engine: None,
            }),
            ..Default::default()
        };
        assert_eq!(ids(&snap, &plan), vec!["a", "b"]);
    }

    #[test]
    fn similarity_hnsw_opt_in() {
        let mut n1 = node("a", "vec", vec![]);
        n1.vector = Some(vec![1.0, 0.0, 0.0, 0.0]);
        let mut n2 = node("b", "vec", vec![]);
        n2.vector = Some(vec![0.9, 0.1, 0.0, 0.0]);
        let mut n3 = node("c", "vec", vec![]);
        n3.vector = Some(vec![0.0, 1.0, 0.0, 0.0]);
        let snap = GraphSnapshot::new(vec![n1, n2, n3], vec![]);
        let mut hnsw = HnswIndex::new(crate::hnsw::HnswConfig { ef_search: 100, ..Default::default() }, 7);
        for n in &snap.nodes {
            hnsw.add(&n.id, n.vector.as_deref().unwrap()).unwrap();
        }
        let plan = QueryPlan {
            similarity: Some(Similarity {
                vector: vec![1.0, 0.0, 0.0, 0.0],
                threshold: 0.5,
                top_k: Some(2),
                engine: Some(SimilarityEngine::Hnsw),
            }),
            ..Default::default()
        };
        let got = execute(&snap, &plan, Some(&hnsw)).unwrap();
        assert_eq!(got, vec!["a", "b"]);
    }

    #[test]
    fn dimension_mismatch_propagates() {
        let mut n1 = node("a", "vec", vec![]);
        n1.vector = Some(vec![1.0, 0.0]);
        let snap = GraphSnapshot::new(vec![n1], vec![]);
        let plan = QueryPlan {
            similarity: Some(Similarity {
                vector: vec![1.0, 0.0, 0.0],
                threshold: 0.0,
                top_k: Some(1),
                engine: None,
            }),
            ..Default::default()
        };
        assert!(matches!(execute(&snap, &plan, None), Err(PolypackError::DimensionMismatch { .. })));
    }
}
