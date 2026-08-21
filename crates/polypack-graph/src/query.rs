//! [`GraphQuery`]: the Rust counterpart to `GraphQuery` (`src/query.ts`).
//!
//! A fluent, chainable query builder over `Graph`'s hot working set,
//! constructed by [`crate::Graph::query`]. TS has two implementations under
//! the hood — its own reference matching/traversal logic, used as a
//! fallback, and an optional native-accelerated path via a registered
//! executor. Rust has no such split: this *is* the native path, so
//! `GraphQuery` directly ports TS's reference algorithm (there is nothing to
//! bridge to or fall back from).

use std::cell::RefCell;
use std::collections::{HashMap, HashSet};
use std::time::Instant;

use polypack_core::activation::{activation_score_of, DEFAULT_ACTIVATION};
use polypack_core::query::Direction;
use polypack_core::vector::cosine;
use polypack_core::{Node, PolypackError, Result};

use crate::edge::EdgeEntry;
use crate::graph::{now_millis, IndexDefinition, QueryMetrics};
use crate::persisted_query::QueryExplain;

fn query_field_value(attributes: &HashMap<String, serde_json::Value>, field: &str) -> serde_json::Value {
    attributes
        .get(field)
        .or_else(|| attributes.get(field.strip_prefix("data.").unwrap_or(field)))
        .cloned()
        .unwrap_or(serde_json::Value::Null)
}

/// A `join` predicate closure: `Some` to filter by the connected node,
/// `None` to require only that a connection exists.
type JoinPredicate<'a> = Box<dyn Fn(&Node) -> bool + 'a>;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum OrderDirection {
    Asc,
    Desc,
}

#[derive(Clone, Debug, Default)]
struct RangeFilter {
    above: Option<f64>,
    below: Option<f64>,
}

#[derive(Clone, Debug)]
struct TraversalStep {
    edge_type: String,
    depth: usize,
    direction: Direction,
}

#[derive(Clone, Debug)]
struct SimilaritySpec {
    vector: Vec<f64>,
    threshold: f64,
    top_k: Option<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AggregateOp {
    Sum,
    Avg,
    Min,
    Max,
    Count,
}

#[derive(Clone, Debug, PartialEq)]
pub struct AggregateResult {
    pub value: f64,
    pub count: usize,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GroupedRow {
    pub key: String,
    pub value: f64,
    pub count: usize,
}

/// `String(value ?? 'null')` semantics for a group-by key: a JSON string
/// unwraps without quotes (unlike `serde_json::Value`'s `Display`, which
/// JSON-encodes it); other JSON types use their normal textual form, which
/// already matches JS's `String()` for numbers/booleans.
fn display_key(value: Option<&serde_json::Value>) -> String {
    match value {
        None => "null".to_string(),
        Some(serde_json::Value::String(s)) => s.clone(),
        Some(v) => v.to_string(),
    }
}

fn aggregate_values(op: AggregateOp, values: &[f64]) -> f64 {
    match op {
        AggregateOp::Sum => values.iter().sum(),
        AggregateOp::Avg => values.iter().sum::<f64>() / values.len() as f64,
        AggregateOp::Min => values.iter().cloned().fold(f64::INFINITY, f64::min),
        AggregateOp::Max => values.iter().cloned().fold(f64::NEG_INFINITY, f64::max),
        AggregateOp::Count => values.len() as f64,
    }
}

/// Mutable fluent query builder over a `Graph`'s hot node/edge working set.
/// Mirrors `GraphQuery` in `src/query.ts`.
pub struct GraphQuery<'a> {
    nodes: &'a HashMap<String, Node>,
    edges: &'a HashMap<String, HashMap<String, EdgeEntry>>,
    node_to_edge: &'a HashMap<String, HashSet<String>>,
    indexes: &'a HashMap<String, IndexDefinition>,
    secondary_indexes: &'a HashMap<String, HashMap<String, HashSet<String>>>,
    metrics: &'a RefCell<QueryMetrics>,

    node_types: Option<Vec<String>>,
    attributes: HashMap<String, serde_json::Value>,
    attribute_ranges: HashMap<String, RangeFilter>,
    edge_type: Option<String>,
    edge_target: Option<String>,
    edge_source: Option<String>,
    order_by: Option<(String, OrderDirection)>,
    limit: Option<usize>,
    offset: Option<usize>,
    traversal_steps: Vec<TraversalStep>,
    similarity: Option<SimilaritySpec>,
    join_filters: Vec<JoinPredicate<'a>>,
    activation_above: Option<f64>,
    activation_order: Option<OrderDirection>,
}

impl<'a> GraphQuery<'a> {
    pub(crate) fn new(
        nodes: &'a HashMap<String, Node>,
        edges: &'a HashMap<String, HashMap<String, EdgeEntry>>,
        node_to_edge: &'a HashMap<String, HashSet<String>>,
        indexes: &'a HashMap<String, IndexDefinition>,
        secondary_indexes: &'a HashMap<String, HashMap<String, HashSet<String>>>,
        metrics: &'a RefCell<QueryMetrics>,
    ) -> Self {
        Self {
            nodes,
            edges,
            node_to_edge,
            indexes,
            secondary_indexes,
            metrics,
            node_types: None,
            attributes: HashMap::new(),
            attribute_ranges: HashMap::new(),
            edge_type: None,
            edge_target: None,
            edge_source: None,
            order_by: None,
            limit: None,
            offset: None,
            traversal_steps: Vec::new(),
            similarity: None,
            join_filters: Vec::new(),
            activation_above: None,
            activation_order: None,
        }
    }

    // ── builder methods ──

    /// Equality filter on a data field (or `"type"` for the node type).
    /// Alias: `where_attribute`. Later calls with the same field overwrite
    /// earlier ones.
    pub fn where_field(mut self, field: &str, value: serde_json::Value) -> Self {
        self.attributes.insert(field.to_string(), value);
        self
    }

    /// Alias for `where_field`.
    pub fn where_attribute(self, name: &str, value: serde_json::Value) -> Self {
        self.where_field(name, value)
    }

    pub fn where_attribute_range(mut self, name: &str, above: Option<f64>, below: Option<f64>) -> Self {
        self.attribute_ranges.insert(name.to_string(), RangeFilter { above, below });
        self
    }

    pub fn where_node_type(mut self, types: Vec<String>) -> Self {
        self.node_types = Some(types);
        self
    }

    pub fn where_edge(mut self, edge_type: &str, target: Option<&str>) -> Self {
        self.edge_type = Some(edge_type.to_string());
        if let Some(target) = target {
            self.edge_target = Some(target.to_string());
        }
        self
    }

    pub fn where_edge_source(mut self, source: &str) -> Self {
        self.edge_source = Some(source.to_string());
        self
    }

    pub fn order_by(mut self, field: &str, direction: OrderDirection) -> Self {
        self.order_by = Some((field.to_string(), direction));
        self
    }

    /// Keep only nodes whose current (decay-corrected) activation exceeds
    /// `above`. Mirrors `GraphQuery.whereActivated`.
    pub fn where_activated(mut self, above: f64) -> Self {
        self.activation_above = Some(above);
        self
    }

    /// Order results by current (decay-corrected) activation instead of a data
    /// field. Mirrors `GraphQuery.orderByActivation`.
    pub fn order_by_activation(mut self, direction: OrderDirection) -> Self {
        self.activation_order = Some(direction);
        self
    }

    pub fn limit(mut self, n: usize) -> Self {
        self.limit = Some(n);
        self
    }

    pub fn offset(mut self, n: usize) -> Self {
        self.offset = Some(n);
        self
    }

    pub fn traverse(mut self, edge_type: &str, depth: usize, direction: Direction) -> Self {
        self.traversal_steps.push(TraversalStep { edge_type: edge_type.to_string(), depth, direction });
        self
    }

    pub fn similar_to(mut self, vector: Vec<f64>, threshold: f64, top_k: Option<usize>) -> Self {
        self.similarity = Some(SimilaritySpec { vector, threshold, top_k });
        self
    }

    /// Describe the hot-query stages without materializing results.
    pub fn explain(&self) -> QueryExplain {
        let selected = self.selected_indexes();
        let indexes: Vec<String> = selected.iter().map(|definition| definition.name.clone()).collect();
        let index = indexes.first().cloned().or_else(|| self.node_types.as_ref().filter(|types| !types.is_empty()).map(|_| "type-index".to_string()));
        let mut stages: Vec<String> = if indexes.is_empty() {
            vec![if index.as_deref() == Some("type-index") { "type-index".to_string() } else { "record-scan".to_string() }]
        } else {
            indexes.iter().map(|name| format!("property-index({name})")).collect()
        };
        if indexes.len() > 1 {
            stages.push(format!("index-intersection({})", indexes.len()));
        }
        if let Some(types) = &self.node_types {
            if !types.is_empty() {
                stages.push(format!("type-filter({})", types.join(",")));
            }
        }
        if !self.attributes.is_empty() || !self.attribute_ranges.is_empty() {
            stages.push("property-filter".to_string());
        }
        if !self.join_filters.is_empty() {
            stages.push(format!("join(count={})", self.join_filters.len()));
        }
        if !self.traversal_steps.is_empty() {
            let depth = self.traversal_steps.iter().map(|step| step.depth).max().unwrap_or(0);
            stages.push(format!("traversal(depth={depth})"));
        }
        if let Some((field, direction)) = &self.order_by {
            stages.push(format!("order({field},{direction:?})").to_lowercase());
        }
        if let Some(limit) = self.limit {
            stages.push(format!("limit({limit})"));
        }
        let loaded_records = self.nodes.len();
        QueryExplain {
            index,
            indexes,
            stages,
            loaded_records,
            estimated_cost: (loaded_records as f64 * if selected.is_empty() { 1.0 } else { 0.25 / selected.len() as f64 }).max(1.0),
        }
    }

    /// Materialize results while enforcing traversal and result limits.
    pub fn to_array_limited(&self, limits: &crate::QueryResourceLimits) -> Result<Vec<Node>> {
        for step in &self.traversal_steps {
            if let Some(max_depth) = limits.max_traversal_depth {
                if step.depth > max_depth {
                    return Err(PolypackError::ResourceLimit { name: "maxTraversalDepth".into(), limit: max_depth });
                }
            }
        }
        let results = self.to_array();
        if let Some(max_nodes) = limits.max_nodes_visited {
            if results.len() > max_nodes {
                return Err(PolypackError::ResourceLimit { name: "maxNodesVisited".into(), limit: max_nodes });
            }
        }
        if let Some(max_results) = limits.max_results {
            if results.len() > max_results {
                return Err(PolypackError::ResourceLimit { name: "maxResults".into(), limit: max_results });
            }
        }
        Ok(results)
    }

    /// Constrain results to nodes connected via `edge_type`. Unlike
    /// `where_edge` (a specific target id), this matches ANY edge of the
    /// given type, optionally filtered by the connected node via
    /// `predicate`. Mirrors `GraphQuery.join`.
    pub fn join(
        mut self,
        edge_type: &str,
        direction: Direction,
        predicate: Option<JoinPredicate<'a>>,
    ) -> Self {
        let nodes = self.nodes;
        let edges = self.edges;
        let node_to_edge = self.node_to_edge;
        let edge_type = edge_type.to_string();
        let filter = move |node: &Node| -> bool {
            let connected: Vec<&Node> = match direction {
                Direction::Out => edges
                    .get(&node.id)
                    .into_iter()
                    .flat_map(|m| m.values())
                    .filter(|e| e.edge_type == edge_type)
                    .filter_map(|e| nodes.get(&e.target))
                    .collect(),
                Direction::In => node_to_edge
                    .get(&node.id)
                    .into_iter()
                    .flatten()
                    .filter(|src| {
                        edges
                            .get(src.as_str())
                            .is_some_and(|m| m.values().any(|e| e.edge_type == edge_type && e.target == node.id))
                    })
                    .filter_map(|src| nodes.get(src))
                    .collect(),
            };
            if connected.is_empty() {
                return false;
            }
            match &predicate {
                Some(p) => connected.iter().any(|n| p(n)),
                None => true,
            }
        };
        self.join_filters.push(Box::new(filter));
        self
    }

    // ── matching/traversal internals ──

    fn matches(&self, node: &Node) -> bool {
        if let Some(types) = &self.node_types {
            if !types.iter().any(|t| t == &node.node_type) {
                return false;
            }
        }
        for (key, expected) in &self.attributes {
            if key == "type" {
                if expected.as_str() != Some(node.node_type.as_str()) {
                    return false;
                }
            } else if node.data.get(key) != Some(expected) {
                return false;
            }
        }
        for (key, range) in &self.attribute_ranges {
            let Some(val) = node.data.get(key).and_then(|v| v.as_f64()) else { return false };
            if let Some(above) = range.above {
                if val <= above {
                    return false;
                }
            }
            if let Some(below) = range.below {
                if val >= below {
                    return false;
                }
            }
        }
        if let Some(edge_type) = &self.edge_type {
            let matched = self.edges.get(&node.id).is_some_and(|edges| {
                edges
                    .values()
                    .any(|e| &e.edge_type == edge_type && self.edge_target.as_deref().is_none_or(|t| e.target == t))
            });
            if !matched {
                return false;
            }
        }
        if let Some(source) = &self.edge_source {
            let matched = self.edges.get(source).is_some_and(|edges| {
                edges
                    .values()
                    .any(|e| e.target == node.id && self.edge_type.as_deref().is_none_or(|t| e.edge_type == t))
            });
            if !matched {
                return false;
            }
        }
        if let Some(above) = self.activation_above {
            if activation_score_of(node, now_millis(), DEFAULT_ACTIVATION.score_half_life_ms) < above {
                return false;
            }
        }
        self.join_filters.iter().all(|f| f(node))
    }

    fn source_nodes(&self) -> Vec<&'a Node> {
        let mut ids: HashSet<String> = HashSet::new();

        if let (Some(source), Some(edge_type)) = (&self.edge_source, &self.edge_type) {
            if let Some(edges) = self.edges.get(source) {
                for e in edges.values() {
                    if &e.edge_type == edge_type {
                        ids.insert(e.target.clone());
                    }
                }
            }
        } else if let (Some(target), Some(edge_type)) = (&self.edge_target, &self.edge_type) {
            for (node_id, edges) in self.edges {
                for e in edges.values() {
                    if &e.edge_type == edge_type && &e.target == target {
                        ids.insert(node_id.clone());
                    }
                }
            }
        }

        if !ids.is_empty() {
            return ids.iter().filter_map(|id| self.nodes.get(id)).collect();
        }

        let selected = self.selected_indexes();
        if !selected.is_empty() {
            let mut candidate_sets = Vec::new();
            for index in selected {
                if index.fields.iter().all(|field| self.attributes.contains_key(field) || self.attributes.contains_key(field.strip_prefix("data.").unwrap_or(field))) {
                    let values = index.fields.iter().map(|field| query_field_value(&self.attributes, field)).collect::<Vec<_>>();
                    let key = serde_json::to_string(&values).expect("JSON index key serialization cannot fail");
                    candidate_sets.push(self.secondary_indexes.get(&index.name).and_then(|buckets| buckets.get(&key)).cloned().unwrap_or_default());
                } else if index.fields.len() == 1 {
                    let field = &index.fields[0];
                    let range = &self.attribute_ranges[field];
                    let ids: HashSet<String> = self.secondary_indexes.get(&index.name).into_iter().flat_map(|buckets| buckets.iter()).filter_map(|(encoded, bucket)| {
                        let values = serde_json::from_str::<Vec<serde_json::Value>>(encoded).ok()?;
                        let value = values.first()?.as_f64()?;
                        if range.above.is_some_and(|above| value <= above) || range.below.is_some_and(|below| value >= below) { return None; }
                        Some(bucket.iter().cloned())
                    }).flatten().collect();
                    candidate_sets.push(ids);
                }
            }
            if let Some((first, rest)) = candidate_sets.split_first() {
                let ids = rest.iter().fold(first.clone(), |mut ids, candidate| { ids.retain(|id| candidate.contains(id)); ids });
                return ids.iter().filter_map(|id| self.nodes.get(id)).collect();
            }
        }

        match &self.node_types {
            Some(types) => self.nodes.values().filter(|n| types.iter().any(|t| t == &n.node_type)).collect(),
            None => self.nodes.values().collect(),
        }
    }

    fn selected_indexes(&self) -> Vec<&IndexDefinition> {
        let mut selected: Vec<&IndexDefinition> = self.indexes.values().filter(|definition| {
            let compatible_type = self.node_types.as_ref().is_none_or(|types| definition.node_type.as_ref().is_none_or(|node_type| types.len() == 1 && types[0] == *node_type));
            let equality = !definition.fields.is_empty()
                && definition.fields.iter().all(|field| self.attributes.contains_key(field) || self.attributes.contains_key(field.strip_prefix("data.").unwrap_or(field)))
                && (!definition.sparse || definition.fields.iter().all(|field| !query_field_value(&self.attributes, field).is_null()));
            let range = definition.fields.len() == 1 && self.attribute_ranges.contains_key(&definition.fields[0]);
            compatible_type && (equality || range)
        }).collect();
        selected.sort_by(|left, right| left.name.cmp(&right.name));
        selected
    }

    fn bfs(&self, seeds: &[String], step: &TraversalStep) -> HashSet<String> {
        let mut visited: HashSet<String> = seeds.iter().cloned().collect();
        let mut frontier: Vec<String> = seeds.to_vec();
        for _ in 0..step.depth {
            if frontier.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for id in &frontier {
                match step.direction {
                    Direction::Out => {
                        if let Some(edges) = self.edges.get(id) {
                            for e in edges.values() {
                                if e.edge_type == step.edge_type && visited.insert(e.target.clone()) {
                                    next.push(e.target.clone());
                                }
                            }
                        }
                    }
                    Direction::In => {
                        if let Some(sources) = self.node_to_edge.get(id) {
                            for src in sources {
                                if visited.contains(src) {
                                    continue;
                                }
                                let matches = self
                                    .edges
                                    .get(src)
                                    .is_some_and(|edges| edges.values().any(|e| e.edge_type == step.edge_type && &e.target == id));
                                if matches && visited.insert(src.clone()) {
                                    next.push(src.clone());
                                }
                            }
                        }
                    }
                }
            }
            frontier = next;
        }
        visited
    }

    fn apply_traversals(&self, ids: Vec<String>) -> HashSet<String> {
        if self.traversal_steps.is_empty() {
            return ids.into_iter().collect();
        }
        let mut current: HashSet<String> = ids.into_iter().collect();
        for step in &self.traversal_steps {
            let seeds: Vec<String> = current.into_iter().collect();
            current = self.bfs(&seeds, step);
        }
        current
    }

    fn resolve(&self, ids: &HashSet<String>) -> Vec<&'a Node> {
        ids.iter().filter_map(|id| self.nodes.get(id)).collect()
    }

    // ── terminals ──

    /// Materialize matched nodes as detached snapshots (cloned out of the
    /// graph's hot working set). Mirrors `GraphQuery.toArray`.
    pub fn to_array(&self) -> Vec<Node> {
        let started = Instant::now();
        let source = self.source_nodes();
        let scanned_records = source.len();
        let mut results: Vec<&Node> = source.into_iter().filter(|n| self.matches(n)).collect();
        if results.is_empty() {
            self.record_metrics(started, scanned_records);
            return Vec::new();
        }

        if !self.traversal_steps.is_empty() {
            let ids: Vec<String> = results.iter().map(|n| n.id.clone()).collect();
            let expanded = self.apply_traversals(ids);
            results = self.resolve(&expanded);
        }

        if let Some((field, direction)) = &self.order_by {
            results.sort_by(|a, b| {
                let av = a.data.get(field).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let bv = b.data.get(field).and_then(|v| v.as_f64()).unwrap_or(0.0);
                let ord = av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal);
                if *direction == OrderDirection::Desc {
                    ord.reverse()
                } else {
                    ord
                }
            });
        }

        if let Some(direction) = self.activation_order {
            let now = now_millis();
            results.sort_by(|a, b| {
                let av = activation_score_of(a, now, DEFAULT_ACTIVATION.score_half_life_ms);
                let bv = activation_score_of(b, now, DEFAULT_ACTIVATION.score_half_life_ms);
                let ord = av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal);
                if direction == OrderDirection::Desc {
                    ord.reverse()
                } else {
                    ord
                }
            });
        }

        if let Some(sim) = &self.similarity {
            let mut scored: Vec<(f64, &Node)> = results
                .into_iter()
                .filter_map(|n| {
                    let v = n.vector.as_ref()?;
                    let score = cosine(&sim.vector, v).ok()?;
                    (score >= sim.threshold).then_some((score, n))
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(top_k) = sim.top_k {
                scored.truncate(top_k);
            }
            results = scored.into_iter().map(|(_, n)| n).collect();
        }

        if let Some(offset) = self.offset {
            results = results.into_iter().skip(offset).collect();
        }
        if let Some(limit) = self.limit {
            results.truncate(limit);
        }

        let output = results.into_iter().cloned().collect();
        self.record_metrics(started, scanned_records);
        output
    }

    fn record_metrics(&self, started: Instant, scanned_records: usize) {
        let selected = self.selected_indexes();
        let mut metrics = self.metrics.borrow_mut();
        metrics.count += 1;
        metrics.duration_ms += started.elapsed().as_secs_f64() * 1000.0;
        metrics.scanned_records += scanned_records;
        for index in selected {
            *metrics.index_usage.entry(index.name.clone()).or_default() += 1;
        }
    }

    pub fn first(&self) -> Option<Node> {
        self.to_array().into_iter().next()
    }

    pub fn count(&self) -> usize {
        let needs_materialization = !self.traversal_steps.is_empty()
            || self.similarity.is_some()
            || self.offset.is_some()
            || self.limit.is_some()
            || self.activation_above.is_some()
            || self.activation_order.is_some();
        if needs_materialization {
            return self.to_array().len();
        }
        self.source_nodes().into_iter().filter(|n| self.matches(n)).count()
    }

    pub fn ids(&self) -> Vec<String> {
        self.to_array().into_iter().map(|n| n.id).collect()
    }

    // ── relational extensions ──

    /// Project selected fields into plain rows (`id`, `type`, then each
    /// requested field). Terminal. Mirrors `GraphQuery.pluck`.
    pub fn pluck(&self, fields: &[&str]) -> Vec<serde_json::Map<String, serde_json::Value>> {
        self.to_array()
            .into_iter()
            .map(|n| {
                let mut row = serde_json::Map::new();
                row.insert("id".into(), serde_json::Value::String(n.id));
                row.insert("type".into(), serde_json::Value::String(n.node_type));
                for f in fields {
                    row.insert((*f).to_string(), n.data.get(*f).cloned().unwrap_or(serde_json::Value::Null));
                }
                row
            })
            .collect()
    }

    /// Aggregate a numeric field across all matched nodes. Terminal.
    pub fn aggregate(&self, field: &str, op: AggregateOp) -> AggregateResult {
        let nodes = self.to_array();
        let values: Vec<f64> = nodes.iter().filter_map(|n| n.data.get(field)).filter_map(|v| v.as_f64()).collect();
        if values.is_empty() {
            return AggregateResult { value: 0.0, count: 0 };
        }
        AggregateResult { value: aggregate_values(op, &values), count: values.len() }
    }

    /// Group by a field and aggregate. Terminal. Mirrors
    /// `GraphQuery.groupAggregate`.
    pub fn group_aggregate(&self, field: &str, op: AggregateOp, group_by_field: &str) -> Vec<GroupedRow> {
        let nodes = self.to_array();
        let mut groups: HashMap<String, Vec<f64>> = HashMap::new();
        for n in &nodes {
            let key = display_key(n.data.get(group_by_field));
            let val = n.data.get(field);
            let include = if op == AggregateOp::Count {
                val.is_some_and(|v| !v.is_null())
            } else {
                val.and_then(|v| v.as_f64()).is_some()
            };
            if include {
                let value = if op == AggregateOp::Count { 1.0 } else { val.and_then(|v| v.as_f64()).unwrap() };
                groups.entry(key).or_default().push(value);
            }
        }
        groups
            .into_iter()
            .map(|(key, values)| GroupedRow { key, value: aggregate_values(op, &values), count: values.len() })
            .collect()
    }

    /// Filter groups produced by `group_aggregate`/`group_by_vector` by
    /// predicate. Pure — doesn't use the query's own state.
    pub fn having(groups: Vec<GroupedRow>, predicate: impl Fn(&GroupedRow) -> bool) -> Vec<GroupedRow> {
        groups.into_iter().filter(predicate).collect()
    }

    /// Group matched nodes by nearest-centroid vector clustering (cosine).
    /// Nodes below `threshold` fall into `"null"`; nodes without a vector
    /// are excluded. Terminal. Mirrors `GraphQuery.groupByVector`.
    pub fn group_by_vector(
        &self,
        groups: &[(String, Vec<f64>)],
        field: &str,
        op: AggregateOp,
        threshold: f64,
    ) -> Vec<GroupedRow> {
        let nodes = self.to_array();
        let mut clusters: HashMap<String, Vec<f64>> = HashMap::new();
        for n in &nodes {
            let Some(vector) = &n.vector else { continue };
            let mut best_key = "null".to_string();
            let mut best_score = threshold;
            for (key, centroid) in groups {
                if let Ok(score) = cosine(centroid, vector) {
                    if score > best_score {
                        best_score = score;
                        best_key = key.clone();
                    }
                }
            }
            let val = n.data.get(field);
            let include = if op == AggregateOp::Count {
                val.is_some_and(|v| !v.is_null())
            } else {
                val.and_then(|v| v.as_f64()).is_some()
            };
            if include {
                let value = if op == AggregateOp::Count { 1.0 } else { val.and_then(|v| v.as_f64()).unwrap() };
                clusters.entry(best_key).or_default().push(value);
            }
        }
        clusters
            .into_iter()
            .map(|(key, values)| GroupedRow { key, value: aggregate_values(op, &values), count: values.len() })
            .collect()
    }

    /// Distinct values of `field` across every node in the graph —
    /// deliberately ignores all filters, matching `GraphQuery.uniqueKeys`.
    pub fn unique_keys(&self, field: &str) -> Vec<serde_json::Value> {
        let mut seen = Vec::new();
        for node in self.nodes.values() {
            if let Some(val) = node.data.get(field) {
                if !seen.contains(val) {
                    seen.push(val.clone());
                }
            }
        }
        seen
    }

    /// From the matched nodes, collect distinct nodes reachable via one
    /// `edge_type` hop, optionally filtered by `predicate`. Terminal.
    /// Mirrors `GraphQuery.collect`.
    pub fn collect(&self, edge_type: &str, direction: Direction, predicate: Option<&dyn Fn(&Node) -> bool>) -> Vec<Node> {
        let seeds = self.to_array();
        let mut collected = Vec::new();
        let mut seen = HashSet::new();

        for seed in &seeds {
            match direction {
                Direction::Out => {
                    if let Some(edges) = self.edges.get(&seed.id) {
                        for e in edges.values() {
                            if e.edge_type != edge_type || !seen.insert(e.target.clone()) {
                                continue;
                            }
                            if let Some(node) = self.nodes.get(&e.target) {
                                if predicate.is_none_or(|p| p(node)) {
                                    collected.push(node.clone());
                                }
                            }
                        }
                    }
                }
                Direction::In => {
                    if let Some(sources) = self.node_to_edge.get(&seed.id) {
                        for src in sources {
                            if seen.contains(src) {
                                continue;
                            }
                            let matched = self
                                .edges
                                .get(src)
                                .is_some_and(|edges| edges.values().any(|e| e.edge_type == edge_type && e.target == seed.id));
                            if !matched {
                                continue;
                            }
                            seen.insert(src.clone());
                            if let Some(node) = self.nodes.get(src) {
                                if predicate.is_none_or(|p| p(node)) {
                                    collected.push(node.clone());
                                }
                            }
                        }
                    }
                }
            }
        }

        collected
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::edge::EdgeOwnership;
    use polypack_core::NodeActivation;

    fn node(id: &str, node_type: &str, data: &[(&str, serde_json::Value)]) -> Node {
        let mut map = serde_json::Map::new();
        for (k, v) in data {
            map.insert((*k).to_string(), v.clone());
        }
        Node { id: id.to_string(), node_type: node_type.to_string(), data: map, vector: None, inserted_at: 1, updated_at: 1, revision: 0, activation: None }
    }

    struct Fixture {
        nodes: HashMap<String, Node>,
        edges: HashMap<String, HashMap<String, EdgeEntry>>,
        node_to_edge: HashMap<String, HashSet<String>>,
        indexes: HashMap<String, IndexDefinition>,
        secondary_indexes: HashMap<String, HashMap<String, HashSet<String>>>,
        metrics: RefCell<QueryMetrics>,
    }

    impl Fixture {
        fn new() -> Self {
            Self { nodes: HashMap::new(), edges: HashMap::new(), node_to_edge: HashMap::new(), indexes: HashMap::new(), secondary_indexes: HashMap::new(), metrics: RefCell::new(QueryMetrics::default()) }
        }

        fn add(&mut self, node: Node) -> &mut Self {
            self.nodes.insert(node.id.clone(), node);
            self
        }

        fn edge(&mut self, source: &str, edge_type: &str, target: &str) -> &mut Self {
            let inner = format!("{edge_type}::{target}");
            self.edges.entry(source.to_string()).or_default().insert(
                inner,
                EdgeEntry {
                    id: format!("{source}::{edge_type}::{target}"),
                    revision: 0,
                    target: target.to_string(),
                    edge_type: edge_type.to_string(),
                    data: None,
                    ownership: EdgeOwnership::Reference,
                },
            );
            self.node_to_edge.entry(target.to_string()).or_default().insert(source.to_string());
            self
        }

        fn query(&self) -> GraphQuery<'_> {
            GraphQuery::new(&self.nodes, &self.edges, &self.node_to_edge, &self.indexes, &self.secondary_indexes, &self.metrics)
        }
    }

    /// alice/bob rate two books; scifi1/fantasy1 carry genre+rating+a vector
    /// leaning toward a "sci-fi" or "fantasy" centroid respectively.
    fn library() -> Fixture {
        let mut f = Fixture::new();
        f.add(node("alice", "user", &[("name", serde_json::json!("Alice"))]));
        f.add(node("bob", "user", &[("name", serde_json::json!("Bob"))]));
        let mut scifi1 =
            node("scifi1", "book", &[("genre", serde_json::json!("sci-fi")), ("rating", serde_json::json!(5.0))]);
        scifi1.vector = Some(vec![1.0, 0.0]);
        f.add(scifi1);
        let mut fantasy1 =
            node("fantasy1", "book", &[("genre", serde_json::json!("fantasy")), ("rating", serde_json::json!(3.0))]);
        fantasy1.vector = Some(vec![0.0, 1.0]);
        f.add(fantasy1);
        f.edge("alice", "RATED", "scifi1");
        f.edge("alice", "RATED", "fantasy1");
        f.edge("bob", "RATED", "scifi1");
        f
    }

    fn ids_of(mut nodes: Vec<Node>) -> Vec<String> {
        nodes.sort_by(|a, b| a.id.cmp(&b.id));
        nodes.into_iter().map(|n| n.id).collect()
    }

    #[test]
    fn where_node_type_filters() {
        let f = library();
        assert_eq!(ids_of(f.query().where_node_type(vec!["user".into()]).to_array()), vec!["alice", "bob"]);
    }

    #[test]
    fn where_field_filters_by_attribute() {
        let f = library();
        let results = f.query().where_field("genre", serde_json::json!("sci-fi")).to_array();
        assert_eq!(ids_of(results), vec!["scifi1"]);
    }

    #[test]
    fn where_field_on_type_key_matches_node_type() {
        let f = library();
        let results = f.query().where_field("type", serde_json::json!("book")).to_array();
        assert_eq!(ids_of(results), vec!["fantasy1", "scifi1"]);
    }

    #[test]
    fn where_attribute_range_filters_exclusive_bounds() {
        let f = library();
        let results = f.query().where_attribute_range("rating", Some(3.0), None).to_array();
        assert_eq!(ids_of(results), vec!["scifi1"], "exclusive bound excludes rating == 3.0");
    }

    #[test]
    fn where_edge_matches_source_of_a_specific_edge_type_and_target() {
        let f = library();
        let results = f.query().where_edge("RATED", Some("fantasy1")).to_array();
        assert_eq!(ids_of(results), vec!["alice"]);
    }

    #[test]
    fn where_edge_source_matches_targets_of_that_source() {
        let f = library();
        let results = f.query().where_edge_source("bob").to_array();
        assert_eq!(ids_of(results), vec!["scifi1"]);
    }

    #[test]
    fn order_by_sorts_numeric_field() {
        let f = library();
        let results = f.query().where_node_type(vec!["book".into()]).order_by("rating", OrderDirection::Desc).to_array();
        assert_eq!(results.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(), vec!["scifi1", "fantasy1"]);
    }

    #[test]
    fn limit_and_offset_paginate() {
        let f = library();
        let results = f
            .query()
            .where_node_type(vec!["book".into()])
            .order_by("rating", OrderDirection::Asc)
            .offset(1)
            .limit(1)
            .to_array();
        assert_eq!(ids_of(results), vec!["scifi1"]);
    }

    #[test]
    fn traverse_expands_via_bfs() {
        let f = library();
        let expanded = f
            .query()
            .where_node_type(vec!["user".into()])
            .where_field("name", serde_json::json!("Alice"))
            .traverse("RATED", 1, Direction::Out);
        assert_eq!(ids_of(expanded.to_array()), vec!["alice", "fantasy1", "scifi1"]);
    }

    #[test]
    fn similar_to_ranks_by_cosine_and_respects_threshold_and_top_k() {
        let f = library();
        let results = f.query().where_node_type(vec!["book".into()]).similar_to(vec![1.0, 0.0], 0.5, Some(1)).to_array();
        assert_eq!(ids_of(results), vec!["scifi1"], "only scifi1's vector clears the 0.5 cosine threshold");
    }

    #[test]
    fn join_filters_by_a_predicate_on_the_connected_node() {
        let f = library();
        let results = f
            .query()
            .where_node_type(vec!["user".into()])
            .join(
                "RATED",
                Direction::Out,
                Some(Box::new(|n: &Node| n.data.get("genre") == Some(&serde_json::json!("fantasy")))),
            )
            .to_array();
        assert_eq!(ids_of(results), vec!["alice"], "only alice rated a fantasy book");
    }

    #[test]
    fn join_without_a_predicate_requires_any_connection() {
        let f = library();
        let results = f.query().where_node_type(vec!["user".into()]).join("RATED", Direction::Out, None).to_array();
        assert_eq!(ids_of(results), vec!["alice", "bob"]);
    }

    #[test]
    fn first_count_and_ids() {
        let f = library();
        let q = f.query().where_node_type(vec!["book".into()]);
        assert_eq!(q.count(), 2);
        let mut ids = q.ids();
        ids.sort();
        assert_eq!(ids, vec!["fantasy1".to_string(), "scifi1".to_string()]);
        assert!(f.query().where_node_type(vec!["book".into()]).first().is_some());
        assert!(f.query().where_node_type(vec!["missing".into()]).first().is_none());
    }

    #[test]
    fn pluck_projects_requested_fields() {
        let f = library();
        let mut rows = f.query().where_node_type(vec!["book".into()]).pluck(&["genre"]);
        rows.sort_by(|a, b| a["id"].as_str().cmp(&b["id"].as_str()));
        assert_eq!(rows[0]["id"], serde_json::json!("fantasy1"));
        assert_eq!(rows[0]["type"], serde_json::json!("book"));
        assert_eq!(rows[0]["genre"], serde_json::json!("fantasy"));
    }

    #[test]
    fn aggregate_sums_a_field() {
        let f = library();
        let result = f.query().where_node_type(vec!["book".into()]).aggregate("rating", AggregateOp::Sum);
        assert_eq!(result, AggregateResult { value: 8.0, count: 2 });
    }

    #[test]
    fn group_aggregate_groups_then_aggregates() {
        let f = library();
        let mut groups = f.query().where_node_type(vec!["book".into()]).group_aggregate("rating", AggregateOp::Sum, "genre");
        groups.sort_by(|a, b| a.key.cmp(&b.key));
        assert_eq!(
            groups,
            vec![
                GroupedRow { key: "fantasy".into(), value: 3.0, count: 1 },
                GroupedRow { key: "sci-fi".into(), value: 5.0, count: 1 },
            ]
        );
    }

    #[test]
    fn having_filters_grouped_rows() {
        let groups = vec![
            GroupedRow { key: "a".into(), value: 1.0, count: 1 },
            GroupedRow { key: "b".into(), value: 10.0, count: 1 },
        ];
        let filtered = GraphQuery::having(groups, |g| g.value > 5.0);
        assert_eq!(filtered, vec![GroupedRow { key: "b".into(), value: 10.0, count: 1 }]);
    }

    #[test]
    fn group_by_vector_clusters_by_nearest_centroid() {
        let f = library();
        let groups = vec![("scifi".to_string(), vec![1.0, 0.0]), ("fantasy".to_string(), vec![0.0, 1.0])];
        let mut rows =
            f.query().where_node_type(vec!["book".into()]).group_by_vector(&groups, "rating", AggregateOp::Sum, 0.5);
        rows.sort_by(|a, b| a.key.cmp(&b.key));
        assert_eq!(
            rows,
            vec![
                GroupedRow { key: "fantasy".into(), value: 3.0, count: 1 },
                GroupedRow { key: "scifi".into(), value: 5.0, count: 1 },
            ]
        );
    }

    #[test]
    fn unique_keys_ignores_all_filters() {
        let f = library();
        let mut keys = f.query().where_node_type(vec!["missing".into()]).unique_keys("genre");
        keys.sort_by_key(|v| v.to_string());
        assert_eq!(keys, vec![serde_json::json!("fantasy"), serde_json::json!("sci-fi")]);
    }

    #[test]
    fn collect_gathers_distinct_connected_nodes() {
        let f = library();
        let mut ids = ids_of(f.query().where_node_type(vec!["user".into()]).collect("RATED", Direction::Out, None));
        ids.sort();
        assert_eq!(ids, vec!["fantasy1".to_string(), "scifi1".to_string()]);
    }

    #[test]
    fn where_activated_filters_by_decayed_score() {
        let mut f = Fixture::new();
        let now = now_millis();
        for (id, score) in [("a", 0.9), ("b", 0.5), ("c", 0.1)] {
            let mut n = node(id, "t", &[]);
            n.activation = Some(NodeActivation {
                score,
                importance: 0.0,
                reinforcement_count: 1,
                last_meaningful_activation: now,
                ..Default::default()
            });
            f.add(n);
        }
        let mut ids = f.query().where_activated(0.4).ids();
        ids.sort();
        assert_eq!(ids, vec!["a".to_string(), "b".to_string()]);
    }

    #[test]
    fn order_by_activation_sorts_descending_and_ascending() {
        let mut f = Fixture::new();
        let now = now_millis();
        for (id, score) in [("a", 0.9), ("b", 0.5), ("c", 0.1)] {
            let mut n = node(id, "t", &[]);
            n.activation = Some(NodeActivation {
                score,
                importance: 0.0,
                reinforcement_count: 1,
                last_meaningful_activation: now,
                ..Default::default()
            });
            f.add(n);
        }
        assert_eq!(
            f.query().order_by_activation(OrderDirection::Desc).ids(),
            vec!["a".to_string(), "b".to_string(), "c".to_string()]
        );
        assert_eq!(
            f.query().order_by_activation(OrderDirection::Asc).ids(),
            vec!["c".to_string(), "b".to_string(), "a".to_string()]
        );
    }
}
