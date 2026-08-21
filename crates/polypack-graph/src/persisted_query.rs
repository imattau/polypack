//! [`PersistedGraphQuery`]: the Rust counterpart to `PersistedGraphQuery`
//! (`src/persisted-query.ts`).
//!
//! A fluent, chainable query over every persisted node via a `Store`,
//! without loading results into `Graph`'s hot working set. TS's version
//! targets a generic `PersistenceAdapter` interface with several optional
//! capability hooks (`queryNodes`, `getEdgesBySources`, `countNodes`, ...),
//! falling back to full scans when absent. `Store` always implements the
//! equivalent methods directly, so there's no capability branching here —
//! every path always uses the `Store` method.

use std::collections::{HashMap, HashSet};
use std::cell::RefCell;
use std::time::Instant;

use polypack_core::activation::{activation_score_of, DEFAULT_ACTIVATION};
use polypack_core::query::Direction;
use polypack_core::storage::{NodeQuery, OrderBy, RangeQuery};
use polypack_core::vector::cosine;
use polypack_core::{Node, Result, Store};

use crate::graph::{now_millis, IndexDefinition, QueryMetrics};
use crate::query::OrderDirection;

/// A `join`/`join`-predicate closure: `Some` to filter by the connected
/// node, `None` to require only that a connection exists.
type JoinPredicate<'a> = Box<dyn Fn(&Node) -> bool + 'a>;

#[derive(Clone, Debug)]
struct SimilaritySpec {
    vector: Vec<f64>,
    threshold: f64,
    top_k: Option<usize>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct QueryExplain {
    pub index: Option<String>,
    pub indexes: Vec<String>,
    pub stages: Vec<String>,
    pub loaded_records: usize,
    pub estimated_cost: f64,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct QueryResourceLimits {
    pub max_traversal_depth: Option<usize>,
    pub max_nodes_visited: Option<usize>,
    pub max_results: Option<usize>,
}

fn query_field_value(attributes: &HashMap<String, serde_json::Value>, field: &str) -> serde_json::Value {
    if let Some(value) = attributes.get(field) {
        return value.clone();
    }
    attributes
        .get(field.strip_prefix("data.").unwrap_or(field))
        .cloned()
        .unwrap_or(serde_json::Value::Null)
}

/// Chainable query over all persisted nodes in a `Store`. Results are
/// detached clones. Mirrors `PersistedGraphQuery`.
pub struct PersistedGraphQuery<'a> {
    store: &'a mut Store,
    indexes: &'a HashMap<String, IndexDefinition>,
    secondary_indexes: &'a HashMap<String, HashMap<String, HashSet<String>>>,
    metrics: &'a RefCell<QueryMetrics>,

    node_types: Option<Vec<String>>,
    attributes: HashMap<String, serde_json::Value>,
    attribute_ranges: HashMap<String, (Option<f64>, Option<f64>)>,
    order_by: Option<(String, OrderDirection)>,
    result_offset: Option<usize>,
    result_limit: Option<usize>,
    similarity: Option<SimilaritySpec>,

    edge_type: Option<String>,
    edge_target: Option<String>,
    edge_source: Option<String>,
    joins: Vec<(String, Direction, Option<JoinPredicate<'a>>)>,
    traversals: Vec<(String, usize, Direction)>,
    activation_above: Option<f64>,
    activation_order: Option<OrderDirection>,
    limits: QueryResourceLimits,
}

impl<'a> PersistedGraphQuery<'a> {
    pub(crate) fn new(
        store: &'a mut Store,
        indexes: &'a HashMap<String, IndexDefinition>,
        secondary_indexes: &'a HashMap<String, HashMap<String, HashSet<String>>>,
        metrics: &'a RefCell<QueryMetrics>,
    ) -> Self {
        Self {
            store,
            indexes,
            secondary_indexes,
            metrics,
            node_types: None,
            attributes: HashMap::new(),
            attribute_ranges: HashMap::new(),
            order_by: None,
            result_offset: None,
            result_limit: None,
            similarity: None,
            edge_type: None,
            edge_target: None,
            edge_source: None,
            joins: Vec::new(),
            traversals: Vec::new(),
            activation_above: None,
            activation_order: None,
            limits: QueryResourceLimits::default(),
        }
    }

    // ── builder methods ──

    pub fn where_field(mut self, field: &str, value: serde_json::Value) -> Self {
        self.attributes.insert(field.to_string(), value);
        self
    }

    pub fn where_attribute(self, name: &str, value: serde_json::Value) -> Self {
        self.where_field(name, value)
    }

    pub fn where_attribute_range(mut self, name: &str, above: Option<f64>, below: Option<f64>) -> Self {
        self.attribute_ranges.insert(name.to_string(), (above, below));
        self
    }

    pub fn where_node_type(mut self, types: Vec<String>) -> Self {
        self.node_types = Some(types);
        self
    }

    pub fn where_edge(mut self, edge_type: &str, target: Option<&str>) -> Self {
        self.edge_type = Some(edge_type.to_string());
        self.edge_target = target.map(|t| t.to_string());
        self
    }

    pub fn where_edge_source(mut self, source: &str) -> Self {
        self.edge_source = Some(source.to_string());
        self
    }

    pub fn join(
        mut self,
        edge_type: &str,
        direction: Direction,
        predicate: Option<JoinPredicate<'a>>,
    ) -> Self {
        self.joins.push((edge_type.to_string(), direction, predicate));
        self
    }

    pub fn traverse(mut self, edge_type: &str, depth: usize, direction: Direction) -> Self {
        self.traversals.push((edge_type.to_string(), depth, direction));
        self
    }

    pub fn order_by(mut self, field: &str, direction: OrderDirection) -> Self {
        self.order_by = Some((field.to_string(), direction));
        self
    }

    /// Keep only nodes whose current (decay-corrected) activation exceeds
    /// `above`. Mirrors `PersistedGraphQuery.whereActivated`.
    pub fn where_activated(mut self, above: f64) -> Self {
        self.activation_above = Some(above);
        self
    }

    /// Order results by current (decay-corrected) activation instead of a data
    /// field. Mirrors `PersistedGraphQuery.orderByActivation`.
    pub fn order_by_activation(mut self, direction: OrderDirection) -> Self {
        self.activation_order = Some(direction);
        self
    }

    pub fn offset(mut self, n: usize) -> Self {
        self.result_offset = Some(n);
        self
    }

    pub fn limit(mut self, n: usize) -> Self {
        self.result_limit = Some(n);
        self
    }

    pub fn similar_to(mut self, vector: Vec<f64>, threshold: f64, top_k: Option<usize>) -> Self {
        self.similarity = Some(SimilaritySpec { vector, threshold, top_k });
        self
    }

    pub fn with_resource_limits(mut self, limits: QueryResourceLimits) -> Self {
        self.limits = limits;
        self
    }

    /// Describe the selected persisted-query stages without materializing rows.
    pub fn explain(&mut self) -> Result<QueryExplain> {
        let loaded_records = self.store.node_count()?;
        let mut stages = Vec::new();
        let selected = self.selected_indexes();
        let indexes: Vec<String> = selected.iter().map(|definition| definition.name.clone()).collect();
        let index = indexes.first().cloned().or_else(|| {
            self.node_types.as_ref().filter(|types| !types.is_empty()).map(|_| "type-index".to_string())
        });
        if indexes.is_empty() {
            stages.push({
            if index.as_deref() == Some("type-index") { "type-index".to_string() } else { "record-scan".to_string() }
            });
        } else {
            stages.extend(indexes.iter().map(|name| format!("property-index({name})")));
        }
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
        if !self.joins.is_empty() {
            stages.push(format!("join(count={})", self.joins.len()));
        }
        if !self.traversals.is_empty() {
            let depth = self.traversals.iter().map(|(_, depth, _)| *depth).max().unwrap_or(0);
            stages.push(format!("traversal(depth={depth})"));
        }
        if let Some((field, direction)) = &self.order_by {
            stages.push(format!("order({field},{direction:?})").to_lowercase());
        }
        if let Some(limit) = self.result_limit {
            stages.push(format!("limit({limit})"));
        }
        Ok(QueryExplain {
            indexes: indexes.clone(),
            index,
            stages,
            loaded_records,
            estimated_cost: (loaded_records as f64 * if indexes.is_empty() { 1.0 } else { 0.25 / indexes.len() as f64 }).max(1.0),
        })
    }

    // ── internals ──

    fn base_query(&self, include_order: bool, include_pagination: bool) -> NodeQuery {
        let candidate_ids = {
            let mut sets = Vec::new();
            for definition in self.selected_indexes() {
                let equality = definition.fields.iter().all(|field| self.attributes.contains_key(field) || self.attributes.contains_key(field.strip_prefix("data.").unwrap_or(field)));
                if equality {
                    let values = definition.fields.iter().map(|field| query_field_value(&self.attributes, field)).collect::<Vec<_>>();
                    if definition.sparse && values.iter().any(serde_json::Value::is_null) { continue; }
                    let Some(key) = serde_json::to_string(&values).ok() else { continue };
                    sets.push(self.secondary_indexes.get(&definition.name).and_then(|buckets| buckets.get(&key)).cloned().unwrap_or_default());
                } else if definition.fields.len() == 1 {
                    let Some(range) = self.attribute_ranges.get(&definition.fields[0]) else { continue };
                    let ids = self.secondary_indexes.get(&definition.name).into_iter().flat_map(|buckets| buckets.iter()).filter_map(|(encoded, bucket)| {
                        let value = serde_json::from_str::<Vec<serde_json::Value>>(encoded).ok()?.first()?.as_f64()?;
                        if range.0.is_some_and(|above| value <= above) || range.1.is_some_and(|below| value >= below) { return None; }
                        Some(bucket.iter().cloned())
                    }).flatten().collect();
                    sets.push(ids);
                }
            }
            sets.into_iter().reduce(|mut ids, candidate| { ids.retain(|id| candidate.contains(id)); ids }).map(|ids| ids.into_iter().collect())
        };
        NodeQuery {
            candidate_ids,
            node_types: self.node_types.clone(),
            attributes: (!self.attributes.is_empty()).then(|| self.attributes.clone().into_iter().collect()),
            attribute_ranges: (!self.attribute_ranges.is_empty()).then(|| {
                self.attribute_ranges
                    .iter()
                    .map(|(k, (above, below))| (k.clone(), RangeQuery { above: *above, below: *below }))
                    .collect()
            }),
            order_by: include_order.then(|| self.order_by.clone()).flatten().map(|(field, dir)| OrderBy {
                field,
                direction: if dir == OrderDirection::Desc { "desc".to_string() } else { "asc".to_string() },
            }),
            offset: if include_pagination { self.result_offset } else { None },
            limit: if include_pagination { self.result_limit } else { None },
            max_nodes_visited: self.limits.max_nodes_visited,
            max_result_size: self.limits.max_results,
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

    fn apply_edge_filters(&mut self, nodes: Vec<Node>) -> Result<Vec<Node>> {
        let mut results = nodes;
        if let Some(edge_type) = self.edge_type.clone() {
            let ids: Vec<String> = results.iter().map(|n| n.id.clone()).collect();
            let edges = self.store.get_edges_by_sources(&ids, Some(&edge_type))?;
            let target = self.edge_target.clone();
            let sources: HashSet<String> = edges
                .into_iter()
                .filter(|e| target.as_deref().is_none_or(|t| e.target == t))
                .map(|e| e.source)
                .collect();
            results.retain(|n| sources.contains(&n.id));
        }
        if let Some(source) = self.edge_source.clone() {
            let edges = self.store.get_edges_by_sources(&[source], self.edge_type.as_deref())?;
            let targets: HashSet<String> = edges.into_iter().map(|e| e.target).collect();
            results.retain(|n| targets.contains(&n.id));
        }
        Ok(results)
    }

    fn apply_joins(&mut self, nodes: Vec<Node>) -> Result<Vec<Node>> {
        let mut results = nodes;
        for i in 0..self.joins.len() {
            let (edge_type, direction, predicate) = &self.joins[i];
            let direction = *direction;
            let ids: Vec<String> = results.iter().map(|n| n.id.clone()).collect();
            let edges = match direction {
                Direction::Out => self.store.get_edges_by_sources(&ids, Some(edge_type))?,
                Direction::In => self.store.get_edges_by_targets(&ids, Some(edge_type))?,
            };
            let connected_ids: HashSet<String> = edges
                .iter()
                .map(|e| match direction { Direction::Out => e.target.clone(), Direction::In => e.source.clone() })
                .collect();
            let mut connected: HashMap<String, Node> = HashMap::new();
            for id in &connected_ids {
                if let Some(n) = self.store.get_node(id)? {
                    connected.insert(id.clone(), n);
                }
            }
            let mut matched = HashSet::new();
            for e in &edges {
                let (candidate_id, connected_id) = match direction {
                    Direction::Out => (&e.source, &e.target),
                    Direction::In => (&e.target, &e.source),
                };
                if let Some(node) = connected.get(connected_id) {
                    let ok = match predicate { Some(p) => p(node), None => true };
                    if ok {
                        matched.insert(candidate_id.clone());
                    }
                }
            }
            results.retain(|n| matched.contains(&n.id));
        }
        Ok(results)
    }

    fn apply_traversals(&mut self, nodes: Vec<Node>) -> Result<Vec<Node>> {
        let mut current_ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
        for i in 0..self.traversals.len() {
            let (edge_type, depth, direction) = self.traversals[i].clone();
            if let Some(max_depth) = self.limits.max_traversal_depth {
                if depth > max_depth {
                    return Err(polypack_core::PolypackError::ResourceLimit { name: "maxTraversalDepth".into(), limit: max_depth });
                }
            }
            let mut visited: HashSet<String> = current_ids.iter().cloned().collect();
            if let Some(max_nodes) = self.limits.max_nodes_visited {
                if visited.len() > max_nodes {
                    return Err(polypack_core::PolypackError::ResourceLimit { name: "maxNodesVisited".into(), limit: max_nodes });
                }
            }
            let mut frontier = current_ids.clone();
            for _ in 0..depth {
                if frontier.is_empty() {
                    break;
                }
                let edges = match direction {
                    Direction::Out => self.store.get_edges_by_sources(&frontier, Some(&edge_type))?,
                    Direction::In => self.store.get_edges_by_targets(&frontier, Some(&edge_type))?,
                };
                let mut next = Vec::new();
                for e in &edges {
                    let id = match direction { Direction::Out => e.target.clone(), Direction::In => e.source.clone() };
                    if visited.insert(id.clone()) {
                        if let Some(max_nodes) = self.limits.max_nodes_visited {
                            if visited.len() > max_nodes {
                                return Err(polypack_core::PolypackError::ResourceLimit { name: "maxNodesVisited".into(), limit: max_nodes });
                            }
                        }
                        next.push(id);
                    }
                }
                frontier = next;
            }
            current_ids = visited.into_iter().collect();
        }
        let mut out = Vec::with_capacity(current_ids.len());
        for id in current_ids {
            if let Some(n) = self.store.get_node(&id)? {
                out.push(n);
            }
        }
        Ok(out)
    }

    // ── terminals ──

    /// Materialize matched nodes. Mirrors `PersistedGraphQuery.toArray`.
    pub fn to_array(&mut self) -> Result<Vec<Node>> {
        let started = Instant::now();
        let has_traversal = !self.traversals.is_empty();
        let adapter_can_paginate =
            self.similarity.is_none() && self.edge_type.is_none() && self.edge_source.is_none() && self.joins.is_empty() && !has_traversal
                && self.activation_above.is_none() && self.activation_order.is_none();

        let query = self.base_query(!has_traversal, adapter_can_paginate);
        let mut nodes = self.store.query_nodes(&query)?;
        let scanned_records = nodes.len();

        nodes = self.apply_edge_filters(nodes)?;
        nodes = self.apply_joins(nodes)?;
        if has_traversal {
            nodes = self.apply_traversals(nodes)?;
        }

        if let Some(above) = self.activation_above {
            let now = now_millis();
            nodes.retain(|n| activation_score_of(n, now, DEFAULT_ACTIVATION.score_half_life_ms) >= above);
        }
        if let Some(direction) = self.activation_order {
            let now = now_millis();
            nodes.sort_by(|a, b| {
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

        if has_traversal {
            if let Some((field, direction)) = &self.order_by {
                let desc = *direction == OrderDirection::Desc;
                nodes.sort_by(|a, b| {
                    let av = a.data.get(field).and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let bv = b.data.get(field).and_then(|v| v.as_f64()).unwrap_or(0.0);
                    let ord = av.partial_cmp(&bv).unwrap_or(std::cmp::Ordering::Equal);
                    if desc { ord.reverse() } else { ord }
                });
            }
        }

        if let Some(sim) = self.similarity.clone() {
            let mut scored: Vec<(f64, Node)> = nodes
                .into_iter()
                .filter_map(|n| {
                    let v = n.vector.clone()?;
                    let score = cosine(&sim.vector, &v).ok()?;
                    (score >= sim.threshold).then_some((score, n))
                })
                .collect();
            scored.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            if let Some(top_k) = sim.top_k {
                scored.truncate(top_k);
            }
            nodes = scored.into_iter().map(|(_, n)| n).collect();
        }

        if !adapter_can_paginate {
            if let Some(offset) = self.result_offset {
                nodes = nodes.into_iter().skip(offset).collect();
            }
            if let Some(limit) = self.result_limit {
                nodes.truncate(limit);
            }
        }

        if let Some(max_results) = self.limits.max_results {
            if nodes.len() > max_results {
                return Err(polypack_core::PolypackError::ResourceLimit { name: "maxResults".into(), limit: max_results });
            }
        }
        self.record_metrics(started, scanned_records);
        Ok(nodes)
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

    pub fn first(&mut self) -> Result<Option<Node>> {
        Ok(self.to_array()?.into_iter().next())
    }

    pub fn ids(&mut self) -> Result<Vec<String>> {
        Ok(self.to_array()?.into_iter().map(|n| n.id).collect())
    }

    pub fn count(&mut self) -> Result<usize> {
        let needs_materialization = self.similarity.is_some()
            || self.result_offset.is_some()
            || self.result_limit.is_some()
            || self.edge_type.is_some()
            || self.edge_source.is_some()
            || !self.joins.is_empty()
            || !self.traversals.is_empty()
            || self.activation_above.is_some()
            || self.activation_order.is_some();
        if needs_materialization {
            return Ok(self.to_array()?.len());
        }
        let query = self.base_query(false, false);
        self.store.count_nodes(&query)
    }

    /// From the matched nodes, collect distinct connected nodes reachable
    /// via one `edge_type` hop, optionally filtered by `predicate`.
    pub fn collect(
        &mut self,
        edge_type: &str,
        direction: Direction,
        predicate: Option<&dyn Fn(&Node) -> bool>,
    ) -> Result<Vec<Node>> {
        let seeds = self.to_array()?;
        let seed_ids: Vec<String> = seeds.iter().map(|n| n.id.clone()).collect();
        let edges = match direction {
            Direction::Out => self.store.get_edges_by_sources(&seed_ids, Some(edge_type))?,
            Direction::In => self.store.get_edges_by_targets(&seed_ids, Some(edge_type))?,
        };
        let connected_ids: HashSet<String> = edges
            .iter()
            .map(|e| match direction { Direction::Out => e.target.clone(), Direction::In => e.source.clone() })
            .collect();
        let mut out = Vec::new();
        for id in connected_ids {
            if let Some(n) = self.store.get_node(&id)? {
                if predicate.is_none_or(|p| p(&n)) {
                    out.push(n);
                }
            }
        }
        Ok(out)
    }
}
