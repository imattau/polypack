//! [`ActivationEngine`]: the adaptive-memory layer, the Rust counterpart to
//! `src/activation.ts` in the TypeScript package.
//!
//! Two tiers:
//! - **Durable** — `reinforce`/`reinforce_all` write `NodeActivation` through
//!   `Graph::reinforce_node`, so it persists (via the core `Store`) and, in
//!   deployments with a sync layer, replicates additively.
//! - **Transient** — `bump_attention`/`attention_of` hold runtime-only
//!   attention in this engine, never serialized. `effective` merges the two.
//!
//! `spread` implements graph-based spreading activation; `pulse`/`absorb`
//! implement the semantic + relational + recency + usage composite; and
//! `working_memory` materializes the current "mental state" (most active
//! nodes). Decay math comes from `polypack_core::activation`.

use std::collections::{HashMap, HashSet};

use polypack_core::activation::{clamp01, decay_activation_state, decay_factor, DEFAULT_ACTIVATION};
use polypack_core::vector::cosine;
use polypack_core::{Node, PolypackError, Result};

use crate::graph::{now_millis, Graph};

const HOUR: i64 = 3_600_000;
const DAY: i64 = 24 * HOUR;

/// Tuning knobs for the engine, field-for-field with the TypeScript
/// `ActivationConfig`.
#[derive(Clone, Debug)]
pub struct ActivationConfig {
    /// Half-life of the short-term `score` curve. Default 24 h.
    pub score_half_life_ms: i64,
    /// Half-life of the long-term `importance` curve. Default 30 days.
    pub importance_half_life_ms: i64,
    /// How much of a reinforcement delta is folded into `importance` (0–1). Default 0.05.
    pub importance_gain: f64,
    /// Per-hop attenuation for spreading activation. Default 0.5.
    pub spread_decay: f64,
    /// Maximum hop distance for spreading activation. Default 2.
    pub spread_depth: usize,
    /// Half-life used to fold a node's age into a composite pulse score. Default 7 days.
    pub recency_half_life_ms: i64,
    /// Weights for the composite pulse score. Defaults to all 1.
    pub weights: ActivationWeights,
    /// Attention changes below this magnitude stay local and never become
    /// durable reinforcement. Default 0.05.
    pub min_reinforce_delta: f64,
    /// Minimum composite score kept by `pulse`. Default 0.
    pub pulse_threshold: f64,
    /// Composite score at which `absorb` reinforces a node. Default 0.3.
    pub absorb_threshold: f64,
    /// Reinforcement delta applied by `absorb` (`gain * composite`). Default 0.05.
    pub absorb_gain: f64,
}

/// Weights for the composite pulse score.
#[derive(Clone, Copy, Debug)]
pub struct ActivationWeights {
    pub semantic: f64,
    pub graph: f64,
    pub recency: f64,
    pub usage: f64,
}

impl Default for ActivationWeights {
    fn default() -> Self {
        Self { semantic: 1.0, graph: 1.0, recency: 1.0, usage: 1.0 }
    }
}

impl Default for ActivationConfig {
    fn default() -> Self {
        Self {
            score_half_life_ms: DEFAULT_ACTIVATION.score_half_life_ms,
            importance_half_life_ms: DEFAULT_ACTIVATION.importance_half_life_ms,
            importance_gain: DEFAULT_ACTIVATION.importance_gain,
            spread_decay: 0.5,
            spread_depth: 2,
            recency_half_life_ms: 7 * DAY,
            weights: ActivationWeights::default(),
            min_reinforce_delta: 0.05,
            pulse_threshold: 0.0,
            absorb_threshold: 0.3,
            absorb_gain: 0.05,
        }
    }
}

/// Options for [`ActivationEngine::spread`].
#[derive(Clone, Debug, Default)]
pub struct SpreadOptions {
    pub depth: Option<usize>,
    pub decay: Option<f64>,
    /// Restrict traversal to these edge types. Default: all edge types.
    pub edge_types: Option<Vec<String>>,
}

/// Options for [`ActivationEngine::pulse`].
#[derive(Clone, Debug, Default)]
pub struct PulseOptions {
    pub depth: Option<usize>,
    pub decay: Option<f64>,
    pub edge_types: Option<Vec<String>>,
    /// Cap on the number of semantic seeds. Default: all loaded vectors.
    pub top_k: Option<usize>,
    /// Minimum composite score kept by `pulse`. Defaults to `config.pulse_threshold`.
    pub pulse_threshold: Option<f64>,
    /// Minimum semantic similarity for a node to seed the activated region.
    /// Nodes with zero similarity never enter the region.
    pub semantic_threshold: Option<f64>,
}

/// Options for [`ActivationEngine::working_memory_with_options`].
pub struct WorkingMemoryOptions {
    /// Maximum nodes to return.
    pub limit: usize,
    /// Discard candidates whose `effective` score is at or below this.
    pub min_score: f64,
    /// Rank/select within this context (see `ActivationEngine::effective`) instead of the global score.
    pub context: Option<String>,
    /// Sum of `cost_of(node)` across selected nodes must not exceed this. Default: unbounded.
    pub token_budget: Option<f64>,
    /// Per-node cost against `token_budget`. The engine has no tokenizer — callers own this. Default: 1 per node.
    pub cost_of: Box<dyn Fn(&Node) -> f64>,
    /// MMR trade-off: 0 = pure relevance (default), 1 = pure diversity.
    pub diversity_lambda: f64,
    /// Redundancy metric between two candidate nodes. Default: cosine similarity of `node.vector` (0 if either lacks one).
    pub similarity_of: Box<dyn Fn(&Node, &Node) -> f64>,
}

impl Default for WorkingMemoryOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            min_score: 0.0,
            context: None,
            token_budget: None,
            cost_of: Box::new(|_| 1.0),
            diversity_lambda: 0.0,
            similarity_of: Box::new(default_similarity),
        }
    }
}

/// Cosine similarity between two nodes' vectors, or 0 when either lacks one.
fn default_similarity(a: &Node, b: &Node) -> f64 {
    match (&a.vector, &b.vector) {
        (Some(av), Some(bv)) => cosine(av, bv).unwrap_or(0.0),
        _ => 0.0,
    }
}

/// Effective activation of a node: its durable decayed score (using
/// `score_half_life_ms`, or the context-scoped score when `context` is given)
/// plus any transient attention, minus decayed inhibition. Free function so
/// the engine can compute it without double-borrowing the graph.
fn effective_activation(node: &Node, attention: f64, now: i64, score_half_life_ms: i64, context: Option<&str>) -> f64 {
    let decayed = node
        .activation
        .as_ref()
        .map(|a| decay_activation_state(a, now, score_half_life_ms, DEFAULT_ACTIVATION.importance_half_life_ms));
    let base = match (&decayed, context) {
        (Some(d), Some(ctx)) => d.context.as_ref().and_then(|c| c.get(ctx)).map(|e| e.score).unwrap_or(0.0),
        (Some(d), None) => d.score,
        (None, _) => 0.0,
    };
    let inhibition = decayed.as_ref().and_then(|d| d.inhibition).unwrap_or(0.0);
    clamp01(base + attention - inhibition)
}

/// Composite score for a node given a semantic hit and a graph contribution.
fn composite_score(
    node: &Node,
    semantic: f64,
    graph_contribution: f64,
    now: i64,
    weights: &ActivationWeights,
    recency_half_life_ms: i64,
) -> f64 {
    let recency = decay_factor(now - node.inserted_at, recency_half_life_ms);
    let usage = node.activation.as_ref().map(|a| a.importance).unwrap_or(0.0);
    weights.semantic * semantic
        + weights.graph * graph_contribution
        + weights.recency * recency
        + weights.usage * usage
}

/// Composes the adaptive "activation" layer over a [`Graph`], mirroring the
/// TypeScript `ActivationEngine`. Holds a mutable borrow of the graph (same
/// pattern as `GraphQuery`).
pub struct ActivationEngine<'a> {
    graph: &'a mut Graph,
    config: ActivationConfig,
    attention: HashMap<String, f64>,
}

impl<'a> ActivationEngine<'a> {
    pub fn new(graph: &'a mut Graph, config: ActivationConfig) -> Self {
        Self { graph, config, attention: HashMap::new() }
    }

    /// Drop transient attention. Durable state is untouched.
    pub fn dispose(&mut self) {
        self.attention.clear();
    }

    /// Access the configured engine parameters.
    pub fn config(&self) -> &ActivationConfig {
        &self.config
    }

    // ── transient attention (local, never synced) ──

    /// Add a runtime-only attention delta. Attention accumulates locally and is
    /// promoted to durable reinforcement once it clears `min_reinforce_delta`,
    /// so tiny events (scrolls, focus) stay local while meaningful ones become
    /// persisted and synced.
    pub fn bump_attention(&mut self, id: &str, amount: f64) {
        let next = clamp01(self.attention.get(id).copied().unwrap_or(0.0) + amount);
        self.attention.insert(id.to_string(), next);
        if next >= self.config.min_reinforce_delta {
            let promoted = next;
            self.attention.remove(id);
            let _ = self.graph.reinforce_node(id, promoted, Some("attention"));
        }
    }

    /// Current transient attention for `id` (0 when none).
    pub fn attention_of(&self, id: &str) -> f64 {
        self.attention.get(id).copied().unwrap_or(0.0)
    }

    /// Durable decayed score plus any transient attention, minus decayed
    /// inhibition. When `context` is given, the context-scoped score is used
    /// instead of the global score (0 if the node has no history in that
    /// context). Inhibition is applied only here, never inside `pulse`'s
    /// composite, so a suppressed node stays re-evaluable.
    pub fn effective(&mut self, id: &str, context: Option<&str>) -> f64 {
        let now = now_millis();
        let attention = self.attention.get(id).copied().unwrap_or(0.0);
        match self.graph.get_node(id) {
            Some(node) => effective_activation(node, attention, now, self.config.score_half_life_ms, context),
            None => 0.0,
        }
    }

    /// Current decayed inhibition for `id` (0 when none).
    pub fn inhibition_of(&self, id: &str) -> f64 {
        self.graph.get_activation_state(id).and_then(|a| a.inhibition).unwrap_or(0.0)
    }

    // ── durable reinforcement (persisted and synced additively) ──

    /// Reinforce a loaded node's durable activation. See `Graph::reinforce_node`.
    pub fn reinforce(&mut self, id: &str, amount: f64, reason: Option<&str>) -> Result<Option<&Node>> {
        self.graph.reinforce_node(id, amount, reason)
    }

    /// Reinforce a loaded node's durable activation within `context`. See
    /// `Graph::reinforce_node_in_context`.
    pub fn reinforce_in_context(&mut self, id: &str, amount: f64, reason: Option<&str>, context: &str) -> Result<Option<&Node>> {
        self.graph.reinforce_node_in_context(id, amount, reason, Some(context))
    }

    /// Suppress a loaded node's durable `inhibition`. A negative `amount`
    /// releases suppression. See `Graph::suppress_node`.
    pub fn suppress(&mut self, id: &str, amount: f64, reason: Option<&str>) -> Result<Option<&Node>> {
        self.graph.suppress_node(id, amount, reason)
    }

    /// Reinforce several nodes, coalescing change events into one batch.
    pub fn reinforce_all(&mut self, entries: &[(String, f64, Option<String>)]) -> Result<()> {
        if entries.is_empty() {
            return Ok(());
        }
        self.graph.start_batch();
        let mut err: Option<PolypackError> = None;
        for (id, amount, reason) in entries {
            if let Err(e) = self.graph.reinforce_node(id, *amount, reason.as_deref()) {
                err = Some(e);
                break;
            }
        }
        self.graph.end_batch()?;
        match err {
            Some(e) => Err(e),
            None => Ok(()),
        }
    }

    // ── relational spreading activation ──

    /// Spread activation outward from `seeds` across outgoing edges. Each hop
    /// attenuates the contribution by `decay` (default `config.spread_decay`),
    /// and multiple paths to a node sum. Returns `{ nodeId: contribution }`.
    pub fn spread(&self, seeds: &[String], options: &SpreadOptions) -> HashMap<String, f64> {
        let depth = options.depth.unwrap_or(self.config.spread_depth);
        let decay = options.decay.unwrap_or(self.config.spread_decay);
        let mut contributions: HashMap<String, f64> = HashMap::new();
        if depth == 0 || seeds.is_empty() {
            return contributions;
        }
        let mut visited: HashSet<String> = seeds.iter().cloned().collect();
        let mut frontier: Vec<String> = seeds.to_vec();
        for hop in 0..depth {
            if frontier.is_empty() {
                break;
            }
            let mut next = Vec::new();
            for id in &frontier {
                for edge in self.graph.get_edges(id, None) {
                    if options
                        .edge_types
                        .as_ref()
                        .is_some_and(|types| !types.iter().any(|t| t == &edge.edge_type))
                    {
                        continue;
                    }
                    if visited.insert(edge.target.clone()) {
                        let contribution = decay.powf(hop as f64 + 1.0);
                        *contributions.entry(edge.target.clone()).or_insert(0.0) += contribution;
                        next.push(edge.target.clone());
                    }
                }
            }
            frontier = next;
        }
        contributions
    }

    // ── semantic pulse / absorb ──

    /// Score the region of the graph around `vector`: semantic seeds via
    /// vector similarity (nodes with zero similarity never seed the region),
    /// outward spreading activation, folded with recency and usage. Read-only —
    /// returns `{ nodeId: composite }` ranked descending for loaded nodes.
    pub fn pulse(&mut self, vector: &[f64], options: &PulseOptions) -> Result<Vec<(String, f64)>> {
        let threshold = options.pulse_threshold.unwrap_or(self.config.pulse_threshold);
        let semantic_threshold = options.semantic_threshold.unwrap_or(0.0).max(0.0);
        let now = now_millis();

        let mut semantic: HashMap<String, f64> = HashMap::new();
        let top_k = options.top_k.unwrap_or_else(|| self.graph.vectors().size());
        for hit in self.graph.vectors().query(vector, top_k, 0.0)? {
            if hit.score > semantic_threshold {
                semantic.insert(hit.id, hit.score);
            }
        }

        let seeds: Vec<String> = semantic.keys().cloned().collect();
        let spread_options = SpreadOptions {
            depth: options.depth,
            decay: options.decay,
            edge_types: options.edge_types.clone(),
        };
        let graph_contributions = self.spread(&seeds, &spread_options);

        let ids: HashSet<String> = semantic.keys().chain(graph_contributions.keys()).cloned().collect();
        let mut scores: Vec<(String, f64)> = Vec::new();
        for id in ids {
            let Some(node) = self.graph.get_node(&id) else { continue };
            let s = semantic.get(&id).copied().unwrap_or(0.0);
            let g = graph_contributions.get(&id).copied().unwrap_or(0.0);
            scores.push((
                id,
                composite_score(node, s, g, now, &self.config.weights, self.config.recency_half_life_ms),
            ));
        }
        scores.retain(|(_, score)| *score > threshold);
        scores.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        Ok(scores)
    }

    /// `pulse` for text: embeds `text` with the graph's provider first.
    pub fn pulse_text(&mut self, text: &str, options: &PulseOptions) -> Result<Vec<(String, f64)>> {
        let vector = self.graph.embed(text)?;
        self.pulse(&vector, options)
    }

    /// `pulse` plus reinforcement: nodes whose composite score clears
    /// `absorb_threshold` receive durable reinforcement of `absorb_gain * score`.
    /// This is the self-maintaining "engine" — reading/querying a region causes
    /// the relevant part of the graph to become more active.
    pub fn absorb(&mut self, vector: &[f64], options: &PulseOptions) -> Result<Vec<(String, f64)>> {
        let scores = self.pulse(vector, options)?;
        let entries: Vec<(String, f64, Option<String>)> = scores
            .iter()
            .filter(|(_, score)| *score >= self.config.absorb_threshold)
            .map(|(id, score)| (id.clone(), clamp01(self.config.absorb_gain * score), Some("pulse".to_string())))
            .collect();
        self.reinforce_all(&entries)?;
        Ok(scores)
    }

    /// `absorb` for text.
    pub fn absorb_text(&mut self, text: &str, options: &PulseOptions) -> Result<Vec<(String, f64)>> {
        let vector = self.graph.embed(text)?;
        self.absorb(&vector, options)
    }

    // ── working memory ──

    /// The current "mental state": loaded nodes ranked by `effective`
    /// activation (durable decayed score + transient attention, minus
    /// inhibition) descending, top `limit`.
    pub fn working_memory(&mut self, limit: usize, min_score: f64) -> Result<Vec<Node>> {
        self.working_memory_with_options(&WorkingMemoryOptions { limit, min_score, ..Default::default() })
    }

    /// Budgeted, diversity-aware working-memory selection — a
    /// memory-flavoured maximal-marginal-relevance pass suited to LLM context
    /// assembly: greedily picks the highest `relevance − λ ×
    /// similarity-to-already-selected` candidate under a token budget, so the
    /// result isn't many near-duplicate highly-activated neighbours.
    pub fn working_memory_with_options(&mut self, options: &WorkingMemoryOptions) -> Result<Vec<Node>> {
        if options.limit == 0 {
            return Ok(Vec::new());
        }
        let now = now_millis();
        let mut candidates: Vec<(f64, Node)> = Vec::new();
        for node in self.graph.query().to_array() {
            let score = effective_activation(&node, self.attention_of(&node.id), now, self.config.score_half_life_ms, options.context.as_deref());
            if score > options.min_score {
                candidates.push((score, node));
            }
        }
        candidates.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));

        if options.diversity_lambda <= 0.0 && options.token_budget.is_none() {
            candidates.truncate(options.limit);
            return Ok(candidates.into_iter().map(|(_, n)| n).collect());
        }

        // Cap the candidate pool before the O(pool * selected) MMR pass, so a
        // graph with many loaded nodes doesn't turn selection quadratic.
        let pool_size = (options.limit * 4).max(options.limit);
        candidates.truncate(pool_size);

        let budget = options.token_budget.unwrap_or(f64::INFINITY);
        let mut remaining = candidates;
        let mut selected: Vec<Node> = Vec::new();
        let mut spent = 0.0;
        while selected.len() < options.limit && !remaining.is_empty() {
            let mut best_index = 0;
            let mut best_score = f64::NEG_INFINITY;
            for (i, (score, node)) in remaining.iter().enumerate() {
                let max_similarity = selected
                    .iter()
                    .map(|s| (options.similarity_of)(node, s))
                    .fold(0.0_f64, f64::max);
                let mmr = if options.diversity_lambda > 0.0 {
                    (1.0 - options.diversity_lambda) * score - options.diversity_lambda * max_similarity
                } else {
                    *score
                };
                if mmr > best_score {
                    best_score = mmr;
                    best_index = i;
                }
            }
            let (_, chosen) = remaining.remove(best_index);
            let cost = (options.cost_of)(&chosen);
            if spent + cost > budget {
                break;
            }
            spent += cost;
            selected.push(chosen);
        }
        Ok(selected)
    }

    /// Alias for querying the most-activated nodes by durable score only —
    /// convenience over `Graph::top_activated` for callers without the engine.
    pub fn top_activated(&self, limit: usize, min_score: f64) -> Vec<&Node> {
        self.graph.top_activated(limit, min_score)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use polypack_core::{InMemoryStorage, StoreConfig};

    use crate::edge::EdgeOwnership;
    use crate::graph::{Graph, GraphConfig};

    fn test_graph() -> Graph {
        Graph::open(Box::new(InMemoryStorage::new()), StoreConfig::default(), GraphConfig::default()).unwrap()
    }

    fn node(id: &str, node_type: &str) -> Node {
        Node {
            id: id.into(),
            node_type: node_type.into(),
            data: serde_json::Map::new(),
            vector: None,
            inserted_at: 1,
            updated_at: 1,
            revision: 0,
            activation: None,
        }
    }

    fn ids(entries: Vec<(String, f64)>) -> Vec<String> {
        entries.into_iter().map(|(id, _)| id).collect()
    }

    #[test]
    fn spread_attenuates_per_hop_and_respects_edge_types() {
        let mut g = test_graph();
        for id in ["ai", "vector-db", "polypack", "nostr"] {
            g.add_node(node(id, "topic")).unwrap();
        }
        g.add_edge("ai", "REL", "vector-db", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("vector-db", "REL", "polypack", None, EdgeOwnership::Reference).unwrap();
        g.add_edge("polypack", "REL", "nostr", None, EdgeOwnership::Reference).unwrap();

        let mut engine = ActivationEngine::new(&mut g, ActivationConfig::default());
        let spread = engine.spread(&["ai".to_string()], &SpreadOptions { depth: Some(2), ..Default::default() });
        assert!((spread.get("vector-db").copied().unwrap_or(0.0) - 0.5).abs() < 1e-9);
        assert!((spread.get("polypack").copied().unwrap_or(0.0) - 0.25).abs() < 1e-9);
        assert!(!spread.contains_key("nostr"));

        let restricted = engine.spread(
            &["ai".to_string()],
            &SpreadOptions { depth: Some(1), edge_types: Some(vec!["MISSING".into()]), ..Default::default() },
        );
        assert!(restricted.is_empty());
        engine.dispose();
    }

    #[test]
    fn bump_attention_stays_local_then_promotes_past_the_threshold() {
        let mut g = test_graph();
        g.add_node(node("a", "t")).unwrap();
        g.reinforce_node("a", 0.5, None).unwrap();
        let half_life = ActivationConfig::default().score_half_life_ms;

        {
            let mut engine = ActivationEngine::new(&mut g, ActivationConfig::default());
            engine.bump_attention("a", 0.02);
            assert!((engine.attention_of("a") - 0.02).abs() < 1e-9);
            // effective()/get_activation() decay-correct against a fresh wall-clock
            // read, so allow for the (small) real elapsed time since reinforcement.
            assert!((engine.effective("a", None) - 0.52).abs() < 1e-5);
            // Sub-threshold attention never touches durable state.
            engine.bump_attention("a", 0.05);
            assert_eq!(engine.attention_of("a"), 0.0);
        }

        assert!((g.get_activation("a", half_life) - 0.57).abs() < 1e-5, "promoted 0.07 accumulates into durable");
    }

    #[test]
    fn pulse_scores_the_relevant_region_and_keeps_unrelated_nodes_cold() {
        let mut g = test_graph();
        g.add_node_with_embedding(node("ev123", "event"), "local AI models vector search").unwrap();
        g.add_node_with_embedding(node("ev456", "event"), "random sports game discussion").unwrap();

        let vector = g.embed("local AI model vector search").unwrap();
        let mut engine = ActivationEngine::new(&mut g, ActivationConfig::default());
        let scores = engine.pulse(&vector, &PulseOptions::default()).unwrap();
        let top = ids(scores);
        assert!(top.iter().any(|id| id == "ev123"));
        assert!(!top.iter().any(|id| id == "ev456"), "zero-similarity nodes never seed the region");
        engine.dispose();
    }

    #[test]
    fn absorb_reinforces_nodes_above_the_threshold() {
        let mut g = test_graph();
        g.add_node_with_embedding(node("a", "t"), "local AI model vector search").unwrap();
        g.add_node_with_embedding(node("unrelated", "t"), "grocery list recipes").unwrap();

        let vector = g.embed("local AI model vector search").unwrap();
        {
            let mut engine = ActivationEngine::new(&mut g, ActivationConfig { absorb_threshold: 0.3, ..Default::default() });
            engine.absorb(&vector, &PulseOptions::default()).unwrap();
        }

        assert!(g.get_activation_state("a").is_some());
        assert!(g.get_activation_state("unrelated").is_none(), "unrelated node stays cold");
    }

    #[test]
    fn working_memory_returns_the_most_active_nodes() {
        let mut g = test_graph();
        for (id, amount) in [("a", 0.9), ("b", 0.5), ("c", 0.1)] {
            g.add_node(node(id, "t")).unwrap();
            g.reinforce_node(id, amount, None).unwrap();
        }
        let mut engine = ActivationEngine::new(&mut g, ActivationConfig::default());
        let wm = engine.working_memory(2, 0.0).unwrap();
        assert_eq!(wm.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
        engine.dispose();
    }

    #[test]
    fn reinforces_a_context_independently_of_the_global_score() {
        let mut g = test_graph();
        g.add_node(node("a", "t")).unwrap();
        g.reinforce_node("a", 0.2, None).unwrap();
        g.reinforce_node_in_context("a", 0.3, None, Some("project-x")).unwrap();

        let mut engine = ActivationEngine::new(&mut g, ActivationConfig::default());
        assert!((engine.effective("a", None) - 0.5).abs() < 1e-5);
        assert!((engine.effective("a", Some("project-x")) - 0.3).abs() < 1e-5);
        assert_eq!(engine.effective("a", Some("coding")), 0.0);
        assert!((engine.graph.get_context_activation("a", "project-x") - 0.3).abs() < 1e-5);
        engine.dispose();
    }

    #[test]
    fn suppress_subtracts_inhibition_from_effective_but_not_the_raw_score() {
        let mut g = test_graph();
        g.add_node(node("a", "t")).unwrap();
        g.reinforce_node("a", 0.8, None).unwrap();
        g.suppress_node("a", 0.5, None).unwrap();

        let mut engine = ActivationEngine::new(&mut g, ActivationConfig::default());
        assert!((engine.inhibition_of("a") - 0.5).abs() < 1e-5);
        assert!((engine.effective("a", None) - 0.3).abs() < 1e-5);
        assert!((engine.graph.get_activation("a", ActivationConfig::default().score_half_life_ms) - 0.8).abs() < 1e-5);

        engine.graph.suppress_node("a", -0.5, None).unwrap();
        assert!((engine.effective("a", None) - 0.8).abs() < 1e-5);
        engine.dispose();
    }

    #[test]
    fn working_memory_with_token_budget_stops_once_spent() {
        let mut g = test_graph();
        for (id, amount) in [("a", 0.9), ("b", 0.5), ("c", 0.1)] {
            g.add_node(node(id, "t")).unwrap();
            g.reinforce_node(id, amount, None).unwrap();
        }
        let mut engine = ActivationEngine::new(&mut g, ActivationConfig::default());
        let options = WorkingMemoryOptions { limit: 10, token_budget: Some(2.0), ..Default::default() };
        let selected = engine.working_memory_with_options(&options).unwrap();
        assert_eq!(selected.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);
        engine.dispose();
    }

    #[test]
    fn working_memory_with_diversity_lambda_penalises_redundant_neighbours() {
        let mut g = test_graph();
        let mut a = node("a", "t");
        a.vector = Some(vec![1.0, 0.0]);
        let mut b = node("b", "t");
        b.vector = Some(vec![1.0, 0.01]);
        let mut c = node("c", "t");
        c.vector = Some(vec![0.0, 1.0]);
        g.add_node(a).unwrap();
        g.add_node(b).unwrap();
        g.add_node(c).unwrap();
        g.reinforce_node("a", 0.9, None).unwrap();
        g.reinforce_node("b", 0.85, None).unwrap();
        g.reinforce_node("c", 0.5, None).unwrap();

        let mut engine = ActivationEngine::new(&mut g, ActivationConfig::default());
        let relevance_only = engine
            .working_memory_with_options(&WorkingMemoryOptions { limit: 2, ..Default::default() })
            .unwrap();
        assert_eq!(relevance_only.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(), vec!["a", "b"]);

        let diverse = engine
            .working_memory_with_options(&WorkingMemoryOptions { limit: 2, diversity_lambda: 0.8, ..Default::default() })
            .unwrap();
        assert_eq!(diverse.iter().map(|n| n.id.as_str()).collect::<Vec<_>>(), vec!["a", "c"]);
        engine.dispose();
    }
}
