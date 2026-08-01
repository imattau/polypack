//! Update-safe HNSW approximate nearest-neighbour index.
//!
//! Ported from the TypeScript `HNSWIndex` (`src/hnsw-index.ts`) with the same
//! behaviour: physical unlinking on remove (no tombstones), remove-then-insert
//! on replacement, and `update` for in-place vector replacement. The index
//! uses cosine distance (`1 - similarity`) internally.

use crate::error::{PolypackError, Result};
use crate::vector::{cosine, euclidean, DistanceFn};
use std::cmp::Ordering;
use std::collections::{BinaryHeap, HashMap, HashSet};

#[derive(Debug, Clone, Copy)]
pub struct HnswConfig {
    /// Max graph connections per node above layer 0. Higher = better recall, more memory.
    pub m: usize,
    /// Max connections per node at layer 0 (usually `2*m`).
    pub mmax0: usize,
    /// Candidate list size while inserting. Higher = better graph quality, slower builds.
    pub ef_construction: usize,
    /// Candidate list size while querying. Higher = better recall, slower queries.
    pub ef_search: usize,
}

impl Default for HnswConfig {
    fn default() -> Self {
        HnswConfig {
            m: 16,
            mmax0: 32,
            ef_construction: 200,
            ef_search: 200,
        }
    }
}

/// A query result: a node id with its similarity score (higher is better).
#[derive(Debug, Clone, PartialEq)]
pub struct ScoredId {
    pub id: String,
    pub score: f64,
}

#[derive(Debug, Clone, PartialEq)]
struct Candidate {
    id: String,
    dist: f64,
}

impl Eq for Candidate {}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(Ordering::Equal)
            .then_with(|| self.id.cmp(&other.id))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

struct LevelRng(u32);

impl LevelRng {
    fn next_f64(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x6d2b79f5);
        let a = self.0 as i32;
        let mut t = i32::wrapping_mul(a ^ ((a as u32) >> 15) as i32, 1 | a);
        t = (t.wrapping_add(i32::wrapping_mul(
            t ^ ((t as u32) >> 7) as i32,
            61 | t,
        ))) ^ t;
        ((t ^ ((t as u32) >> 14) as i32) as u32) as f64 / 4294967296.0
    }
}

/// Approximate, update-safe HNSW vector index. See the module docs for the
/// remove/update semantics that differ from a naive HNSW implementation.
pub struct HnswIndex {
    nodes: HashMap<String, Vec<f64>>,
    node_level: HashMap<String, u32>,
    adjacency: HashMap<u32, HashMap<String, Vec<String>>>,
    entry_point: Option<String>,
    max_layer: i32,
    distance: DistanceFn,
    config: HnswConfig,
    ml: f64,
    level_rng: LevelRng,
}

impl HnswIndex {
    /// `m`, `mmax0`, `ef_construction`, and `ef_search` must all be at least 1.
    /// A value of 0 isn't just a bad tuning choice: `search_layer`'s
    /// candidate list is capped at `ef`, so `ef == 0` collapses every search
    /// (including the internal ones `add` uses to wire up the graph) to an
    /// empty result — silently returning no query hits, or panicking on
    /// insert when graph-building code indexes into that empty result.
    pub fn new(config: HnswConfig, level_seed: u32) -> Result<Self> {
        for (name, value) in [
            ("m", config.m),
            ("mmax0", config.mmax0),
            ("ef_construction", config.ef_construction),
            ("ef_search", config.ef_search),
        ] {
            if value < 1 {
                return Err(PolypackError::InvalidArgument(format!("HnswConfig.{name} must be at least 1")));
            }
        }
        Ok(HnswIndex {
            nodes: HashMap::new(),
            node_level: HashMap::new(),
            adjacency: HashMap::new(),
            entry_point: None,
            max_layer: -1,
            distance: DistanceFn::Cosine,
            ml: 1.0 / (config.m.max(2) as f64).ln(),
            config,
            level_rng: LevelRng(level_seed),
        })
    }

    pub fn add(&mut self, id: &str, vector: &[f64]) -> Result<()> {
        if id.is_empty() {
            return Err(PolypackError::InvalidArgument("vector id must not be empty".into()));
        }
        if !vector.iter().all(|x| x.is_finite()) {
            return Err(PolypackError::InvalidArgument("vector must contain finite values".into()));
        }
        if self.nodes.contains_key(id) {
            self.remove(id);
        }
        let stored = vector.to_vec();
        let level = self.assign_level();
        self.nodes.insert(id.to_string(), stored);
        self.insert_into_graph(id, level);
        Ok(())
    }

    /// Replace the vector and topology of an existing id (same as `add`).
    pub fn update(&mut self, id: &str, vector: &[f64]) -> Result<()> {
        self.add(id, vector)
    }

    pub fn remove(&mut self, id: &str) {
        if !self.nodes.contains_key(id) {
            return;
        }
        self.nodes.remove(id);
        self.unlink_node(id);
    }

    pub fn clear(&mut self) {
        self.nodes.clear();
        self.node_level.clear();
        self.adjacency.clear();
        self.entry_point = None;
        self.max_layer = -1;
    }

    pub fn size(&self) -> usize {
        self.nodes.len()
    }

    pub fn nodes(&self) -> &HashMap<String, Vec<f64>> {
        &self.nodes
    }

    pub fn has(&self, id: &str) -> bool {
        self.nodes.contains_key(id)
    }

    pub fn get(&self, id: &str) -> Option<&[f64]> {
        self.nodes.get(id).map(|v| v.as_slice())
    }

    pub fn query(&self, vector: &[f64], top_k: usize, threshold: f64) -> Result<Vec<ScoredId>> {
        if !vector.iter().all(|x| x.is_finite()) {
            return Err(PolypackError::InvalidArgument("query vector must contain finite values".into()));
        }
        if !threshold.is_finite() {
            return Err(PolypackError::InvalidArgument("threshold must be finite".into()));
        }
        if top_k == 0 || self.nodes.is_empty() {
            return Ok(Vec::new());
        }
        let ep = match &self.entry_point {
            Some(ep) => ep.clone(),
            None => return Ok(Vec::new()),
        };

        let mut current = ep;
        for lc in (1..=self.max_layer).rev() {
            current = self.greedy_search(vector, &current, lc as u32);
        }
        let candidates = self.search_layer(vector, &current, self.config.ef_search, 0);

        let mut out: Vec<ScoredId> = candidates
            .into_iter()
            .filter(|c| 1.0 - c.dist >= threshold)
            .map(|c| ScoredId { id: c.id, score: 1.0 - c.dist })
            .collect();
        out.truncate(top_k);
        Ok(out)
    }

    // ── Internals ──

    fn dist(&self, a: &[f64], b: &[f64]) -> f64 {
        let sim = match self.distance {
            DistanceFn::Cosine => cosine(a, b),
            DistanceFn::Euclidean => euclidean(a, b),
        };
        1.0 - sim.unwrap_or(0.0)
    }

    fn assign_level(&mut self) -> u32 {
        let r = self.level_rng.next_f64();
        let level = (-r.ln() * self.ml).floor();
        level.max(0.0) as u32
    }

    fn ensure_layer(&mut self, layer: u32) {
        self.adjacency.entry(layer).or_default();
    }

    fn get_neighbors(&self, node_id: &str, layer: u32) -> Vec<String> {
        match self.adjacency.get(&layer) {
            Some(layer_adj) => match layer_adj.get(node_id) {
                Some(neighbors) => neighbors
                    .iter()
                    .filter(|nid| self.nodes.contains_key(*nid))
                    .cloned()
                    .collect(),
                None => Vec::new(),
            },
            None => Vec::new(),
        }
    }

    fn unlink_node(&mut self, id: &str) {
        let layers: Vec<u32> = self.adjacency.keys().copied().collect();
        for layer in layers {
            let layer_adj = match self.adjacency.get_mut(&layer) {
                Some(a) => a,
                None => continue,
            };
            let neighbors = layer_adj.remove(id);
            if let Some(neighbors) = neighbors {
                for nid in neighbors {
                    if let Some(nset) = layer_adj.get_mut(&nid) {
                        nset.retain(|x| x != id);
                        if nset.is_empty() {
                            layer_adj.remove(&nid);
                        }
                    }
                }
            }
        }
        self.node_level.remove(id);
        if self.entry_point.as_deref() == Some(id) {
            self.entry_point = self.pick_new_entry_point();
        }
        if self.entry_point.is_none() && !self.nodes.is_empty() {
            // Degenerate case: nodes remain but the unlink emptied every layer
            // (e.g. a 2-node graph whose entry point is removed). Re-seed the
            // index from an arbitrary survivor so later inserts reconnect.
            if let Some(survivor) = self.nodes.keys().next().cloned() {
                let level = self.node_level.get(&survivor).copied().unwrap_or(0);
                self.max_layer = level as i32;
                self.entry_point = Some(survivor);
                return;
            }
        }
        while self.max_layer >= 0 {
            let layer = self.max_layer as u32;
            let keep = match self.adjacency.get(&layer) {
                Some(layer_adj) => !layer_adj.is_empty(),
                None => false,
            };
            if keep {
                break;
            }
            self.adjacency.remove(&layer);
            self.max_layer -= 1;
        }
    }

    fn pick_new_entry_point(&self) -> Option<String> {
        for lc in (0..=self.max_layer).rev() {
            if let Some(layer_adj) = self.adjacency.get(&(lc as u32)) {
                for nid in layer_adj.keys() {
                    if self.nodes.contains_key(nid) {
                        return Some(nid.clone());
                    }
                }
            }
        }
        None
    }

    fn greedy_search(&self, q: &[f64], ep: &str, layer: u32) -> String {
        let mut current = ep.to_string();
        let mut current_dist = self.dist(q, &self.nodes[current.as_str()].clone());
        let mut visited = HashSet::from([current.clone()]);

        loop {
            let neighbors = self.get_neighbors(&current, layer);
            let mut improved = false;
            for nid in neighbors {
                if visited.contains(&nid) {
                    continue;
                }
                visited.insert(nid.clone());
                let n_dist = self.dist(q, &self.nodes[&nid].clone());
                if n_dist < current_dist {
                    current = nid;
                    current_dist = n_dist;
                    improved = true;
                }
            }
            if !improved {
                break;
            }
        }
        current
    }

    fn search_layer(&self, q: &[f64], entry: &str, ef: usize, layer: u32) -> Vec<Candidate> {
        let entry_dist = self.dist(q, &self.nodes[entry].clone());

        let mut visited = HashSet::from([entry.to_string()]);
        let mut candidates: BinaryHeap<Reverse<Candidate>> = BinaryHeap::new();
        let mut results: BinaryHeap<Candidate> = BinaryHeap::new();
        candidates.push(Reverse(Candidate { id: entry.to_string(), dist: entry_dist }));
        results.push(Candidate { id: entry.to_string(), dist: entry_dist });

        while let Some(Reverse(c)) = candidates.pop() {
            let worst = match results.peek() {
                Some(w) => w.dist,
                None => break,
            };
            if c.dist > worst {
                break;
            }
            let neighbors = self.get_neighbors(&c.id, layer);
            for eid in neighbors {
                if visited.contains(&eid) {
                    continue;
                }
                visited.insert(eid.clone());
                let e_dist = self.dist(q, &self.nodes[&eid].clone());
                let worst = results.peek().map(|w| w.dist).unwrap_or(f64::INFINITY);
                if results.len() < ef || e_dist < worst {
                    candidates.push(Reverse(Candidate { id: eid.clone(), dist: e_dist }));
                    results.push(Candidate { id: eid, dist: e_dist });
                    if results.len() > ef {
                        results.pop();
                    }
                }
            }
        }

        let mut out: Vec<Candidate> = results.into_iter().collect();
        out.sort_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap_or(Ordering::Equal));
        if out.len() > ef {
            out.truncate(ef);
        }
        out
    }

    fn insert_into_graph(&mut self, id: &str, level: u32) {
        if self.nodes.len() == 1 {
            self.entry_point = Some(id.to_string());
            self.max_layer = level as i32;
            self.node_level.insert(id.to_string(), level);
            self.ensure_layer(level);
            self.adjacency.get_mut(&level).unwrap().insert(id.to_string(), Vec::new());
            return;
        }

        let q = self.nodes[id].clone();
        let mut ep = self.entry_point.clone().unwrap();

        for lc in ((level as i32 + 1)..=self.max_layer).rev() {
            ep = self.greedy_search(&q, &ep, lc as u32);
        }

        let top = (level as i32).min(self.max_layer);
        for lc in (0..=top).rev() {
            let layer = lc as u32;
            let candidates = self.search_layer(&q, &ep, self.config.ef_construction, layer);
            let max_conn = if layer == 0 { self.config.mmax0 } else { self.config.m };
            let neighbor_ids: Vec<String> = candidates.iter().take(max_conn).map(|c| c.id.clone()).collect();

            self.ensure_layer(layer);
            let layer_adj = self.adjacency.get_mut(&layer).unwrap();
            let my_set = layer_adj.entry(id.to_string()).or_default();
            for nid in &neighbor_ids {
                if !my_set.contains(nid) {
                    my_set.push(nid.clone());
                }
            }
            for nid in &neighbor_ids {
                let nset = layer_adj.entry(nid.clone()).or_default();
                if !nset.contains(&id.to_string()) {
                    nset.push(id.to_string());
                }
            }

            for nid in &neighbor_ids {
                self.shrink_neighbors(nid, layer, max_conn);
            }

            ep = candidates[0].id.clone();
        }

        if (level as i32) > self.max_layer {
            self.max_layer = level as i32;
            self.entry_point = Some(id.to_string());
        }
        self.node_level.insert(id.to_string(), level);
    }

    fn shrink_neighbors(&mut self, nid: &str, layer: u32, max: usize) {
        let Some(layer_adj) = self.adjacency.get(&layer) else { return };
        let Some(neighbors) = layer_adj.get(nid) else { return };
        if neighbors.len() <= max {
            return;
        }
        let vec = self.nodes.get(nid).cloned();
        let Some(vec) = vec else { return };

        let mut with_dist: Vec<(f64, String)> = neighbors
            .iter()
            .filter_map(|n| {
                self.nodes.get(n).map(|nvec| (self.dist(&vec, nvec), n.clone()))
            })
            .collect();
        with_dist.sort_by(|a, b| {
            a.0.partial_cmp(&b.0)
                .unwrap_or(Ordering::Equal)
                .then_with(|| a.1.cmp(&b.1))
        });

        let kept: Vec<String> = with_dist.into_iter().take(max).map(|(_, id)| id).collect();
        let layer_adj = self.adjacency.get_mut(&layer).unwrap();
        layer_adj.insert(nid.to_string(), kept);
    }
}

use std::cmp::Reverse;

#[cfg(test)]
mod tests {
    use super::*;
    use crate::vector::ExactIndex;

    const DIMS: usize = 8;

    fn vec8(i: f64, j: f64) -> Vec<f64> {
        let mut v = vec![0.0; DIMS];
        v[0] = i;
        v[1] = j;
        v
    }

    #[test]
    fn new_rejects_non_positive_config_values() {
        for cfg in [
            HnswConfig { m: 0, ..Default::default() },
            HnswConfig { mmax0: 0, ..Default::default() },
            HnswConfig { ef_construction: 0, ..Default::default() },
            HnswConfig { ef_search: 0, ..Default::default() },
        ] {
            match HnswIndex::new(cfg, 7) {
                Err(PolypackError::InvalidArgument(_)) => {}
                _ => panic!("zero config value must be rejected with InvalidArgument"),
            }
        }
    }

    #[test]
    fn basic_recall_matches_exact() {
        let mut hnsw = HnswIndex::new(HnswConfig { ef_search: 300, ..Default::default() }, 7).unwrap();
        let mut exact = ExactIndex::new(DistanceFn::Cosine);
        let mut rand = crate::rng::Mulberry32::new(42);
        for i in 0..500 {
            let v: Vec<f64> = (0..DIMS).map(|_| rand.next_f64() * 2.0 - 1.0).collect();
            hnsw.add(&format!("v{i}"), &v).unwrap();
            exact.add(&format!("v{i}"), &v).unwrap();
        }
        let mut hits = 0;
        let trials = 50;
        for _t in 0..trials {
            let q: Vec<f64> = (0..DIMS).map(|_| rand.next_f64() * 2.0 - 1.0).collect();
            let exact_ids: HashSet<String> = exact.query(&q, 10, 0.0).unwrap().into_iter().map(|s| s.id).collect();
            let ann: Vec<String> = hnsw.query(&q, 10, 0.0).unwrap().into_iter().map(|s| s.id).collect();
            hits += ann.iter().filter(|id| exact_ids.contains(*id)).count();
        }
        let recall = hits as f64 / (trials * 10) as f64;
        assert!(recall >= 0.9, "recall@10 was {recall}");
    }

    #[test]
    fn remove_then_readd_works() {
        let mut hnsw = HnswIndex::new(HnswConfig::default(), 7).unwrap();
        hnsw.add("a", &vec8(1.0, 0.0)).unwrap();
        hnsw.add("b", &vec8(0.0, 1.0)).unwrap();
        hnsw.remove("a");
        assert_eq!(hnsw.size(), 1);
        assert!(!hnsw.has("a"));
        hnsw.add("a", &vec8(1.0, 0.0)).unwrap();
        assert_eq!(hnsw.size(), 2);
        let r = hnsw.query(&vec8(1.0, 0.0), 5, 0.0).unwrap();
        assert_eq!(r[0].id, "a");
    }

    #[test]
    fn update_replaces_vector_without_duplicate() {
        let mut hnsw = HnswIndex::new(HnswConfig::default(), 7).unwrap();
        hnsw.add("a", &vec8(1.0, 0.0)).unwrap();
        hnsw.add("b", &vec8(0.0, 1.0)).unwrap();
        hnsw.update("a", &vec8(0.0, 1.0)).unwrap();
        assert_eq!(hnsw.size(), 2);
        assert!(hnsw.has("a"));
        let q = vec8(0.0, 1.0);
        let r = hnsw.query(&q, 5, 0.0).unwrap();
        assert!(r.iter().any(|s| s.id == "a"));
    }

    #[test]
    fn clears_topology_after_removing_everything() {
        let mut hnsw = HnswIndex::new(HnswConfig::default(), 7).unwrap();
        for i in 0..50 {
            hnsw.add(&format!("n{i}"), &vec8(i as f64 / 50.0, 0.0)).unwrap();
        }
        for i in 0..50 {
            hnsw.remove(&format!("n{i}"));
        }
        assert_eq!(hnsw.size(), 0);
        hnsw.add("fresh", &vec8(1.0, 0.0)).unwrap();
        let r = hnsw.query(&vec8(1.0, 0.0), 5, 0.0).unwrap();
        assert_eq!(r[0].id, "fresh");
    }

    #[test]
    fn reuses_ids_after_churn_without_collapse() {
        let mut hnsw = HnswIndex::new(HnswConfig { ef_search: 300, ..Default::default() }, 7).unwrap();
        let mut exact = ExactIndex::new(DistanceFn::Cosine);
        let mut rand = crate::rng::Mulberry32::new(3);
        for i in 0..200 {
            let v: Vec<f64> = (0..DIMS).map(|_| rand.next_f64() * 2.0 - 1.0).collect();
            hnsw.add(&format!("v{i}"), &v).unwrap();
            exact.add(&format!("v{i}"), &v).unwrap();
        }
        // Remove and re-add half with new vectors, update a quarter in place.
        for i in (0..200).step_by(2) {
            exact.remove(&format!("v{i}"));
            hnsw.remove(&format!("v{i}"));
        }
        for i in (0..200).step_by(2) {
            let v: Vec<f64> = (0..DIMS).map(|_| rand.next_f64() * 2.0 - 1.0).collect();
            exact.add(&format!("v{i}"), &v).unwrap();
            hnsw.add(&format!("v{i}"), &v).unwrap();
        }
        for i in (1..200).step_by(4) {
            let v: Vec<f64> = (0..DIMS).map(|_| rand.next_f64() * 2.0 - 1.0).collect();
            exact.add(&format!("v{i}"), &v).unwrap();
            hnsw.update(&format!("v{i}"), &v).unwrap();
        }
        assert_eq!(hnsw.size(), exact.size());
        let mut hits = 0;
        let trials = 30;
        for _ in 0..trials {
            let q: Vec<f64> = (0..DIMS).map(|_| rand.next_f64() * 2.0 - 1.0).collect();
            let exact_ids: HashSet<String> = exact.query(&q, 10, 0.0).unwrap().into_iter().map(|s| s.id).collect();
            let ann: Vec<String> = hnsw.query(&q, 10, 0.0).unwrap().into_iter().map(|s| s.id).collect();
            hits += ann.iter().filter(|id| exact_ids.contains(*id)).count();
        }
        let recall = hits as f64 / (trials * 10) as f64;
        assert!(recall >= 0.85, "recall after churn was {recall}");
    }
}
