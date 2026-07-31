//! Similarity functions and an exact vector index.
//!
//! Scores are similarity values in `[0, 1]`, larger is better, matching
//! `specification/data-model.md` section 6.

use crate::error::{PolypackError, Result};
use std::collections::HashMap;

pub fn cosine(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(PolypackError::DimensionMismatch { expected: a.len(), got: b.len() });
    }
    let mut dot = 0.0;
    let mut na = 0.0;
    let mut nb = 0.0;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = na.sqrt() * nb.sqrt();
    Ok(if denom == 0.0 { 0.0 } else { dot / denom })
}

pub fn euclidean(a: &[f64], b: &[f64]) -> Result<f64> {
    if a.len() != b.len() {
        return Err(PolypackError::DimensionMismatch { expected: a.len(), got: b.len() });
    }
    let mut sum = 0.0;
    for i in 0..a.len() {
        let d = a[i] - b[i];
        sum += d * d;
    }
    Ok(1.0 / (1.0 + sum.sqrt()))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum DistanceFn {
    Cosine,
    Euclidean,
}

#[derive(Debug, Clone)]
pub struct ScoredId {
    pub id: String,
    pub score: f64,
}

/// Exact in-memory vector index with the same ordering semantics as the
/// TypeScript `VectorIndex`: ties break by insertion order.
pub struct ExactIndex {
    entries: Vec<(String, Vec<f64>)>,
    positions: HashMap<String, usize>,
    distance: DistanceFn,
}

impl ExactIndex {
    pub fn new(distance: DistanceFn) -> Self {
        ExactIndex {
            entries: Vec::new(),
            positions: HashMap::new(),
            distance,
        }
    }

    pub fn add(&mut self, id: &str, vector: &[f64]) -> Result<()> {
        if id.is_empty() {
            return Err(PolypackError::InvalidArgument("vector id must not be empty".into()));
        }
        if !vector.iter().all(|x| x.is_finite()) {
            return Err(PolypackError::InvalidArgument("vector must contain finite values".into()));
        }
        if let Some(&pos) = self.positions.get(id) {
            self.entries[pos] = (id.to_string(), vector.to_vec());
            return Ok(());
        }
        self.positions.insert(id.to_string(), self.entries.len());
        self.entries.push((id.to_string(), vector.to_vec()));
        Ok(())
    }

    pub fn remove(&mut self, id: &str) {
        if let Some(&pos) = self.positions.get(id) {
            self.entries.remove(pos);
            self.positions.remove(id);
            // Rebuild the position map; keeps insertion order of the remainder.
            self.positions = self.entries.iter().enumerate().map(|(i, (id, _))| (id.clone(), i)).collect();
        }
    }

    pub fn has(&self, id: &str) -> bool {
        self.positions.contains_key(id)
    }

    pub fn get(&self, id: &str) -> Option<&[f64]> {
        self.positions.get(id).map(|&pos| self.entries[pos].1.as_slice())
    }

    pub fn size(&self) -> usize {
        self.entries.len()
    }

    pub fn query(&self, vector: &[f64], top_k: usize, threshold: f64) -> Result<Vec<ScoredId>> {
        if !vector.iter().all(|x| x.is_finite()) {
            return Err(PolypackError::InvalidArgument("query vector must contain finite values".into()));
        }
        if top_k == 0 || self.entries.is_empty() {
            return Ok(Vec::new());
        }

        // Min-heap of the top-K so far. Root is the worst candidate:
        // smallest score, and for equal scores the later-inserted one.
        let mut heap: Vec<(f64, usize, &str)> = Vec::with_capacity(top_k + 1);
        let mut order = 0usize;

        for (id, v) in &self.entries {
            if v.len() != vector.len() {
                return Err(PolypackError::DimensionMismatch { expected: vector.len(), got: v.len() });
            }
            let score = match self.distance {
                DistanceFn::Cosine => cosine(vector, v)?,
                DistanceFn::Euclidean => euclidean(vector, v)?,
            };
            if score < threshold {
                continue;
            }
            let candidate = (score, order, id.as_str());
            if heap.len() < top_k {
                heap.push(candidate);
                let idx = heap.len() - 1;
                sift_up(&mut heap, idx);
            } else if candidate.0 > heap[0].0 {
                heap[0] = candidate;
                sift_down(&mut heap, 0);
            }
            order += 1;
        }

        let mut result: Vec<ScoredId> = heap
            .iter()
            .map(|(score, _, id)| ScoredId { id: (*id).to_string(), score: *score })
            .collect();
        // Sort by score desc; ties keep insertion order (stable).
        result.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap_or(std::cmp::Ordering::Equal));
        Ok(result)
    }
}

/// `a` is worse than `b` if lower score, or equal score and later insertion.
fn is_worse(a: (f64, usize, &str), b: (f64, usize, &str)) -> bool {
    a.0 < b.0 || (a.0 == b.0 && a.1 > b.1)
}

fn sift_up(heap: &mut Vec<(f64, usize, &str)>, mut i: usize) {
    while i > 0 {
        let parent = (i - 1) >> 1;
        if !is_worse(heap[i], heap[parent]) {
            break;
        }
        heap.swap(i, parent);
        i = parent;
    }
}

fn sift_down(heap: &mut Vec<(f64, usize, &str)>, mut i: usize) {
    let n = heap.len();
    loop {
        let l = i * 2 + 1;
        let r = l + 1;
        let mut smallest = i;
        if l < n && is_worse(heap[l], heap[smallest]) {
            smallest = l;
        }
        if r < n && is_worse(heap[r], heap[smallest]) {
            smallest = r;
        }
        if smallest == i {
            break;
        }
        heap.swap(i, smallest);
        i = smallest;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_ranking_is_descending() {
        let mut idx = ExactIndex::new(DistanceFn::Cosine);
        idx.add("x", &[1.0, 0.0, 0.0]).unwrap();
        idx.add("y", &[0.8, 0.6, 0.0]).unwrap();
        idx.add("z", &[0.6, 0.8, 0.0]).unwrap();
        idx.add("w", &[0.0, 1.0, 0.0]).unwrap();
        let r = idx.query(&[1.0, 0.0, 0.0], 3, 0.0).unwrap();
        let ids: Vec<&str> = r.iter().map(|s| s.id.as_str()).collect();
        assert_eq!(ids, ["x", "y", "z"]);
        assert!(r[0].score > r[1].score && r[1].score > r[2].score);
    }

    #[test]
    fn dimension_mismatch_errors() {
        let mut idx = ExactIndex::new(DistanceFn::Cosine);
        idx.add("x", &[1.0, 0.0]).unwrap();
        let err = idx.query(&[1.0, 0.0, 0.0], 1, 0.0).unwrap_err();
        assert_eq!(err, PolypackError::DimensionMismatch { expected: 3, got: 2 });
    }

    #[test]
    fn replace_and_remove() {
        let mut idx = ExactIndex::new(DistanceFn::Cosine);
        idx.add("a", &[1.0, 0.0]).unwrap();
        idx.add("a", &[0.0, 1.0]).unwrap();
        assert_eq!(idx.size(), 1);
        idx.remove("a");
        assert!(!idx.has("a"));
        assert_eq!(idx.size(), 0);
    }

    #[test]
    fn euclidean_similarity() {
        let mut idx = ExactIndex::new(DistanceFn::Euclidean);
        idx.add("a", &[0.0, 0.0]).unwrap();
        idx.add("b", &[1.0, 1.0]).unwrap();
        let r = idx.query(&[0.0, 0.0], 2, 0.0).unwrap();
        assert_eq!(r[0].id, "a");
        assert!(r[0].score > r[1].score);
    }
}
