//! Pure activation math, shared by `polypack-graph`'s `Graph`, the query
//! builders, and the `ActivationEngine`. Mirrors the activation helpers in the
//! TypeScript `src/utils.ts`.
//!
//! Decay is a pure function of elapsed time anchored at
//! `last_meaningful_activation`, so replicas with the same stored state and
//! their own clocks compute identical current scores.

use crate::model::{Node, NodeActivation};

const HOUR: i64 = 3_600_000;
const DAY: i64 = 24 * HOUR;

/// The two decay curves plus the importance-gain rate, mirroring the TypeScript
/// `ACTIVATION_DEFAULTS` constants.
pub struct ActivationCurves {
    /// Half-life of the short-term `score` curve (24 h).
    pub score_half_life_ms: i64,
    /// Half-life of the long-term `importance` curve (30 days).
    pub importance_half_life_ms: i64,
    /// Fraction of a reinforcement delta folded into `importance`.
    pub importance_gain: f64,
}

/// Default activation decay curves, shared by `polypack-core` and
/// `polypack-graph`.
pub const DEFAULT_ACTIVATION: ActivationCurves = ActivationCurves {
    score_half_life_ms: DAY,
    importance_half_life_ms: 30 * DAY,
    importance_gain: 0.05,
};

/// Clamp a number into [0, 1]. Matches the TypeScript `clamp01` semantics,
/// including returning NaN for NaN input.
pub fn clamp01(value: f64) -> f64 {
    value.clamp(0.0, 1.0)
}

/// Exponential-decay multiplier: `0.5 ** (elapsed / halfLife)`. Returns 1 for
/// non-positive elapsed times and a non-decaying (non-positive) half-life.
/// Deterministic and monotonic, matching `decayFactor` in the TypeScript
/// reference.
pub fn decay_factor(elapsed_ms: i64, half_life_ms: i64) -> f64 {
    if elapsed_ms <= 0 || half_life_ms <= 0 {
        return 1.0;
    }
    let factor = 0.5_f64.powf(elapsed_ms as f64 / half_life_ms as f64);
    if factor < 1.0 {
        factor
    } else {
        1.0
    }
}

/// Current decayed activation score of a node (0 when it has none).
pub fn activation_score_of(node: &Node, now: i64, half_life_ms: i64) -> f64 {
    match &node.activation {
        Some(a) => clamp01(a.score * decay_factor(now - a.last_meaningful_activation, half_life_ms)),
        None => 0.0,
    }
}

/// Read-only decay-corrected view of an activation record. Does NOT re-anchor
/// `last_meaningful_activation`; the stored values always represent the state
/// at that anchor, so reinforcement/merge can decay-correct and re-anchor.
pub fn decay_activation_state(
    activation: &NodeActivation,
    now: i64,
    score_half_life_ms: i64,
    importance_half_life_ms: i64,
) -> NodeActivation {
    NodeActivation {
        score: clamp01(activation.score * decay_factor(now - activation.last_meaningful_activation, score_half_life_ms)),
        importance: clamp01(
            activation.importance * decay_factor(now - activation.last_meaningful_activation, importance_half_life_ms),
        ),
        reinforcement_count: activation.reinforcement_count,
        last_meaningful_activation: activation.last_meaningful_activation,
    }
}

/// Apply a reinforcement delta: decay-correct the prior state to `now`, add
/// `delta` to `score`, fold a fraction (`importance_gain`) into `importance`,
/// increment the reinforcement counter, and re-anchor
/// `last_meaningful_activation` to `now` so the stored record again represents
/// "value at the anchor".
pub fn reinforce_activation(
    previous: Option<&NodeActivation>,
    delta: f64,
    now: i64,
    curves: &ActivationCurves,
) -> NodeActivation {
    let (score, importance) = match previous {
        Some(prev) => {
            let decayed = decay_activation_state(prev, now, curves.score_half_life_ms, curves.importance_half_life_ms);
            (decayed.score, decayed.importance)
        }
        None => (0.0, 0.0),
    };
    NodeActivation {
        score: clamp01(score + delta),
        importance: clamp01(importance + curves.importance_gain * delta),
        reinforcement_count: previous.map(|p| p.reinforcement_count + 1).unwrap_or(1),
        last_meaningful_activation: now,
    }
}

/// Merge two durable activation records (e.g. when a full node payload arrives
/// from sync). Decay-corrects both to `now`, keeps the stronger component of
/// each, and re-anchors to `now` so future decay is self-consistent.
///
/// This is the **total-state** merge (max, idempotent for re-delivered
/// snapshots). Concurrent *deltas* accumulate additively instead — activation
/// is accumulated knowledge, not last-write-wins data.
pub fn merge_activation(existing: &NodeActivation, incoming: &NodeActivation, now: i64) -> NodeActivation {
    let ex = decay_activation_state(existing, now, DEFAULT_ACTIVATION.score_half_life_ms, DEFAULT_ACTIVATION.importance_half_life_ms);
    let inc = decay_activation_state(incoming, now, DEFAULT_ACTIVATION.score_half_life_ms, DEFAULT_ACTIVATION.importance_half_life_ms);
    NodeActivation {
        score: ex.score.max(inc.score),
        importance: ex.importance.max(inc.importance),
        reinforcement_count: existing.reinforcement_count.max(incoming.reinforcement_count),
        last_meaningful_activation: now,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const DAY_MS: i64 = 24 * 3_600_000;

    fn activation(score: f64, importance: f64, count: u64, anchor: i64) -> NodeActivation {
        NodeActivation { score, importance, reinforcement_count: count, last_meaningful_activation: anchor }
    }

    #[test]
    fn decay_factor_halves_per_half_life() {
        assert_eq!(decay_factor(0, DAY_MS), 1.0);
        assert!((decay_factor(DAY_MS, DAY_MS) - 0.5).abs() < 1e-9);
        assert!((decay_factor(2 * DAY_MS, DAY_MS) - 0.25).abs() < 1e-9);
        assert_eq!(decay_factor(-1, DAY_MS), 1.0);
        assert_eq!(decay_factor(DAY_MS, 0), 1.0);
    }

    #[test]
    fn activation_score_of_decays_and_is_zero_without_activation() {
        let node = |activation: Option<NodeActivation>| Node {
            id: "a".into(),
            node_type: "t".into(),
            data: serde_json::Map::new(),
            vector: None,
            inserted_at: 1,
            updated_at: 1,
            revision: 0,
            activation,
        };
        let now = 1000;
        let n = node(Some(activation(1.0, 1.0, 1, now - DAY_MS)));
        assert!((activation_score_of(&n, now, DEFAULT_ACTIVATION.score_half_life_ms) - 0.5).abs() < 1e-9);
        assert_eq!(activation_score_of(&node(None), now, DEFAULT_ACTIVATION.score_half_life_ms), 0.0);
    }

    #[test]
    fn reinforce_activation_decay_corrects_then_anchors() {
        let now = 1000;
        let prev = activation(1.0, 0.5, 1, now - DAY_MS);
        let next = reinforce_activation(Some(&prev), 0.5, now, &DEFAULT_ACTIVATION);
        // Decayed score 0.5 + 0.5 = 1.0; importance folded in; re-anchored.
        assert!((next.score - 1.0).abs() < 1e-9);
        assert_eq!(next.reinforcement_count, 2);
        assert_eq!(next.last_meaningful_activation, now);
    }

    #[test]
    fn reinforce_activation_clamps() {
        let next = reinforce_activation(None, 5.0, 1000, &DEFAULT_ACTIVATION);
        assert_eq!(next.score, 1.0);
        assert_eq!(next.reinforcement_count, 1);
    }

    #[test]
    fn merge_activation_max_merges_and_re_anchors() {
        let now = 1000;
        let a = activation(0.6, 0.3, 2, now - DAY_MS);
        let b = activation(0.9, 0.1, 1, now);
        let merged = merge_activation(&a, &b, now);
        assert!((merged.score - 0.9).abs() < 1e-9);
        // a.importance decayed one day on the 30-day curve, still above b's.
        let ex_importance = 0.3 * decay_factor(DAY_MS, DEFAULT_ACTIVATION.importance_half_life_ms);
        assert!((merged.importance - ex_importance).abs() < 1e-9);
        assert_eq!(merged.reinforcement_count, 2);
        assert_eq!(merged.last_meaningful_activation, now);
    }

    #[test]
    fn node_serde_omits_activation_when_absent() {
        let n = Node {
            id: "a".into(),
            node_type: "t".into(),
            data: serde_json::Map::new(),
            vector: None,
            inserted_at: 1,
            updated_at: 1,
            revision: 0,
            activation: None,
        };
        let json = serde_json::to_value(&n).unwrap();
        assert!(json.get("activation").is_none());
        let with = Node { activation: Some(activation(0.5, 0.1, 1, 2)), ..n.clone() };
        let json = serde_json::to_value(&with).unwrap();
        assert_eq!(json["activation"]["score"], json!(0.5));
        let back: Node = serde_json::from_value(json).unwrap();
        assert_eq!(back.activation, with.activation);
    }
}
