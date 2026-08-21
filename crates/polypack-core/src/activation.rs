//! Pure activation math, shared by `polypack-graph`'s `Graph`, the query
//! builders, and the `ActivationEngine`. Mirrors the activation helpers in the
//! TypeScript `src/utils.ts`.
//!
//! Decay is a pure function of elapsed time anchored at
//! `last_meaningful_activation`, so replicas with the same stored state and
//! their own clocks compute identical current scores.

use crate::model::{ContextActivation, Node, NodeActivation};
use std::collections::HashMap;

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
    /// Half-life of `inhibition` (12 h) — shorter than `score` by default, so suppression fades unless reinforced.
    pub inhibition_half_life_ms: i64,
    /// Half-life of a per-context `score` entry. Defaults to the same curve as global `score`.
    pub context_half_life_ms: i64,
}

/// Default activation decay curves, shared by `polypack-core` and
/// `polypack-graph`.
pub const DEFAULT_ACTIVATION: ActivationCurves = ActivationCurves {
    score_half_life_ms: DAY,
    importance_half_life_ms: 30 * DAY,
    importance_gain: 0.05,
    inhibition_half_life_ms: 12 * HOUR,
    context_half_life_ms: DAY,
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

fn decay_context_entry(entry: &ContextActivation, now: i64, half_life_ms: i64) -> ContextActivation {
    ContextActivation {
        score: clamp01(entry.score * decay_factor(now - entry.last_meaningful_activation, half_life_ms)),
        last_meaningful_activation: entry.last_meaningful_activation,
    }
}

/// Read-only decay-corrected view of an activation record. Does NOT re-anchor
/// `last_meaningful_activation`; the stored values always represent the state
/// at that anchor, so reinforcement/merge can decay-correct and re-anchor.
/// Also decay-corrects `inhibition` (own half-life) and every `context` entry
/// (each against its own anchor) when present.
pub fn decay_activation_state(
    activation: &NodeActivation,
    now: i64,
    score_half_life_ms: i64,
    importance_half_life_ms: i64,
) -> NodeActivation {
    decay_activation_state_full(
        activation,
        now,
        score_half_life_ms,
        importance_half_life_ms,
        DEFAULT_ACTIVATION.inhibition_half_life_ms,
        DEFAULT_ACTIVATION.context_half_life_ms,
    )
}

/// Full-control variant of [`decay_activation_state`] with explicit inhibition
/// and context half-lives.
pub fn decay_activation_state_full(
    activation: &NodeActivation,
    now: i64,
    score_half_life_ms: i64,
    importance_half_life_ms: i64,
    inhibition_half_life_ms: i64,
    context_half_life_ms: i64,
) -> NodeActivation {
    let inhibition = activation.inhibition.map(|inhibition| {
        clamp01(
            inhibition
                * decay_factor(
                    now - activation.last_inhibited_at.unwrap_or(activation.last_meaningful_activation),
                    inhibition_half_life_ms,
                ),
        )
    });
    let context = activation.context.as_ref().map(|context| {
        context
            .iter()
            .map(|(key, entry)| (key.clone(), decay_context_entry(entry, now, context_half_life_ms)))
            .collect::<HashMap<_, _>>()
    });
    NodeActivation {
        score: clamp01(activation.score * decay_factor(now - activation.last_meaningful_activation, score_half_life_ms)),
        importance: clamp01(
            activation.importance * decay_factor(now - activation.last_meaningful_activation, importance_half_life_ms),
        ),
        reinforcement_count: activation.reinforcement_count,
        last_meaningful_activation: activation.last_meaningful_activation,
        inhibition,
        last_inhibited_at: activation.last_inhibited_at,
        context,
    }
}

/// Apply a reinforcement delta: decay-correct the prior state to `now`, add
/// `delta` to `score`, fold a fraction (`importance_gain`) into `importance`,
/// increment the reinforcement counter, and re-anchor
/// `last_meaningful_activation` to `now` so the stored record again represents
/// "value at the anchor". When `context` is given, the same delta additionally
/// reinforces `activation.context[context]` — an independently-decaying,
/// additional lens, not a replacement for the global score.
pub fn reinforce_activation(
    previous: Option<&NodeActivation>,
    delta: f64,
    now: i64,
    curves: &ActivationCurves,
    context: Option<&str>,
) -> NodeActivation {
    let decayed = previous.map(|prev| decay_activation_state(prev, now, curves.score_half_life_ms, curves.importance_half_life_ms));
    let (score, importance) = decayed.as_ref().map(|d| (d.score, d.importance)).unwrap_or((0.0, 0.0));

    let mut context_map = decayed
        .as_ref()
        .and_then(|d| d.context.clone())
        .or_else(|| previous.and_then(|p| p.context.clone()));
    if let Some(ctx) = context {
        let entry = context_map.get_or_insert_with(HashMap::new).entry(ctx.to_string()).or_insert(ContextActivation {
            score: 0.0,
            last_meaningful_activation: now,
        });
        entry.score = clamp01(entry.score + delta);
        entry.last_meaningful_activation = now;
    }

    NodeActivation {
        score: clamp01(score + delta),
        importance: clamp01(importance + curves.importance_gain * delta),
        reinforcement_count: previous.map(|p| p.reinforcement_count + 1).unwrap_or(1),
        last_meaningful_activation: now,
        inhibition: decayed.as_ref().and_then(|d| d.inhibition),
        last_inhibited_at: decayed.as_ref().and_then(|d| d.last_inhibited_at),
        context: context_map,
    }
}

/// Apply a suppression delta to an activation record's `inhibition` (mirrors
/// [`reinforce_activation`] but for the inhibition axis): decay-correct the
/// prior inhibition to `now`, add `delta`, clamp, and re-anchor
/// `last_inhibited_at`. A negative `delta` releases suppression. `score` and
/// `importance` are only decay-corrected, not re-anchored.
pub fn suppress_activation(
    previous: Option<&NodeActivation>,
    delta: f64,
    now: i64,
    inhibition_half_life_ms: i64,
) -> NodeActivation {
    let decayed = previous.map(|prev| {
        decay_activation_state_full(
            prev,
            now,
            DEFAULT_ACTIVATION.score_half_life_ms,
            DEFAULT_ACTIVATION.importance_half_life_ms,
            inhibition_half_life_ms,
            DEFAULT_ACTIVATION.context_half_life_ms,
        )
    });
    NodeActivation {
        score: decayed.as_ref().map(|d| d.score).unwrap_or(0.0),
        importance: decayed.as_ref().map(|d| d.importance).unwrap_or(0.0),
        reinforcement_count: previous.map(|p| p.reinforcement_count).unwrap_or(0),
        last_meaningful_activation: decayed.as_ref().map(|d| d.last_meaningful_activation).unwrap_or(now),
        inhibition: Some(clamp01(decayed.as_ref().and_then(|d| d.inhibition).unwrap_or(0.0) + delta)),
        last_inhibited_at: Some(now),
        context: decayed.as_ref().and_then(|d| d.context.clone()),
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

    let inhibition = if ex.inhibition.is_some() || inc.inhibition.is_some() {
        Some(ex.inhibition.unwrap_or(0.0).max(inc.inhibition.unwrap_or(0.0)))
    } else {
        None
    };
    let context = if ex.context.is_some() || inc.context.is_some() {
        let mut merged: HashMap<String, ContextActivation> = HashMap::new();
        let empty = HashMap::new();
        let ex_context = ex.context.as_ref().unwrap_or(&empty);
        let inc_context = inc.context.as_ref().unwrap_or(&empty);
        for key in ex_context.keys().chain(inc_context.keys()).cloned().collect::<std::collections::HashSet<_>>() {
            let a = ex_context.get(&key).map(|e| e.score).unwrap_or(0.0);
            let b = inc_context.get(&key).map(|e| e.score).unwrap_or(0.0);
            merged.insert(key, ContextActivation { score: a.max(b), last_meaningful_activation: now });
        }
        Some(merged)
    } else {
        None
    };

    NodeActivation {
        score: ex.score.max(inc.score),
        importance: ex.importance.max(inc.importance),
        reinforcement_count: existing.reinforcement_count.max(incoming.reinforcement_count),
        last_meaningful_activation: now,
        inhibition,
        last_inhibited_at: inhibition.map(|_| now),
        context,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    const DAY_MS: i64 = 24 * 3_600_000;

    fn activation(score: f64, importance: f64, count: u64, anchor: i64) -> NodeActivation {
        NodeActivation { score, importance, reinforcement_count: count, last_meaningful_activation: anchor, ..Default::default() }
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
            activation, ..Default::default()
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
        let next = reinforce_activation(Some(&prev), 0.5, now, &DEFAULT_ACTIVATION, None);
        // Decayed score 0.5 + 0.5 = 1.0; importance folded in; re-anchored.
        assert!((next.score - 1.0).abs() < 1e-9);
        assert_eq!(next.reinforcement_count, 2);
        assert_eq!(next.last_meaningful_activation, now);
    }

    #[test]
    fn reinforce_activation_clamps() {
        let next = reinforce_activation(None, 5.0, 1000, &DEFAULT_ACTIVATION, None);
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
            ..Default::default()
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
