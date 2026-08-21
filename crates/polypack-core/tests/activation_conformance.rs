//! Activation-math conformance cases shared with the TypeScript and Python
//! runners (`fixtures/conformance/activation-math.json`). Exercises the pure
//! decay/reinforce/merge functions directly — no `Graph` instance involved.

use polypack_core::activation::{decay_activation_state, merge_activation, reinforce_activation, suppress_activation, DEFAULT_ACTIVATION};
use polypack_core::model::NodeActivation;
use serde_json::Value;
use std::path::PathBuf;

const EPSILON: f64 = 1e-9;

fn fixture() -> Value {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../fixtures/conformance/activation-math.json");
    serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap()
}

fn activation_from(value: &Value) -> NodeActivation {
    serde_json::from_value(value.clone()).unwrap()
}

fn assert_activation_eq(name: &str, got: &NodeActivation, expect: &NodeActivation) {
    assert!(
        (got.score - expect.score).abs() < EPSILON,
        "{name}: score = {} != {}",
        got.score,
        expect.score
    );
    assert!(
        (got.importance - expect.importance).abs() < EPSILON,
        "{name}: importance = {} != {}",
        got.importance,
        expect.importance
    );
    assert_eq!(
        got.reinforcement_count, expect.reinforcement_count,
        "{name}: reinforcementCount mismatch"
    );
    assert_eq!(
        got.last_meaningful_activation, expect.last_meaningful_activation,
        "{name}: lastMeaningfulActivation mismatch"
    );
    let got_inhibition = got.inhibition.unwrap_or(0.0);
    let expect_inhibition = expect.inhibition.unwrap_or(0.0);
    assert!(
        (got_inhibition - expect_inhibition).abs() < EPSILON,
        "{name}: inhibition = {got_inhibition} != {expect_inhibition}"
    );
    let empty = std::collections::HashMap::new();
    let expect_context = expect.context.as_ref().unwrap_or(&empty);
    let got_context = got.context.as_ref().unwrap_or(&empty);
    for (key, entry) in expect_context {
        let got_score = got_context.get(key).map(|e| e.score).unwrap_or(0.0);
        assert!(
            (got_score - entry.score).abs() < EPSILON,
            "{name}: context[{key}].score = {got_score} != {}",
            entry.score
        );
    }
    for (key, entry) in got_context {
        if !expect_context.contains_key(key) {
            assert!(entry.score <= EPSILON, "{name}: unexpected context[{key}] with score {}", entry.score);
        }
    }
}

#[test]
fn activation_math_fixture_passes() {
    let fixture = fixture();
    assert_eq!(fixture["schemaVersion"], 1);
    assert_eq!(fixture["group"], "activation");

    for case in fixture["activationCases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let now = case["now"].as_i64().unwrap();
        let kind = case["kind"].as_str().unwrap();

        let got = match kind {
            "decay" => {
                let input = activation_from(&case["input"]);
                let score_half_life_ms = case
                    .get("scoreHalfLifeMs")
                    .and_then(Value::as_i64)
                    .unwrap_or(DEFAULT_ACTIVATION.score_half_life_ms);
                let importance_half_life_ms = case
                    .get("importanceHalfLifeMs")
                    .and_then(Value::as_i64)
                    .unwrap_or(DEFAULT_ACTIVATION.importance_half_life_ms);
                decay_activation_state(&input, now, score_half_life_ms, importance_half_life_ms)
            }
            "reinforce" => {
                let previous = case["previous"].as_object().map(|_| activation_from(&case["previous"]));
                let delta = case["delta"].as_f64().unwrap();
                let context = case.get("context").and_then(Value::as_str);
                reinforce_activation(previous.as_ref(), delta, now, &DEFAULT_ACTIVATION, context)
            }
            "suppress" => {
                let previous = case["previous"].as_object().map(|_| activation_from(&case["previous"]));
                let delta = case["delta"].as_f64().unwrap();
                let inhibition_half_life_ms = case
                    .get("inhibitionHalfLifeMs")
                    .and_then(Value::as_i64)
                    .unwrap_or(DEFAULT_ACTIVATION.inhibition_half_life_ms);
                suppress_activation(previous.as_ref(), delta, now, inhibition_half_life_ms)
            }
            "merge" => {
                let existing = activation_from(&case["existing"]);
                let incoming = activation_from(&case["incoming"]);
                merge_activation(&existing, &incoming, now)
            }
            other => panic!("unsupported activation case kind {other}"),
        };

        let expect = activation_from(&case["expect"]);
        assert_activation_eq(name, &got, &expect);
    }
}
