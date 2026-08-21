import tempfile
import time

import pytest

from polypack import (
    PolyGraph,
    ActivationEngine,
    merge_activation,
    decay_factor,
    PolypackValueError,
)

DAY = 24 * 3_600_000


def _node(id_, **extra):
    node = {
        "id": id_,
        "type": "t",
        "data": {},
        "insertedAt": 1,
        "updatedAt": 1,
    }
    node.update(extra)
    return node


def _activation(score, importance, count, anchor):
    return {
        "score": score,
        "importance": importance,
        "reinforcementCount": count,
        "lastMeaningfulActivation": anchor,
    }


def test_reinforce_node_sets_activation():
    g = PolyGraph()
    g.add_node(_node("a"))
    updated = g.reinforce_node("a", 0.5, "user_read")
    assert updated["activation"]["score"] == pytest.approx(0.5, abs=1e-9)
    assert updated["activation"]["importance"] == pytest.approx(0.025, abs=1e-9)
    assert updated["activation"]["reinforcementCount"] == 1
    # get_activation/get_activation_state decay-correct against a fresh wall-clock
    # read, so allow for the (small) real elapsed time since reinforce_node wrote it.
    assert g.get_activation("a") == pytest.approx(0.5, abs=1e-5)
    assert g.get_activation_state("a")["score"] == pytest.approx(0.5, abs=1e-5)


def test_reinforce_node_clamps_and_accumulates():
    g = PolyGraph()
    g.add_node(_node("a"))
    g.reinforce_node("a", 5.0)
    assert g.get_activation_state("a")["score"] == pytest.approx(1.0, abs=1e-5)

    g.add_node(_node("b"))
    g.reinforce_node("b", 0.2)
    g.reinforce_node("b", 0.3)
    state = g.get_activation_state("b")
    assert state["score"] == pytest.approx(0.5, abs=1e-4)
    assert state["reinforcementCount"] == 2


def test_reinforce_node_missing_returns_none_and_rejects_bad_amount():
    g = PolyGraph()
    assert g.reinforce_node("missing", 0.5) is None
    with pytest.raises(PolypackValueError):
        g.reinforce_node("a", float("nan"))


def test_activation_validated_on_insert():
    g = PolyGraph()
    with pytest.raises(PolypackValueError):
        g.add_node(_node("bad", activation=_activation(2.0, 0.0, 0, 0)))
    with pytest.raises(PolypackValueError):
        g.add_node(_node("bad", activation=_activation(0.0, 0.0, 0, -1)))


def test_get_activation_decays_as_a_pure_function_of_time():
    g = PolyGraph()
    now = int(time.time() * 1000)
    g.add_node(_node("a", activation=_activation(1.0, 1.0, 3, now - DAY)))
    # Decay is evaluated against a fresh wall-clock read, not the captured `now`.
    assert g.get_activation("a") == pytest.approx(0.5, abs=1e-5)
    assert g.get_activation("a", DAY / 2) == pytest.approx(0.25, abs=1e-5)


def test_activation_persists_through_the_store():
    with tempfile.TemporaryDirectory() as d:
        with PolyGraph.open(d) as g:
            g.add_node(_node("a"))
            g.reinforce_node("a", 0.6)
            g.save()
        with PolyGraph.open(d) as g2:
            state = g2.get_activation_state("a")
            assert state["score"] == pytest.approx(0.6, abs=1e-4)
            assert state["reinforcementCount"] == 1


def test_top_activated_ranks_by_current_score():
    g = PolyGraph()
    for id_, amount in [("a", 0.9), ("b", 0.5), ("c", 0.1)]:
        g.add_node(_node(id_))
        g.reinforce_node(id_, amount)
    assert [n["id"] for n in g.top_activated(2)] == ["a", "b"]
    assert [n["id"] for n in g.top_activated(10, 0.6)] == ["a"]


def test_graph_query_activation_filters_and_orders():
    g = PolyGraph()
    for id_, amount in [("a", 0.9), ("b", 0.5), ("c", 0.1)]:
        g.add_node(_node(id_))
        g.reinforce_node(id_, amount)
    assert sorted(g.query().where_activated(0.4).ids()) == ["a", "b"]
    assert g.query().order_by_activation("desc").ids() == ["a", "b", "c"]
    assert g.query().order_by_activation("asc").ids() == ["c", "b", "a"]
    assert g.query().where_activated(0.4).order_by_activation("desc").ids() == ["a", "b"]


def test_merge_activation_max_merges_and_re_anchors():
    now = int(time.time() * 1000)
    a = _activation(0.6, 0.3, 2, now - DAY)
    b = _activation(0.9, 0.1, 1, now)
    merged = merge_activation(a, b, now)
    assert merged["score"] == pytest.approx(0.9, abs=1e-9)
    assert merged["reinforcementCount"] == 2
    assert merged["lastMeaningfulActivation"] == now


def test_decay_factor_halves_per_half_life():
    assert decay_factor(0, DAY) == 1.0
    assert decay_factor(DAY, DAY) == pytest.approx(0.5, abs=1e-9)
    assert decay_factor(2 * DAY, DAY) == pytest.approx(0.25, abs=1e-9)
    assert decay_factor(DAY, 0) == 1.0


def _vector_graph():
    g = PolyGraph()
    g.add_node(_node("ai", vector=[1.0, 0.0]))
    g.add_node(_node("nostr", vector=[0.0, 1.0]))
    g.add_node(_node("sports", vector=[0.0, 1.0]))
    g.add_edge("ai", "REL", "nostr")
    return g


def test_engine_spread_attenuates_per_hop():
    g = _vector_graph()
    g.add_node(_node("polypack"))
    g.add_edge("nostr", "REL", "polypack")
    engine = ActivationEngine(g)
    spread = engine.spread(["ai"], depth=2, decay=0.5)
    assert spread["nostr"] == pytest.approx(0.5, abs=1e-9)
    assert spread["polypack"] == pytest.approx(0.25, abs=1e-9)
    engine.dispose()


def test_engine_attention_stays_local_then_promotes():
    g = PolyGraph()
    g.add_node(_node("a"))
    g.reinforce_node("a", 0.5)
    engine = ActivationEngine(g)
    engine.bump_attention("a", 0.02)
    assert engine.attention_of("a") == pytest.approx(0.02, abs=1e-9)
    # effective()/get_activation() decay-correct against a fresh wall-clock read.
    assert engine.effective("a") == pytest.approx(0.52, abs=1e-5)
    assert g.get_activation("a") == pytest.approx(0.5, abs=1e-5)
    engine.bump_attention("a", 0.05)
    assert engine.attention_of("a") == 0.0
    assert g.get_activation("a") == pytest.approx(0.57, abs=1e-5)
    engine.dispose()


def test_engine_pulse_keeps_unrelated_nodes_cold():
    g = _vector_graph()
    engine = ActivationEngine(g)
    scores = engine.pulse([1.0, 0.0])
    ids = [id_ for id_, _ in scores]
    assert "ai" in ids
    assert "sports" not in ids
    engine.dispose()


def test_engine_absorb_reinforces_above_threshold():
    g = _vector_graph()
    engine = ActivationEngine(g, {"absorbThreshold": 0.3})
    engine.absorb([1.0, 0.0])
    assert g.get_activation_state("ai") is not None
    assert g.get_activation_state("sports") is None
    engine.dispose()


def test_engine_working_memory_returns_most_active():
    g = PolyGraph()
    for id_, amount in [("a", 0.9), ("b", 0.5), ("c", 0.1)]:
        g.add_node(_node(id_))
        g.reinforce_node(id_, amount)
    engine = ActivationEngine(g)
    assert [n["id"] for n in engine.working_memory(2)] == ["a", "b"]
    engine.dispose()


def test_reinforces_a_context_independently_of_the_global_score():
    g = PolyGraph()
    g.add_node(_node("a"))
    g.reinforce_node("a", 0.2)
    g.reinforce_node("a", 0.3, context="project-x")

    engine = ActivationEngine(g)
    assert engine.effective("a") == pytest.approx(0.5, abs=1e-5)
    assert engine.effective("a", "project-x") == pytest.approx(0.3, abs=1e-5)
    assert engine.effective("a", "coding") == 0.0
    assert g.get_context_activation("a", "project-x") == pytest.approx(0.3, abs=1e-5)
    engine.dispose()


def test_suppress_subtracts_inhibition_from_effective_but_not_the_raw_score():
    g = PolyGraph()
    g.add_node(_node("a"))
    g.reinforce_node("a", 0.8)
    g.suppress_node("a", 0.5)

    engine = ActivationEngine(g)
    assert engine.inhibition_of("a") == pytest.approx(0.5, abs=1e-5)
    assert engine.effective("a") == pytest.approx(0.3, abs=1e-5)
    assert g.get_activation("a") == pytest.approx(0.8, abs=1e-5)

    g.suppress_node("a", -0.5)
    assert engine.effective("a") == pytest.approx(0.8, abs=1e-5)
    engine.dispose()


def test_resolves_a_per_type_default_memory_class_and_decays_accordingly():
    g = PolyGraph()
    g.register_node_type("episode", memory_class="episodic")
    g.register_node_type("fact", memory_class="semantic")
    now = int(time.time() * 1000)
    HOUR = 3_600_000
    g.add_node({**_node("e1"), "type": "episode", "activation": _activation(0.8, 0.0, 1, now - 12 * HOUR)})
    g.add_node({**_node("f1"), "type": "fact", "activation": _activation(0.8, 0.0, 1, now - 12 * HOUR)})

    engine = ActivationEngine(g)
    # Episodic's 12h default half-life: exactly one half-life elapsed halves it.
    assert engine.effective("e1") == pytest.approx(0.4, abs=1e-3)
    # Semantic's half-life (7 days) is much longer, so far less decay at 12h.
    assert engine.effective("f1") > 0.7
    engine.dispose()


def test_per_node_memory_class_override_takes_precedence_over_the_type_default():
    g = PolyGraph()
    g.register_node_type("episode", memory_class="episodic")
    now = int(time.time() * 1000)
    HOUR = 3_600_000
    g.add_node({**_node("n1"), "type": "episode", "memoryClass": "entity", "activation": _activation(0.8, 0.0, 1, now - 12 * HOUR)})

    engine = ActivationEngine(g)
    assert engine.effective("n1") > 0.79
    engine.dispose()


def test_unclassified_nodes_keep_the_flat_default_half_life():
    g = PolyGraph()
    now = int(time.time() * 1000)
    g.add_node({**_node("n1"), "activation": _activation(0.8, 0.0, 1, now - DAY)})
    engine = ActivationEngine(g)
    assert engine.effective("n1") == pytest.approx(0.4, abs=1e-3)
    engine.dispose()


def test_working_memory_with_token_budget_stops_once_spent():
    g = PolyGraph()
    for id_, amount in [("a", 0.9), ("b", 0.5), ("c", 0.1)]:
        g.add_node(_node(id_))
        g.reinforce_node(id_, amount)
    engine = ActivationEngine(g)
    selected = engine.working_memory(limit=10, token_budget=2)
    assert [n["id"] for n in selected] == ["a", "b"]
    engine.dispose()


def test_working_memory_with_diversity_lambda_penalises_redundant_neighbours():
    g = PolyGraph()
    g.add_node(_node("a", vector=[1.0, 0.0]))
    g.add_node(_node("b", vector=[1.0, 0.01]))
    g.add_node(_node("c", vector=[0.0, 1.0]))
    g.reinforce_node("a", 0.9)
    g.reinforce_node("b", 0.85)
    g.reinforce_node("c", 0.5)

    engine = ActivationEngine(g)
    relevance_only = engine.working_memory(limit=2)
    assert [n["id"] for n in relevance_only] == ["a", "b"]

    diverse = engine.working_memory(limit=2, diversity_lambda=0.8)
    assert [n["id"] for n in diverse] == ["a", "c"]
    engine.dispose()
