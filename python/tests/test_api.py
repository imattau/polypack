"""Smoke tests for the Python API surface."""

import numpy as np
import pytest
import json
from pathlib import Path

import polypack
from polypack import (
    ChangeBatch,
    AdapterCapabilityError,
    ExactIndex,
    GraphSnapshot,
    HnswIndex,
    PolyGraph,
    PolypackDimensionError,
    PolypackError,
    PolypackStorageError,
    PolypackValueError,
    engine_info,
)


def test_engine_info():
    info = engine_info()
    assert info["vector"] == "rust-native"


def test_exact_index_numpy_and_lists():
    idx = ExactIndex("cosine")
    idx.add("a", np.array([1.0, 0.0, 0.0]))
    idx.add("b", [0.8, 0.6, 0.0])
    assert idx.query(np.array([1.0, 0.0, 0.0]), 2) == [("a", 1.0), ("b", 0.8)]
    assert idx.get("a") == [1.0, 0.0, 0.0]
    assert idx.has("a")
    assert len(idx) == 2
    assert sorted(i for i, _ in idx.entries()) == ["a", "b"]


def test_exact_index_dimension_error():
    idx = ExactIndex()
    idx.add("a", [1.0, 0.0])
    with pytest.raises(PolypackDimensionError):
        idx.query([1.0, 0.0, 0.0], 1)


def test_hnsw_index_churn():
    idx = HnswIndex(ef_search=300)
    idx.add("a", [1.0, 0.0, 0.0])
    idx.add("b", [0.0, 1.0, 0.0])
    idx.remove("a")
    assert len(idx) == 1
    assert not idx.has("a")
    idx.add("a", [1.0, 0.0, 0.0])
    idx.update("b", [1.0, 0.0, 0.0])
    assert sorted(i for i, _ in idx.query([1.0, 0.0, 0.0], 2)) == ["a", "b"]


@pytest.mark.parametrize("kwarg", ["m", "mmax0", "ef_construction", "ef_search"])
def test_hnsw_index_rejects_non_positive_config(kwarg):
    with pytest.raises(PolypackValueError):
        HnswIndex(**{kwarg: 0})


def test_graph_ownership_cascade():
    graph = PolyGraph()
    graph.add_node({"id": "a", "type": "user", "data": {}, "insertedAt": 1, "updatedAt": 1})
    graph.add_node({"id": "d", "type": "doc", "data": {}, "insertedAt": 1, "updatedAt": 1})
    graph.add_edge("a", "OWNS", "d", ownership="owned")
    graph.remove_node("a")
    assert graph.get_node("d") is None


def test_graph_query_and_vectors():
    graph = PolyGraph()
    for i, v in enumerate([[1, 0, 0], [0.8, 0.6, 0], [0, 1, 0]]):
        graph.add_node(
            {"id": f"n{i}", "type": "doc", "data": {"score": i}, "vector": v, "insertedAt": i, "updatedAt": i}
        )
    assert graph.query().order_by("score", "desc").ids() == ["n2", "n1", "n0"]
    assert graph.query().similar_to([1, 0, 0], 0.5, 1).ids() == ["n0"]
    assert [r[0] for r in graph.vectors.query([1, 0, 0], 2)] == ["n0", "n1"]


def test_secondary_indexes_filter_equality_and_numeric_ranges():
    graph = PolyGraph()
    graph.define_index("email", ["email"], unique=True)
    graph.define_index("birth-year", ["birthYear"], sparse=True)
    graph.add_node({"id": "a", "type": "person", "data": {"email": "a@test", "birthYear": 1980}, "insertedAt": 1, "updatedAt": 1})
    graph.add_node({"id": "b", "type": "person", "data": {"email": "b@test", "birthYear": 2020}, "insertedAt": 1, "updatedAt": 1})
    graph.add_node({"id": "c", "type": "person", "data": {"email": "c@test", "birthYear": 2050}, "insertedAt": 1, "updatedAt": 1})
    assert graph.query().where("email", "b@test").ids() == ["b"]
    assert sorted(graph.query().where_range("birthYear", above=2000).ids()) == ["b", "c"]
    graph.update_node("b", {"email": "updated@test"})
    assert graph.query().where("email", "b@test").ids() == []
    assert graph.query().where("email", "updated@test").ids() == ["b"]


def test_indexed_query_metrics_report_candidate_scan_count():
    graph = PolyGraph()
    graph.define_index("email", ["email"])
    for id_, email in [("a", "a@test"), ("b", "b@test")]:
        graph.add_node({"id": id_, "type": "person", "data": {"email": email}, "insertedAt": 1, "updatedAt": 1})
    assert graph.query().where("email", "a@test").ids() == ["a"]
    stats = graph.stats()
    assert stats["queryIndexUsage"]["email"] == 1
    assert stats["queryScannedRecords"] == 1


def test_index_intersection_reports_and_uses_all_matching_indexes():
    graph = PolyGraph()
    graph.define_index("surname", ["surname"])
    graph.define_index("birth-year", ["birthYear"])
    for id_, surname, year in [("a", "Smith", 1980), ("b", "Smith", 1990), ("c", "Jones", 1980)]:
        graph.add_node({"id": id_, "type": "person", "data": {"surname": surname, "birthYear": year}, "insertedAt": 1, "updatedAt": 1})

    query = graph.query().where("surname", "Smith").where("birthYear", 1980)
    explanation = query.explain()
    assert explanation["indexes"] == ["surname", "birth-year"]
    assert "index-intersection(2)" in explanation["stages"]
    assert query.ids() == ["a"]
    assert graph.stats()["queryIndexUsage"] == {"surname": 1, "birth-year": 1}


def test_index_catalog_rolls_back_when_metadata_write_fails(tmp_path, monkeypatch):
    graph = PolyGraph.open(str(tmp_path))
    graph.define_index("email", ["email"])

    def fail_persist():
        raise PolypackStorageError("metadata write failed")

    monkeypatch.setattr(graph, "_persist_index_metadata", fail_persist)
    with pytest.raises(PolypackStorageError):
        graph.define_index("birth-year", ["birthYear"])
    assert [index["name"] for index in graph.indexes] == ["email"]
    with pytest.raises(PolypackStorageError):
        graph.drop_index("email")
    assert [index["name"] for index in graph.indexes] == ["email"]


def test_clear_preserves_attached_store_index_constraints(tmp_path):
    graph = PolyGraph.open(str(tmp_path))
    graph.define_index("email", ["email"], unique=True)
    graph.add_node({"id": "a", "type": "person", "data": {"email": "a@test"}, "insertedAt": 1, "updatedAt": 1})
    graph.save()
    graph.clear()
    with pytest.raises(polypack.UniqueConstraintError):
        graph.add_node({"id": "b", "type": "person", "data": {"email": "a@test"}, "insertedAt": 1, "updatedAt": 1})


def test_validation_errors():
    graph = PolyGraph()
    with pytest.raises(PolypackValueError):
        graph.add_node({"id": "", "type": "t", "data": {}, "insertedAt": 1, "updatedAt": 1})
    with pytest.raises(PolypackError):
        graph.add_edge("a::b", "REL", "c")


def test_adapter_capability_requirements():
    graph = PolyGraph()
    graph.require_adapter_capabilities({"transactions": True, "vectorSearch": "exact"})
    assert graph.capabilities["changeFeed"] is True
    with pytest.raises(AdapterCapabilityError, match="fsync"):
        graph.require_adapter_capabilities(fsync=True)


def test_index_metadata_rejects_duplicate_or_empty_definitions(tmp_path):
    (tmp_path / "indexes.json").write_text(
        '[{"name":"dup","fields":["email"]},{"name":"dup","fields":[]}]',
        encoding="utf-8",
    )
    with pytest.raises(PolypackStorageError, match="invalid index metadata"):
        PolyGraph.open(str(tmp_path))


def test_schema_metadata_persists_and_reloads(tmp_path):
    graph = PolyGraph.open(str(tmp_path))
    graph.register_node_type("person", required_fields=["name"], data_types={"age": "integer"})
    graph.register_edge_type("PARENT_OF", source_types=["person"], target_types=["person"], cardinality="many-to-many")
    graph.close_store()

    reopened = PolyGraph.open(str(tmp_path))
    with pytest.raises(PolypackValueError, match="required field name"):
        reopened.add_node({"id": "p1", "type": "person", "data": {}, "insertedAt": 1, "updatedAt": 1})
    reopened.add_node({"id": "p1", "type": "person", "data": {"name": "A", "age": 1}, "insertedAt": 1, "updatedAt": 1})
    with pytest.raises(PolypackValueError, match="missing endpoint"):
        reopened.add_edge("p1", "PARENT_OF", "missing")
    reopened.close_store()


def test_schema_metadata_rejects_duplicate_definitions(tmp_path):
    (tmp_path / "schemas.json").write_text(
        '{"nodeTypes":[{"nodeType":"person","requiredFields":[],"dataTypes":{}},{"nodeType":"person","requiredFields":[],"dataTypes":{}}],"edgeTypes":[]}',
        encoding="utf-8",
    )
    with pytest.raises(PolypackStorageError, match="invalid schema metadata"):
        PolyGraph.open(str(tmp_path))


def test_context_manager_and_change_batch():
    with PolyGraph() as graph:
        graph.add_node({"id": "a", "type": "t", "data": {}, "insertedAt": 1, "updatedAt": 1})
    assert graph.size == 0
    batch = ChangeBatch(put_nodes=[{"id": "x", "type": "t"}])
    batch.validate()
    with pytest.raises(PolypackValueError):
        ChangeBatch(put_nodes=[{"id": "", "type": "t"}]).validate()


def test_snapshot_is_detached_and_queryable():
    graph = PolyGraph()
    _seed_graph(graph)

    snapshot = graph.snapshot()
    assert isinstance(snapshot, GraphSnapshot)
    assert snapshot.get_node("n1")["data"] == {"title": "Hello"}
    assert len(snapshot.get_edges("n1", "LINKS")) == 1
    assert [node["id"] for node in snapshot.query().where_type("doc").to_list()] == ["n1", "n2"]

    graph.update_node("n1", {"title": "changed"})
    graph.add_node({"id": "n3", "type": "doc", "data": {}, "insertedAt": 3, "updatedAt": 3})
    assert snapshot.get_node("n1")["data"] == {"title": "Hello"}
    assert snapshot.get_node("n3") is None


def test_migrations_reject_identity_changes():
    graph = PolyGraph()
    graph.add_node({"id": "a", "type": "record", "data": {}, "insertedAt": 1, "updatedAt": 1})
    graph.migrations.register({"from": 1, "to": 2, "migrateNode": lambda node: {**node, "id": "changed"}})
    with pytest.raises(polypack.MigrationError, match="changed identity or type"):
        graph.migrations.run(graph, 1, 2)


def test_patch_compare_and_set_is_conditional():
    graph = PolyGraph()
    graph.add_node({"id": "a", "type": "t", "data": {"state": "ready"}, "insertedAt": 1, "updatedAt": 1})

    updated = graph.patch_node("a", compare_and_set={"data.state": {"expected": "ready", "value": "done"}})
    assert updated["data"]["state"] == "done"
    with pytest.raises(polypack.ConflictError):
        graph.patch_node("a", compare_and_set={"data.state": {"expected": "ready", "value": "stale"}})


def _seed_graph(graph):
    graph.add_node({"id": "n1", "type": "doc", "data": {"title": "Hello"}, "vector": [0.1, 0.2, 0.3], "insertedAt": 1, "updatedAt": 1})
    graph.add_node({"id": "n2", "type": "doc", "data": {}, "vector": None, "insertedAt": 2, "updatedAt": 2})
    graph.add_edge("n1", "LINKS", "n2", ownership="reference")


def test_persist_round_trip(tmp_path):
    g = PolyGraph()
    _seed_graph(g)
    g.open_store(str(tmp_path))
    g.save()
    g.close_store()

    g2 = PolyGraph.open(str(tmp_path))
    assert g2.get_node("n1")["vector"] == [0.1, 0.2, 0.3]
    assert g2.get_node("n1")["data"] == {"title": "Hello"}
    assert g2.get_edge_targets("n1", "LINKS") == ["n2"]
    g2.close_store()


def test_latest_mutation_sequence_cursor(tmp_path):
    graph = PolyGraph.open(str(tmp_path))
    graph.add_node({"id": "a", "type": "t", "data": {}, "insertedAt": 1, "updatedAt": 1})
    graph.save()
    assert graph.latest_mutation_sequence() == graph.mutation_log()[-1]["sequence"]


def test_durable_mutation_log_fixture(tmp_path):
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "database-core" / "durable-mutation-log.json"
    fixture = json.loads(fixture_path.read_text())
    graph = PolyGraph.open(str(tmp_path))
    graph.transaction(
        lambda tx: tx.add_node(fixture["transaction"]["node"]),
        operation_id=fixture["transaction"]["operationId"],
    )
    records = graph.mutation_log_since(0)
    assert graph.latest_mutation_sequence() == fixture["expect"]["latestSequence"]
    assert records[-1]["operationId"] == fixture["expect"]["operationId"]
    assert records[-1]["operations"][0]["operationType"] == fixture["expect"]["operationType"]
    assert records[-1]["operations"][0]["payload"]["id"] == fixture["expect"]["nodeId"]


def test_context_manager_saves_before_closing_the_store(tmp_path):
    # No explicit save() — closing the `with` block must not silently lose
    # mutations, mirroring Rust's Graph::close (which flushes before closing).
    with PolyGraph.open(str(tmp_path)) as g:
        _seed_graph(g)

    with PolyGraph.open(str(tmp_path)) as g2:
        assert g2.get_node("n1")["data"] == {"title": "Hello"}
        assert g2.get_edge_targets("n1", "LINKS") == ["n2"]


def test_persist_recovers_from_truncated_wal(tmp_path):
    g = PolyGraph()
    _seed_graph(g)
    g.open_store(str(tmp_path))
    g.save()
    # No snapshot is written until close/compact.
    assert not (tmp_path / "snapshot.msgpack").exists()
    wal = (tmp_path / "wal.msgpack").read_bytes()
    assert len(wal) > 0
    # Crash mid-append: the final in-flight frame is partial. Recovery must
    # still read the acknowledged frames before it.
    (tmp_path / "wal.msgpack").write_bytes(wal[: len(wal) - 2])
    # Release the writer lock without saving or compacting, simulating the
    # original process exiting before recovery starts.
    g._store.close()
    g._store = None

    g2 = PolyGraph.open(str(tmp_path))
    assert g2.get_node("n1") is not None
    assert g2.get_node("n2") is not None
    g2.close_store()
    # Recovery already compacted the truncated WAL down to a snapshot; the
    # save() that close_store() now does first (to avoid discarding unsaved
    # changes) re-applies the unchanged state, so the WAL is empty rather
    # than absent — both mean "nothing left to replay".
    assert (tmp_path / "snapshot.msgpack").exists()
    wal_file = tmp_path / "wal.msgpack"
    assert not wal_file.exists() or wal_file.stat().st_size == 0


def test_transaction_conformance_fixture():
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "database-core" / "transaction.json"
    fixture = json.loads(fixture_path.read_text())
    graph = PolyGraph()
    graph.add_node(fixture["setup"]["nodes"][0])
    tx_spec = fixture["transaction"]

    def apply(tx):
        patch = tx_spec["patch"]
        tx.patch_node(
            patch["id"],
            increment=patch["increment"],
            expected_revision=patch["expectedRevision"],
        )
        tx.add_node(tx_spec["addNode"])
        edge = tx_spec["addEdge"]
        tx.add_edge(edge["source"], edge["type"], edge["target"])
        assert tx.get_node(tx_spec["readYourWrites"]["id"])["data"]["count"] == tx_spec["readYourWrites"]["count"]

    graph.transaction(apply)
    assert graph.size == fixture["expect"]["nodeCount"]
    assert graph.get_node("person-1")["data"]["count"] == fixture["expect"]["person1Count"]
    assert graph.get_node("person-1")["revision"] == fixture["expect"]["person1Revision"]
    assert graph.get_edge_targets("person-1", "RELATED_TO") == fixture["expect"]["edgeTargets"]

    def rollback(tx):
        rollback_spec = fixture["rollback"]
        tx.patch_node(
            rollback_spec["patch"]["id"],
            set=rollback_spec["patch"]["set"],
            expected_revision=rollback_spec["patch"]["expectedRevision"],
        )
        tx.add_node(rollback_spec["addNode"])
        raise RuntimeError("rollback fixture failure")

    with pytest.raises(RuntimeError, match="rollback fixture failure"):
        graph.transaction(rollback)
    assert graph.size == fixture["expect"]["rollbackCount"]
    assert graph.get_node("temporary") is None
    assert graph.get_node("person-1")["data"]["count"] == fixture["expect"]["person1Count"]
    assert graph.get_node("person-1")["revision"] == fixture["expect"]["rollbackRevision"]


def test_schema_and_unique_index_conformance_fixture():
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "database-core" / "schema-and-indexes.json"
    fixture = json.loads(fixture_path.read_text())
    graph = PolyGraph()
    node_type = fixture["nodeType"]
    graph.register_node_type(
        node_type["name"],
        required_fields=node_type["requiredFields"],
        data_types=node_type["dataTypes"],
    )
    graph.define_index(fixture["index"])
    graph.add_node(fixture["validNode"])
    with pytest.raises(PolypackError):
        graph.add_node(fixture["invalidNode"])
    with pytest.raises(polypack.UniqueConstraintError):
        graph.add_node(fixture["duplicateNode"])
    assert graph.size == fixture["expect"]["nodeCount"]
    assert graph.get_node(fixture["expect"]["presentId"])["data"]["name"] == "Mary"


def test_secondary_index_conformance_fixture():
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "database-core" / "secondary-indexes.json"
    fixture = json.loads(fixture_path.read_text())
    graph = PolyGraph()
    for index in fixture["indexes"]:
        graph.define_index(index)
    for node in fixture["nodes"]:
        graph.add_node(node)
    query = graph.query().where("surname", fixture["query"]["surname"]).where("birthYear", fixture["query"]["birthYear"])
    explanation = query.explain()
    assert query.ids() == fixture["expect"]["ids"]
    assert sorted(explanation["indexes"]) == sorted(fixture["expect"]["indexes"])
    assert fixture["expect"]["intersectionStage"] in explanation["stages"]


def test_migration_conformance_fixture():
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "database-core" / "migration.json"
    fixture = json.loads(fixture_path.read_text())
    graph = PolyGraph()
    for node in fixture["nodes"]:
        graph.add_node(node)
    graph.migrations.register({
        "from": fixture["from"],
        "to": fixture["to"],
        "migrateNode": lambda node: {**node, "data": {**node["data"], "displayName": node["data"]["name"]}},
    })
    report = graph.migrations.run(graph, fixture["from"], fixture["to"], {"batchSize": 1})
    assert report["migrated"] == fixture["expect"]["migrated"]
    assert sorted(graph.query().ids()) == fixture["expect"]["ids"]
    for id_ in fixture["expect"]["ids"]:
        assert graph.get_node(id_)["data"]["displayName"] == fixture["expect"]["displayNames"][id_]


def test_parallel_edge_identity_conformance_fixture():
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "database-core" / "parallel-edges.json"
    fixture = json.loads(fixture_path.read_text())
    graph = PolyGraph()
    for node in fixture["nodes"]:
        graph.add_node(node)
    for edge in fixture["edges"]:
        graph.add_edge(edge["source"], edge["type"], edge["target"], edge["data"], id=edge["id"])
    update = fixture["update"]
    graph.update_edge(update["id"], update["data"], expected_revision=update["expectedRevision"])
    remove = fixture["remove"]
    assert graph.remove_edge(remove["id"], expected_revision=remove["expectedRevision"])
    edges = graph.get_edges("a", "RELATED")
    assert [edge["id"] for edge in edges] == fixture["expect"]["edgeIds"]
    assert edges[0]["revision"] == fixture["expect"]["revision"]
    assert edges[0]["data"]["confidence"] == fixture["expect"]["confidence"]


def test_snapshot_isolation_conformance_fixture():
    fixture_path = Path(__file__).resolve().parents[2] / "fixtures" / "database-core" / "snapshot-isolation.json"
    fixture = json.loads(fixture_path.read_text())
    graph = PolyGraph()
    for node in fixture["nodes"]:
        graph.add_node(node)
    snapshot = graph.snapshot()
    graph.add_node(fixture["mutation"]["add"])
    graph.remove_node(fixture["mutation"]["remove"])
    assert sorted(snapshot.query().ids()) == fixture["expect"]["snapshotIds"]
    assert sorted(graph.query().ids()) == fixture["expect"]["liveIds"]
