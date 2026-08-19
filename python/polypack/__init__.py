"""Polypack — embedded local-first property graph with vector search.

Python-native graph layer over the Rust vector core (`polypack._core`).
Semantics mirror `specification/data-model.md` and the TypeScript reference.
"""

from __future__ import annotations

import copy
import ast
import fcntl
import json
import math
import os
import shutil
import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

import numpy as np

from ._core import (
    ExactIndex as _NativeExactIndex,
    HnswIndex as _NativeHnswIndex,
    NativeStore as _NativeStore,
    engine_info as _engine_info,
    execute_query_plan as _execute_query_plan,
)
from ._core import (
    PolypackClosedError,
    PolypackCorruptDataError,
    PolypackDimensionError,
    PolypackError,
    PolypackStorageError,
    PolypackValueError,
    PolypackVersionError,
)


class ConflictError(PolypackError):
    """Raised when a conditional write observes a stale record revision."""

    pass


class ResourceLimitError(PolypackError):
    """Raised when a configured graph or query resource limit is exceeded."""

    def __init__(self, limit_name: str, limit: int) -> None:
        super().__init__(f"Resource exceeded {limit_name} limit of {limit}")
        self.limit_name = limit_name
        self.limit = limit


class MigrationError(PolypackError):
    """Raised when an application migration is invalid or cannot run."""


class MigrationRegistry:
    """Registry for resumable application-level node and edge migrations."""

    def __init__(self) -> None:
        self._definitions: dict[int, dict] = {}

    def register(self, definition: dict) -> None:
        source = definition.get("from")
        target = definition.get("to")
        if not isinstance(source, int) or not isinstance(target, int) or target <= source:
            raise MigrationError("migration versions must be integers with to greater than from")
        if source in self._definitions:
            raise MigrationError(f"migration from {source} is already registered")
        if not callable(definition.get("migrateNode")):
            raise MigrationError("migration requires migrateNode")
        self._definitions[source] = dict(definition)

    @property
    def all(self) -> list[dict]:
        return [self._definitions[key] for key in sorted(self._definitions)]

    def _path(self, source: int, target: int) -> list[dict]:
        path = []
        version = source
        while version < target:
            definition = self._definitions.get(version)
            if definition is None or definition["to"] > target:
                raise MigrationError(f"no contiguous migration path from {version} to {target}")
            path.append(definition)
            version = definition["to"]
        return path

    def run(self, graph: "PolyGraph", source: int, target: int, options: Optional[dict] = None) -> dict:
        options = dict(options or {})
        if not isinstance(source, int) or not isinstance(target, int) or target < source:
            raise MigrationError("migration versions must be integers with to greater than or equal to from")
        batch_size = options.get("batchSize", 2**63 - 1)
        if not isinstance(batch_size, int) or batch_size < 1:
            raise MigrationError("migration batchSize must be a positive integer")
        path = self._path(source, target)
        all_nodes = sorted(graph._nodes.values(), key=lambda node: node["id"])
        all_edges = sorted((edge for edges in graph._edges.values() for edge in edges.values()), key=lambda edge: edge["id"])
        resume = options.get("resumeAfter") or {}
        nodes = [node for node in all_nodes if not resume.get("nodeId") or node["id"] > resume["nodeId"]]
        edges = [edge for edge in all_edges if not resume.get("edgeId") or edge["id"] > resume["edgeId"]]
        report = {"from": source, "to": target, "processed": 0, "total": len(nodes) + len(edges), "migrated": 0, "dryRun": options.get("dryRun", False)}
        migrate_node = lambda node: self._apply_node(path, node)
        migrate_edge = lambda edge: self._apply_edge(path, edge)
        for start in range(0, len(nodes), batch_size):
            batch = [migrate_node(copy.deepcopy(node)) for node in nodes[start:start + batch_size]]
            if not report["dryRun"] and path:
                with graph.transaction() as tx:
                    for node in batch:
                        tx.add_node(node)
            report["processed"] += len(batch)
            report["migrated"] += len(batch) if path else 0
            report["lastProcessed"] = {"kind": "node", "id": batch[-1]["id"]}
            callback = options.get("onProgress")
            if callback:
                callback(dict(report))
        edge_migrations = any(callable(definition.get("migrateEdge")) for definition in path)
        for start in range(0, len(edges), batch_size):
            original = edges[start:start + batch_size]
            batch = [migrate_edge(copy.deepcopy(edge)) for edge in original]
            changed = [edge for before, edge in zip(original, batch) if edge.get("data") != before.get("data")]
            if not report["dryRun"] and changed:
                with graph.transaction() as tx:
                    for edge in changed:
                        tx.update_edge(edge["id"], edge.get("data") or {})
            report["processed"] += len(batch)
            report["migrated"] += len(batch) if edge_migrations else 0
            report["lastProcessed"] = {"kind": "edge", "id": batch[-1]["id"]}
            callback = options.get("onProgress")
            if callback:
                callback(dict(report))
        return report

    @staticmethod
    def _apply_node(path: list[dict], node: Node) -> Node:
        for definition in path:
            result = definition["migrateNode"](_copy_node(node))
            if result is not None:
                node = result
        return node

    @staticmethod
    def _apply_edge(path: list[dict], edge: dict) -> dict:
        for definition in path:
            callback = definition.get("migrateEdge")
            if callback:
                result = callback(copy.deepcopy(edge))
                if result is not None:
                    edge = result
        return edge


class UniqueConstraintError(PolypackError):
    """Raised when a unique secondary index would contain duplicate data."""

__all__ = [
    "PolyGraph",
    "GraphTransaction",
    "GraphQuery",
    "ExactIndex",
    "HnswIndex",
    "ActivationEngine",
    "ChangeBatch",
    "merge_activation",
    "decay_factor",
    "engine_info",
    "DirectoryStorage",
    "PolypackError",
    "PolypackValueError",
    "PolypackDimensionError",
    "PolypackClosedError",
    "PolypackVersionError",
    "PolypackCorruptDataError",
    "PolypackStorageError",
    "ConflictError",
    "ResourceLimitError",
    "MigrationError",
    "MigrationRegistry",
    "UniqueConstraintError",
]

Node = dict
Edge = dict
Ownership = str

_OWNERSHIP_KEY = "__ownership"


def engine_info() -> dict:
    graph, vector, storage = _engine_info()
    return {"graph": graph, "vector": vector, "storage": storage}


# ── Validation helpers ──


def _as_list(vector: Any) -> list:
    if isinstance(vector, np.ndarray):
        return vector.astype(float).tolist()
    return list(vector)


def _validate_vector(vector: Any, name: str = "vector") -> list:
    values = _as_list(vector)
    if not values:
        raise PolypackValueError(f"{name} must not be empty")
    for v in values:
        if not isinstance(v, (int, float)) or not math.isfinite(float(v)):
            raise PolypackValueError(f"{name} must contain finite numbers")
    return [float(v) for v in values]


def _validate_id(id_: str, name: str = "id") -> None:
    if not isinstance(id_, str) or not id_:
        raise PolypackValueError(f"{name} must not be empty")


def _validate_timestamp(ts: Any) -> None:
    if not isinstance(ts, (int, float)) or not math.isfinite(float(ts)) or ts < 0:
        raise PolypackValueError("timestamps must be finite non-negative numbers")


def _patch_parts(path: str) -> list[str]:
    if not isinstance(path, str) or not path or any(not part for part in path.split(".")):
        raise PolypackValueError("patch paths must not be empty")
    return path.split(".")


def _patch_set(root: dict, path: str, value: Any) -> None:
    parts = _patch_parts(path)
    current = root
    for part in parts[:-1]:
        if not isinstance(current.get(part), dict):
            current[part] = {}
        current = current[part]
    current[parts[-1]] = copy.deepcopy(value)


def _patch_unset(root: dict, path: str) -> None:
    parts = _patch_parts(path)
    current: Any = root
    for part in parts[:-1]:
        if not isinstance(current, dict) or part not in current:
            return
        current = current[part]
    if isinstance(current, dict):
        current.pop(parts[-1], None)


def _patch_get(root: dict, path: str) -> Any:
    current: Any = root
    for part in _patch_parts(path):
        if not isinstance(current, dict) or part not in current:
            return None
        current = current[part]
    return current


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    if len(a) != len(b):
        raise PolypackDimensionError(f"expected {len(a)} dimensions, got {len(b)}")
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a))
    nb = math.sqrt(sum(x * x for x in b))
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


# ── Activation (mirrors the TypeScript / Rust activation model) ──

_HOUR = 3_600_000
_DAY = 24 * _HOUR

DEFAULT_ACTIVATION = {
    "scoreHalfLifeMs": _DAY,
    "importanceHalfLifeMs": 30 * _DAY,
    "importanceGain": 0.05,
}


def _clamp01(value: float) -> float:
    if value < 0.0:
        return 0.0
    if value > 1.0:
        return 1.0
    return value


def decay_factor(elapsed_ms: float, half_life_ms: float) -> float:
    """Exponential-decay multiplier `0.5 ** (elapsed / halfLife)`. Returns 1 for
    non-positive elapsed times and a non-decaying half-life."""
    if not math.isfinite(elapsed_ms) or elapsed_ms <= 0:
        return 1.0
    if not math.isfinite(half_life_ms) or half_life_ms <= 0:
        return 1.0
    factor = 0.5 ** (elapsed_ms / half_life_ms)
    return factor if factor < 1.0 else 1.0


def _activation_score_of(node: Node, now: Optional[int] = None, half_life_ms: Optional[float] = None) -> float:
    activation = node.get("activation")
    if not activation:
        return 0.0
    now = now if now is not None else int(time.time() * 1000)
    half_life_ms = half_life_ms if half_life_ms is not None else DEFAULT_ACTIVATION["scoreHalfLifeMs"]
    return _clamp01(activation["score"] * decay_factor(now - activation["lastMeaningfulActivation"], half_life_ms))


def _decay_activation_state(
    activation: dict,
    now: int,
    score_half_life_ms: float = DEFAULT_ACTIVATION["scoreHalfLifeMs"],
    importance_half_life_ms: float = DEFAULT_ACTIVATION["importanceHalfLifeMs"],
) -> dict:
    return {
        "score": _clamp01(activation["score"] * decay_factor(now - activation["lastMeaningfulActivation"], score_half_life_ms)),
        "importance": _clamp01(
            activation["importance"] * decay_factor(now - activation["lastMeaningfulActivation"], importance_half_life_ms)
        ),
        "reinforcementCount": activation["reinforcementCount"],
        "lastMeaningfulActivation": activation["lastMeaningfulActivation"],
    }


def _reinforce_activation(previous: Optional[dict], delta: float, now: int) -> dict:
    if previous:
        decayed = _decay_activation_state(previous, now)
        score, importance = decayed["score"], decayed["importance"]
        count = previous.get("reinforcementCount", 0) + 1
    else:
        score, importance, count = 0.0, 0.0, 1
    return {
        "score": _clamp01(score + delta),
        "importance": _clamp01(importance + DEFAULT_ACTIVATION["importanceGain"] * delta),
        "reinforcementCount": count,
        "lastMeaningfulActivation": now,
    }


def merge_activation(existing: dict, incoming: dict, now: Optional[int] = None) -> dict:
    """Merge two durable activation records (total-state max-merge). Decay-corrects
    both to `now`, keeps the stronger component of each, and re-anchors to `now`.
    Concurrent deltas accumulate additively instead — activation is accumulated
    knowledge, not last-write-wins data."""
    now = now if now is not None else int(time.time() * 1000)
    ex = _decay_activation_state(existing, now)
    inc = _decay_activation_state(incoming, now)
    return {
        "score": max(ex["score"], inc["score"]),
        "importance": max(ex["importance"], inc["importance"]),
        "reinforcementCount": max(existing["reinforcementCount"], incoming["reinforcementCount"]),
        "lastMeaningfulActivation": now,
    }


def _validate_activation(activation: Any) -> dict:
    if not isinstance(activation, dict):
        raise PolypackValueError("activation must be an object")
    for key in ("score", "importance"):
        value = activation.get(key)
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)) or not 0 <= float(value) <= 1:
            raise PolypackValueError(f"activation.{key} must be a finite number in [0, 1]")
    count = activation.get("reinforcementCount", 0)
    if not isinstance(count, int) or count < 0:
        raise PolypackValueError("activation.reinforcementCount must be a non-negative integer")
    anchor = activation.get("lastMeaningfulActivation", 0)
    if not isinstance(anchor, (int, float)) or not math.isfinite(float(anchor)) or anchor < 0:
        raise PolypackValueError("activation.lastMeaningfulActivation must be a finite non-negative number")
    return {
        "score": float(activation["score"]),
        "importance": float(activation["importance"]),
        "reinforcementCount": int(count),
        "lastMeaningfulActivation": int(anchor),
    }


# ── Vector index wrappers ──


class ExactIndex:
    """Exact vector index backed by the Rust core.

    Mirrors the TypeScript `VectorIndex`: detached reads, validation, and
    `on_change` notifications on mutations.
    """

    def __init__(self, distance: str = "cosine", on_change: Optional[Callable[[str], None]] = None) -> None:
        self._inner = _NativeExactIndex(distance)
        self._on_change = on_change

    def add(self, id_: str, vector: Any) -> None:
        _validate_id(id_, "vector id")
        values = _validate_vector(vector)
        self._inner.add(id_, values)
        if self._on_change:
            self._on_change(id_)

    def add_many(self, entries: Iterable[tuple]) -> None:
        ids = []
        vectors = []
        for id_, vector in entries:
            _validate_id(id_, "vector id")
            ids.append(id_)
            vectors.append(_validate_vector(vector))
        self._inner.add_many(ids, vectors)
        if self._on_change:
            for id_ in ids:
                self._on_change(id_)

    def hydrate(self, id_: str, vector: Any) -> None:
        _validate_id(id_, "vector id")
        self._inner.add(id_, _validate_vector(vector))

    def remove(self, id_: str) -> None:
        self._inner.remove(id_)

    def remove_many(self, ids: Iterable[str]) -> None:
        self._inner.remove_many(list(ids))

    def query(self, vector: Any, top_k: int, threshold: float = 0.0) -> list:
        values = _validate_vector(vector, "query vector")
        if not isinstance(top_k, int) or top_k < 0:
            raise PolypackValueError("topK must be a non-negative integer")
        if not math.isfinite(float(threshold)):
            raise PolypackValueError("threshold must be finite")
        if top_k == 0:
            return []
        return self._inner.query(values, top_k, float(threshold))

    def clear(self) -> None:
        self._inner.clear()

    def has(self, id_: str) -> bool:
        return self._inner.has(id_)

    def get(self, id_: str) -> Optional[list]:
        arr = self._inner.get(id_)
        return None if arr is None else arr.tolist()

    def entries(self) -> Iterator[tuple]:
        for id_, arr in self._inner.entries():
            yield id_, arr.tolist()

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def size(self) -> int:
        return len(self._inner)


class HnswIndex:
    """Approximate vector index backed by the Rust update-safe HNSW."""

    def __init__(
        self,
        on_change: Optional[Callable[[str], None]] = None,
        m: int = 16,
        mmax0: int = 32,
        ef_construction: int = 200,
        ef_search: int = 200,
        level_seed: int = 7,
    ) -> None:
        self._inner = _NativeHnswIndex(m, mmax0, ef_construction, ef_search, level_seed)
        self._on_change = on_change

    def add(self, id_: str, vector: Any) -> None:
        _validate_id(id_, "vector id")
        self._inner.add(id_, _validate_vector(vector))
        if self._on_change:
            self._on_change(id_)

    def update(self, id_: str, vector: Any) -> None:
        _validate_id(id_, "vector id")
        self._inner.update(id_, _validate_vector(vector))
        if self._on_change:
            self._on_change(id_)

    def add_many(self, entries: Iterable[tuple]) -> None:
        ids = []
        vectors = []
        for id_, vector in entries:
            _validate_id(id_, "vector id")
            ids.append(id_)
            vectors.append(_validate_vector(vector))
        self._inner.add_many(ids, vectors)
        if self._on_change:
            for id_ in ids:
                self._on_change(id_)

    def remove(self, id_: str) -> None:
        self._inner.remove(id_)

    def remove_many(self, ids: Iterable[str]) -> None:
        self._inner.remove_many(list(ids))

    def query(self, vector: Any, top_k: int, threshold: float = 0.0) -> list:
        values = _validate_vector(vector, "query vector")
        if not isinstance(top_k, int) or top_k < 0:
            raise PolypackValueError("topK must be a non-negative integer")
        if not math.isfinite(float(threshold)):
            raise PolypackValueError("threshold must be finite")
        if top_k == 0:
            return []
        return self._inner.query(values, top_k, float(threshold))

    def clear(self) -> None:
        self._inner.clear()

    def has(self, id_: str) -> bool:
        return self._inner.has(id_)

    def get(self, id_: str) -> Optional[list]:
        arr = self._inner.get(id_)
        return None if arr is None else arr.tolist()

    def entries(self) -> Iterator[tuple]:
        for id_, arr in self._inner.entries():
            yield id_, arr.tolist()

    def __len__(self) -> int:
        return len(self._inner)

    @property
    def size(self) -> int:
        return len(self._inner)


# ── Graph ──


class DirectoryStorage:
    """Host byte-storage adapter over a directory (snapshot.msgpack / wal.msgpack)."""

    def __init__(self, directory: str, read_only: bool = False) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only
        self._lock_file = open(self._dir / "store.lock", "a+b")
        try:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_SH if read_only else fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            self._lock_file.close()
            self._lock_file = None
            raise PolypackStorageError(f"store is already locked: {directory}") from exc
        if not read_only:
            self._lock_file.seek(0)
            self._lock_file.truncate()
            self._lock_file.write(f"pid={os.getpid()}\n".encode())
            self._lock_file.flush()

    def close(self) -> None:
        if getattr(self, "_lock_file", None) is not None:
            fcntl.flock(self._lock_file.fileno(), fcntl.LOCK_UN)
            self._lock_file.close()
            self._lock_file = None

    def __del__(self) -> None:
        try:
            self.close()
        except (OSError, AttributeError):
            pass

    @property
    def capabilities(self) -> dict:
        return {
            "atomicBatches": True,
            "transactions": True,
            "fsync": True,
            "secondaryIndexes": True,
            "snapshots": True,
            "changeFeed": False,
            "concurrentWriters": False,
            "vectorSearch": "exact",
        }

    @property
    def read_only(self) -> bool:
        return self._read_only

    def _path(self, name: str) -> Path:
        return self._dir / name

    def read(self, name: str) -> Optional[bytes]:
        path = self._path(name)
        return path.read_bytes() if path.exists() else None

    def write(self, name: str, data: bytes) -> None:
        if self._read_only:
            raise PolypackStorageError("store was opened read-only")
        # Write-then-rename: a crash mid-write leaves the previous snapshot
        # intact instead of a torn file, and the tmp file is fsynced first
        # so the rename can never land ahead of its data on disk.
        target = self._path(name)
        tmp_path = self._dir / f"{name}.tmp"
        with open(tmp_path, "wb") as f:
            f.write(bytes(data))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, target)

    def append(self, name: str, data: bytes) -> None:
        if self._read_only:
            raise PolypackStorageError("store was opened read-only")
        with open(self._path(name), "ab") as f:
            f.write(bytes(data))

    def delete(self, name: str) -> None:
        if self._read_only:
            raise PolypackStorageError("store was opened read-only")
        try:
            self._path(name).unlink()
        except FileNotFoundError:
            pass

    def exists(self, name: str) -> bool:
        return self._path(name).exists()


def _copy_node(node: Node) -> Node:
    out = {
        "id": node["id"],
        "type": node["type"],
        "data": copy.deepcopy(node.get("data") or {}),
        "vector": None if node.get("vector") is None else list(node.get("vector")),
        "insertedAt": node["insertedAt"],
        "updatedAt": node["updatedAt"],
        "revision": int(node.get("revision", 0)),
    }
    if node.get("activation") is not None:
        out["activation"] = dict(node["activation"])
    return out


def _edge_key(source: str, edge_type: str, target: str) -> str:
    if "::" in source or "::" in edge_type:
        raise PolypackValueError('edge source and type must not contain "::"')
    return f"{source}::{edge_type}::{target}"


class GraphTransaction:
    """Checkpointed transaction for ``PolyGraph``.

    Mutations are applied to the graph immediately, providing read-your-own-
    writes. A failed callback or explicit rollback restores the graph and
    vector index. This is an in-process transaction boundary; callers should
    not concurrently mutate the same graph from another thread.
    """

    def __init__(self, graph: "PolyGraph", actor: Optional[str] = None, base_revision: Optional[int] = None, metadata: Optional[dict] = None, operation_id: Optional[str] = None) -> None:
        if graph._active_transaction is not None:
            raise PolypackValueError("nested transactions are not supported")
        self.graph = graph
        self.id = f"tx-{uuid.uuid4().hex}"
        if operation_id is not None and not operation_id:
            raise PolypackValueError("operation_id must not be empty")
        self.operation_id = operation_id or f"op-{uuid.uuid4().hex}"
        self.actor = actor
        self.base_revision = base_revision
        self.metadata = copy.deepcopy(metadata) if metadata is not None else None
        self._snapshot = {
            "nodes": copy.deepcopy(graph._nodes),
            "edges": copy.deepcopy(graph._edges),
            "incoming": copy.deepcopy(graph._incoming),
            "removed_nodes": set(graph._removed_node_ids),
            "removed_edges": set(graph._removed_edge_ids),
            "removed_vectors": set(graph._removed_vector_ids),
            "vectors": list(graph.vectors.entries()),
        }
        self._closed = False
        self._mutation_count = 0
        graph._active_transaction = self

    def _before_mutation(self) -> None:
        limit = self.graph._resource_limits.get("maxBatchSize")
        if limit is not None and self._mutation_count >= limit:
            raise ResourceLimitError("maxBatchSize", limit)
        self._mutation_count += 1

    def _restore(self) -> None:
        self.graph._nodes = self._snapshot["nodes"]
        self.graph._edges = self._snapshot["edges"]
        self.graph._incoming = self._snapshot["incoming"]
        self.graph._removed_node_ids = self._snapshot["removed_nodes"]
        self.graph._removed_edge_ids = self._snapshot["removed_edges"]
        self.graph._removed_vector_ids = self._snapshot["removed_vectors"]
        self.graph.vectors.clear()
        self.graph.vectors.add_many(self._snapshot["vectors"])
        self.graph._rebuild_secondary_indexes()

    def commit(self) -> None:
        if self._closed:
            return
        try:
            if self.graph._store is not None:
                self.graph.save()
        except Exception:
            self._restore()
            self.graph._active_transaction = None
            self._closed = True
            raise
        self.graph._active_transaction = None
        self._closed = True

    def rollback(self) -> None:
        if self._closed:
            return
        self._restore()
        self.graph._active_transaction = None
        self._closed = True

    def __enter__(self) -> "GraphTransaction":
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        if exc_type is None:
            self.commit()
        else:
            self.rollback()

    def add_node(self, node: Node) -> None:
        self._before_mutation()
        self.graph.add_node(node)

    def update_node(self, *args: Any, **kwargs: Any) -> Optional[Node]:
        self._before_mutation()
        return self.graph.update_node(*args, **kwargs)

    def patch_node(self, *args: Any, **kwargs: Any) -> Optional[Node]:
        self._before_mutation()
        return self.graph.patch_node(*args, **kwargs)

    def remove_node(self, *args: Any, **kwargs: Any) -> None:
        self._before_mutation()
        self.graph.remove_node(*args, **kwargs)

    def add_edge(self, *args: Any, **kwargs: Any) -> None:
        self._before_mutation()
        self.graph.add_edge(*args, **kwargs)

    def update_edge(self, *args: Any, **kwargs: Any) -> Optional[dict]:
        self._before_mutation()
        return self.graph.update_edge(*args, **kwargs)

    def remove_edge(self, *args: Any, **kwargs: Any) -> bool:
        self._before_mutation()
        return self.graph.remove_edge(*args, **kwargs)

    def get_node(self, id_: str) -> Optional[Node]:
        return self.graph.get_node(id_)

    def get_edges(self, source: str, edge_type: Optional[str] = None) -> list:
        return self.graph.get_edges(source, edge_type)


class PolyGraph:
    """In-memory property graph with vector search and ownership semantics."""

    def __init__(
        self,
        on_orphan: Optional[Callable[[str], None]] = None,
        vector_index: Optional[ExactIndex] = None,
    ) -> None:
        self._nodes: dict[str, Node] = {}
        self._edges: dict[str, dict] = {}
        self._incoming: dict[str, dict] = {}
        self.vectors = vector_index or ExactIndex()
        self._on_orphan = on_orphan
        self._store: Optional[_NativeStore] = None
        self._store_directory: Optional[Path] = None
        self._store_read_only = False
        self._removed_node_ids: set = set()
        self._removed_edge_ids: set = set()
        self._removed_vector_ids: set = set()
        self._active_transaction: Optional["GraphTransaction"] = None
        self._node_type_definitions: dict[str, dict] = {}
        self._edge_type_definitions: dict[str, dict] = {}
        self._resource_limits: dict[str, int] = {}
        self.migrations = MigrationRegistry()
        self._indexes: dict[str, dict] = {}
        self._secondary_index_data: dict[str, dict[str, set[str]]] = {}
        self._query_count = 0
        self._query_duration_ms = 0.0
        self._query_scanned_records = 0
        self._query_index_usage: dict[str, int] = {}

    @property
    def capabilities(self) -> dict:
        if self._store is not None:
            return dict(self._store.capabilities())
        return {
            "atomicBatches": True,
            "transactions": True,
            "fsync": False,
            "secondaryIndexes": True,
            "snapshots": True,
            "changeFeed": False,
            "concurrentWriters": False,
            "vectorSearch": "exact",
        }

    def require_adapter_capabilities(self, required: Optional[dict] = None, **kwargs: Any) -> None:
        """Reject the graph unless its adapter declares the requested guarantees."""
        expectations = dict(required or {})
        expectations.update(kwargs)
        capabilities = self.capabilities
        for name, expected in expectations.items():
            if capabilities.get(name) != expected:
                raise PolypackValueError(f"persistence adapter does not support capability: {name}")

    def set_resource_limits(self, limits: Optional[dict] = None, **kwargs: int) -> None:
        """Configure positive integer limits for writes and graph queries."""
        values = dict(limits or {})
        values.update(kwargs)
        allowed = {"maxVectorDimensions", "maxNodePayloadBytes", "maxBatchSize", "maxTraversalDepth", "maxNodesVisited", "maxResults"}
        unknown = set(values) - allowed
        if unknown:
            raise PolypackValueError(f"unknown resource limit: {sorted(unknown)[0]}")
        for name, value in values.items():
            if value is not None and (not isinstance(value, int) or isinstance(value, bool) or value < 1):
                raise PolypackValueError(f"{name} must be a positive integer")
        self._resource_limits = {name: value for name, value in values.items() if value is not None}

    @property
    def resource_limit_config(self) -> dict:
        return dict(self._resource_limits)

    def register_node_type(
        self,
        node_type: str,
        validate: Optional[Callable[[Node], Any]] = None,
        required_fields: Optional[Iterable[str]] = None,
        data_types: Optional[dict[str, str]] = None,
    ) -> None:
        _validate_id(node_type, "node type")
        previous = self._node_type_definitions.get(node_type)
        self._node_type_definitions[node_type] = {
            "validate": validate,
            "requiredFields": tuple(required_fields or ()),
            "dataTypes": dict(data_types or {}),
        }
        try:
            for node in self._nodes.values():
                if node["type"] == node_type:
                    self._validate_node_schema(node)
        except Exception:
            if previous is None:
                self._node_type_definitions.pop(node_type, None)
            else:
                self._node_type_definitions[node_type] = previous
            raise

    def define_index(
        self,
        name: str,
        fields: Optional[Iterable[str]] = None,
        node_type: Optional[str] = None,
        unique: bool = False,
        sparse: bool = False,
    ) -> None:
        """Define a single or compound node-data index.

        The Python engine keeps metadata and maintained candidate buckets,
        while every indexed result still passes the complete query predicate.
        """
        if isinstance(name, dict):
            definition = name
            name = definition.get("name", "")
            fields = definition.get("fields")
            node_type = definition.get("nodeType")
            unique = bool(definition.get("unique", False))
            sparse = bool(definition.get("sparse", False))
        if self._store_read_only:
            raise PolypackStorageError("store was opened read-only")
        field_list = list(fields or ())
        if not isinstance(name, str) or not name or not field_list or any(not isinstance(field, str) or not field for field in field_list) or len(set(field_list)) != len(field_list):
            raise PolypackValueError("index name and unique fields must not be empty")
        previous = self._indexes.get(name)
        self._indexes[name] = {"name": name, "fields": field_list, "nodeType": node_type, "unique": unique, "sparse": sparse}
        self._secondary_index_data[name] = {}
        try:
            self._validate_all_indexes()
            self._rebuild_secondary_indexes()
            self._persist_index_metadata()
        except Exception:
            if previous is None:
                self._indexes.pop(name, None)
            else:
                self._indexes[name] = previous
            self._rebuild_secondary_indexes()
            raise

    def drop_index(self, name: str) -> bool:
        if self._store_read_only:
            raise PolypackStorageError("store was opened read-only")
        previous = self._indexes.pop(name, None)
        if previous is None:
            return False
        self._secondary_index_data.pop(name, None)
        try:
            self._persist_index_metadata()
        except Exception:
            self._indexes[name] = previous
            self._rebuild_secondary_indexes()
            raise
        return True

    @property
    def indexes(self) -> list[dict]:
        return [dict(definition, fields=list(definition["fields"])) for definition in self._indexes.values()]

    def _index_key(self, node: Node, definition: dict) -> Optional[str]:
        if definition["nodeType"] and node["type"] != definition["nodeType"]:
            return None
        values = [_patch_get(node.get("data") or {}, field) for field in definition["fields"]]
        if definition["sparse"] and any(value is None for value in values):
            return None
        return repr(values)

    def _add_secondary_index_entry(self, node: Node) -> None:
        for name, definition in self._indexes.items():
            key = self._index_key(node, definition)
            if key is not None:
                self._secondary_index_data.setdefault(name, {}).setdefault(key, set()).add(node["id"])

    def _remove_secondary_index_entry(self, node: Node) -> None:
        for name, definition in self._indexes.items():
            key = self._index_key(node, definition)
            if key is None:
                continue
            bucket = self._secondary_index_data.get(name, {}).get(key)
            if bucket is not None:
                bucket.discard(node["id"])
                if not bucket:
                    self._secondary_index_data[name].pop(key, None)

    def _rebuild_secondary_indexes(self) -> None:
        self._secondary_index_data = {name: {} for name in self._indexes}
        for node in self._nodes.values():
            self._add_secondary_index_entry(node)

    def _validate_all_indexes(self) -> None:
        for definition in self._indexes.values():
            if not definition["unique"]:
                continue
            seen: dict[str, str] = {}
            for node in self._nodes.values():
                key = self._index_key(node, definition)
                if key is None:
                    continue
                previous = seen.get(key)
                if previous is not None and previous != node["id"]:
                    raise UniqueConstraintError(f"unique index {definition['name']} conflicts with node {previous}")
                seen[key] = node["id"]

    def _record_query(self, duration_ms: float, scanned_records: int, index: Optional[str]) -> None:
        self._query_count += 1
        self._query_duration_ms += duration_ms
        self._query_scanned_records += scanned_records
        if index is not None:
            self._query_index_usage[index] = self._query_index_usage.get(index, 0) + 1

    def _validate_index_candidate(self, candidate: Node) -> None:
        for definition in self._indexes.values():
            if not definition["unique"]:
                continue
            key = self._index_key(candidate, definition)
            if key is None:
                continue
            for existing_id in self._secondary_index_data.get(definition["name"], {}).get(key, set()):
                if existing_id != candidate["id"]:
                    raise UniqueConstraintError(f"unique index {definition['name']} conflicts with node {existing_id}")
            for node in self._nodes.values():
                if node["id"] == candidate["id"]:
                    continue
                if self._index_key(node, definition) == key:
                    raise UniqueConstraintError(f"unique index {definition['name']} conflicts with node {node['id']}")

    def _persist_index_metadata(self) -> None:
        if self._store_directory is None:
            return
        if self._store_read_only:
            raise PolypackStorageError("store was opened read-only")
        target = self._store_directory / "indexes.json"
        temporary = self._store_directory / "indexes.json.tmp"
        temporary.write_text(json.dumps(self.indexes, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)

    def _load_index_metadata(self) -> None:
        if self._store_directory is None:
            return
        source = self._store_directory / "indexes.json"
        if not source.exists():
            return
        try:
            definitions = json.loads(source.read_text(encoding="utf-8"))
            self._indexes = {}
            if not isinstance(definitions, list):
                raise ValueError("index metadata must be an array")
            seen_names: set[str] = set()
            for definition in definitions:
                if not isinstance(definition, dict):
                    raise ValueError("index definition must be an object")
                name = definition.get("name")
                fields = definition.get("fields")
                if not isinstance(name, str) or not name or name in seen_names:
                    raise ValueError("index names must be unique and non-empty")
                if not isinstance(fields, list) or not fields or any(not isinstance(field, str) or not field for field in fields) or len(set(fields)) != len(fields):
                    raise ValueError("invalid index definition")
                seen_names.add(name)
                self._indexes[name] = {
                    "name": name,
                    "fields": fields,
                    "nodeType": definition.get("nodeType"),
                    "unique": bool(definition.get("unique", False)),
                    "sparse": bool(definition.get("sparse", False)),
                }
            self._validate_all_indexes()
            self._rebuild_secondary_indexes()
        except (OSError, ValueError, TypeError, PolypackError) as exc:
            raise PolypackStorageError(f"invalid index metadata: {exc}") from exc

    def register_edge_type(
        self,
        edge_type: str,
        source_types: Optional[Iterable[str]] = None,
        target_types: Optional[Iterable[str]] = None,
        validate: Optional[Callable[[dict], Any]] = None,
        cardinality: Optional[str] = None,
        required_fields: Optional[Iterable[str]] = None,
        data_types: Optional[dict[str, str]] = None,
    ) -> None:
        _validate_id(edge_type, "edge type")
        if cardinality not in (None, "one-to-one", "one-to-many", "many-to-one", "many-to-many"):
            raise PolypackValueError("invalid edge cardinality")
        previous = self._edge_type_definitions.get(edge_type)
        self._edge_type_definitions[edge_type] = {
            "sourceTypes": frozenset(source_types or ()),
            "targetTypes": frozenset(target_types or ()),
            "validate": validate,
            "cardinality": cardinality,
            "requiredFields": tuple(required_fields or ()),
            "dataTypes": dict(data_types or {}),
        }
        try:
            for edges in self._edges.values():
                for edge in edges.values():
                    if edge["type"] == edge_type:
                        self._validate_edge_schema(edge)
        except Exception:
            if previous is None:
                self._edge_type_definitions.pop(edge_type, None)
            else:
                self._edge_type_definitions[edge_type] = previous
            raise

    def _validate_node_schema(self, node: Node) -> None:
        definition = self._node_type_definitions.get(node["type"])
        if not definition:
            return
        for field_name in definition["requiredFields"]:
            if _patch_get(node.get("data") or {}, field_name) is None:
                raise PolypackValueError(f"node {node['id']} is missing required field {field_name}")
        for field_name, expected in definition["dataTypes"].items():
            value = _patch_get(node.get("data") or {}, field_name)
            if value is None:
                continue
            valid = {
                "string": isinstance(value, str),
                "number": isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
                "integer": isinstance(value, int) and not isinstance(value, bool),
                "boolean": isinstance(value, bool),
                "array": isinstance(value, list),
                "object": isinstance(value, dict),
            }.get(expected)
            if valid is not True:
                raise PolypackValueError(f"node {node['id']} field {field_name} must be {expected}")
        validator = definition["validate"]
        if validator is not None and validator(_copy_node(node)) is False:
            raise PolypackValueError(f"node validator rejected {node['id']}")

    def _validate_edge_schema(self, edge: dict) -> None:
        definition = self._edge_type_definitions.get(edge["type"])
        if not definition:
            return
        source = self._nodes.get(edge["source"])
        target = self._nodes.get(edge["target"])
        if source is None or target is None:
            raise PolypackValueError(f"edge {edge['id']} references a missing endpoint")
        if definition["sourceTypes"] and (source is None or source["type"] not in definition["sourceTypes"]):
            raise PolypackValueError(f"edge source type is not permitted for {edge['type']}")
        if definition["targetTypes"] and (target is None or target["type"] not in definition["targetTypes"]):
            raise PolypackValueError(f"edge target type is not permitted for {edge['type']}")
        for field_name in definition["requiredFields"]:
            if _patch_get(edge.get("data") or {}, field_name) is None:
                raise PolypackValueError(f"edge {edge['id']} is missing required field {field_name}")
        for field_name, expected in definition["dataTypes"].items():
            value = _patch_get(edge.get("data") or {}, field_name)
            if value is None:
                continue
            valid = {
                "string": isinstance(value, str),
                "number": isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value)),
                "integer": isinstance(value, int) and not isinstance(value, bool),
                "boolean": isinstance(value, bool),
                "array": isinstance(value, list),
                "object": isinstance(value, dict),
            }.get(expected)
            if valid is not True:
                raise PolypackValueError(f"edge {edge['id']} field {field_name} must be {expected}")
        cardinality = definition["cardinality"]
        if cardinality and cardinality != "many-to-many":
            outgoing = 0
            incoming = 0
            for edges in self._edges.values():
                for candidate in edges.values():
                    if candidate["id"] == edge["id"] or candidate["type"] != edge["type"]:
                        continue
                    outgoing += candidate["source"] == edge["source"]
                    incoming += candidate["target"] == edge["target"]
            violates = (
                cardinality == "one-to-one" and (outgoing > 0 or incoming > 0)
            ) or (
                cardinality == "one-to-many" and incoming > 0
            ) or (
                cardinality == "many-to-one" and outgoing > 0
            )
            if violates:
                raise PolypackValueError(f"edge cardinality {cardinality} would be exceeded")
        validator = definition["validate"]
        if validator is not None and validator(dict(edge, data=dict(edge.get("data") or {}))) is False:
            raise PolypackValueError(f"edge validator rejected {edge['id']}")

    # ── context manager ──

    def transaction(
        self,
        callback: Optional[Callable[["GraphTransaction"], Any]] = None,
        actor: Optional[str] = None,
        base_revision: Optional[int] = None,
        metadata: Optional[dict] = None,
        operation_id: Optional[str] = None,
    ) -> "GraphTransaction":
        """Create a checkpointed transaction, or execute a callback in one."""
        tx = GraphTransaction(self, actor=actor, base_revision=base_revision, metadata=metadata, operation_id=operation_id)
        if callback is not None:
            with tx:
                return callback(tx)
        return tx

    def __enter__(self) -> "PolyGraph":
        return self

    def __exit__(self, *exc) -> None:
        if self._store is not None:
            self.close_store()
        else:
            self.clear()

    # ── node CRUD ──

    def add_node(self, node: Node) -> None:
        """Insert or replace a node. Replacement updates the vector index; data and vector are copied on entry."""
        _validate_id(node.get("id"), "node id")
        _validate_id(node.get("type"), "node type")
        _validate_timestamp(node.get("insertedAt"))
        _validate_timestamp(node.get("updatedAt"))
        stored = {
            "id": node["id"],
            "type": node["type"],
            "data": dict(node.get("data") or {}),
            "insertedAt": node["insertedAt"],
            "updatedAt": node["updatedAt"],
            "revision": int(node.get("revision", 0)),
        }
        previous = self._nodes.get(stored["id"])
        if previous is not None:
            stored["revision"] = int(previous.get("revision", 0)) + 1
        if node.get("activation") is not None:
            stored["activation"] = _validate_activation(node["activation"])
        if node.get("vector") is not None:
            stored["vector"] = _validate_vector(node["vector"])
        else:
            stored["vector"] = None
        self._validate_node_schema(stored)
        self._validate_index_candidate(stored)
        if previous is not None:
            self._remove_secondary_index_entry(previous)
        if stored["vector"] is None:
            self.vectors.remove(stored["id"])
        self._nodes[stored["id"]] = stored
        self._removed_node_ids.discard(stored["id"])
        if stored["vector"] is not None:
            self.vectors.add(stored["id"], stored["vector"])
            self._removed_vector_ids.discard(stored["id"])
        else:
            self._removed_vector_ids.add(stored["id"])
        self._add_secondary_index_entry(stored)

    def update_node(
        self,
        id_: str,
        data: dict,
        vector: Any = None,
        activation: Optional[dict] = None,
        expected_revision: Optional[int] = None,
    ) -> Optional[Node]:
        """Shallow-merge `data` into a loaded node and optionally replace its vector or durable activation. Returns `None` if the node isn't loaded."""
        node = self._nodes.get(id_)
        if node is None:
            return None
        if expected_revision is not None and int(node.get("revision", 0)) != expected_revision:
            raise ConflictError(
                f"record {id_} has revision {node.get('revision', 0)}, expected {expected_revision}"
            )
        candidate = _copy_node(node)
        candidate["data"].update(data or {})
        if vector is not None:
            candidate["vector"] = _validate_vector(vector)
        if activation is not None:
            candidate["activation"] = _validate_activation(activation)
        candidate["updatedAt"] = int(time.time() * 1000)
        candidate["revision"] = int(node.get("revision", 0)) + 1
        self._validate_node_schema(candidate)
        self._validate_index_candidate(candidate)
        self._remove_secondary_index_entry(node)
        node.clear()
        node.update(candidate)
        self._add_secondary_index_entry(node)
        if vector is not None:
            self.vectors.add(id_, node["vector"])
            self._removed_vector_ids.discard(id_)
        return _copy_node(node)

    def get_node(self, id_: str) -> Optional[Node]:
        node = self._nodes.get(id_)
        return None if node is None else _copy_node(node)

    def stats(self) -> dict:
        """Return operational graph counters without a monitoring dependency."""
        if self._store is None:
            persisted_node_count = len(self._nodes)
            edge_count = sum(len(edges) for edges in self._edges.values())
            vector_count = self.vectors.size
        else:
            persisted_node_count = self._store.node_count()
            edge_count = len(self._store.all_edges())
            vector_count = len(self._store.all_vectors())
        memory_estimate = len(repr(self._nodes).encode("utf-8")) + len(repr(self._edges).encode("utf-8"))
        return {
            "loadedNodeCount": len(self._nodes),
            "persistedNodeCount": persisted_node_count,
            "edgeCount": edge_count,
            "vectorCount": vector_count,
            "dirtyRecordCount": len(self._removed_node_ids) + len(self._removed_edge_ids) + len(self._removed_vector_ids),
            "pendingPersistence": bool(self._removed_node_ids or self._removed_edge_ids or self._removed_vector_ids),
            "indexCount": len(self._indexes),
            "memoryEstimateBytes": memory_estimate,
            "queryCount": self._query_count,
            "queryDurationMs": self._query_duration_ms,
            "queryScannedRecords": self._query_scanned_records,
            "queryIndexUsage": dict(self._query_index_usage),
        }

    def verify(self) -> dict:
        """Verify persisted storage or the currently loaded in-memory graph."""
        if self._store is not None:
            report = dict(self._store.verify())
            errors = list(report.get("errors", []))
            try:
                self._validate_all_indexes()
            except UniqueConstraintError as exc:
                errors.append(str(exc))
            report["errors"] = errors
            report["ok"] = not errors
            return report

        errors = []
        node_ids = set(self._nodes)
        edge_count = 0
        for source, edges in self._edges.items():
            for edge in edges.values():
                edge_count += 1
                if source not in node_ids:
                    errors.append(f"edge {edge.get('id')} has missing source {source}")
                if edge.get("target") not in node_ids:
                    errors.append(f"edge {edge.get('id')} has missing target {edge.get('target')}")
        for node_id, vector in self.vectors.entries():
            if node_id not in node_ids:
                errors.append(f"vector {node_id} has no node")
            if any(not math.isfinite(float(value)) for value in vector):
                errors.append(f"vector {node_id} contains a non-finite value")
        try:
            self._validate_all_indexes()
        except UniqueConstraintError as exc:
            errors.append(str(exc))
        return {
            "ok": not errors,
            "errors": errors,
            "nodeCount": len(node_ids),
            "edgeCount": edge_count,
            "vectorCount": self.vectors.size,
            "mutationCount": 0,
        }

    def mutation_log(self) -> list[dict]:
        """Return durable logical mutations from the attached store."""
        if self._store is None:
            return []
        return list(self._store.mutation_log())

    def mutation_log_since(self, sequence: int = 0) -> list[dict]:
        """Return durable logical mutations after a sequence cursor."""
        if self._store is None:
            return []
        return list(self._store.mutation_log_since(sequence))

    def mutation_log_page(self, sequence: int = 0, limit: int = 1000) -> list[dict]:
        """Return a bounded mutation-log page after a sequence cursor."""
        if self._store is None:
            return []
        if not isinstance(limit, int) or limit < 0:
            raise PolypackValueError("mutation log limit must be a non-negative integer")
        return list(self._store.mutation_log_page(sequence, limit))

    def latest_mutation_sequence(self) -> int:
        """Return the latest acknowledged durable mutation cursor."""
        if self._store is None:
            return 0
        return int(self._store.latest_mutation_sequence())

    def patch_node(
        self,
        id_: str,
        set: Optional[dict] = None,
        unset: Optional[Iterable[str]] = None,
        increment: Optional[dict] = None,
        expected_revision: Optional[int] = None,
    ) -> Optional[Node]:
        """Apply dotted-path set/unset/increment operations atomically to a node."""
        node = self._nodes.get(id_)
        if node is None:
            return None
        actual = int(node.get("revision", 0))
        if expected_revision is not None and actual != expected_revision:
            raise ConflictError(f"record {id_} has revision {actual}, expected {expected_revision}")

        candidate = copy.deepcopy(node.get("data") or {})
        for path, value in (set or {}).items():
            _patch_set(candidate, path, value)
        for path in unset or ():
            _patch_unset(candidate, path)
        for path, delta in (increment or {}).items():
            if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
                raise PolypackValueError("increment values must be finite numbers")
            current = _patch_get(candidate, path)
            if current is None:
                current = 0
            if not isinstance(current, (int, float)) or isinstance(current, bool) or not math.isfinite(float(current)):
                raise PolypackValueError("increment targets must be numeric")
            _patch_set(candidate, path, float(current) + float(delta))

        candidate_node = _copy_node(node)
        candidate_node["data"] = candidate
        candidate_node["updatedAt"] = int(time.time() * 1000)
        candidate_node["revision"] = actual + 1
        self._validate_node_schema(candidate_node)
        node.clear()
        node.update(candidate_node)
        return _copy_node(node)

    def get_nodes(self, ids: Iterable[str]) -> list:
        return [self.get_node(i) for i in ids if self.get_node(i) is not None]

    def remove_node(
        self,
        id_: str,
        _visited: Optional[set] = None,
        expected_revision: Optional[int] = None,
    ) -> None:
        """Remove `id_` and cascade through 'owned' edges. A target of an 'owned' edge is also
        removed unless another 'owned' source keeps it alive. Cyclic owned edges are detected
        so each node is only removed once; `_visited` is an internal recursion argument."""
        visited = _visited if _visited is not None else set()
        if id_ in visited:
            return
        visited.add(id_)
        node = self._nodes.get(id_)
        if node is None:
            return
        if expected_revision is not None and int(node.get("revision", 0)) != expected_revision:
            raise ConflictError(
                f"record {id_} has revision {node.get('revision', 0)}, expected {expected_revision}"
            )
        for edge in list(self._edges.get(id_, {}).values()):
            if self._ownership(edge) == "owned" and not self._has_other_owned_source(edge["target"], id_):
                self.remove_node(edge["target"], visited)
        self._cleanup_edges(id_)
        self._remove_secondary_index_entry(node)
        del self._nodes[id_]
        self.vectors.remove(id_)
        self._removed_node_ids.add(id_)
        self._removed_vector_ids.add(id_)

    # ── edge CRUD ──

    def add_edge(
        self,
        source: str,
        edge_type: str,
        target: str,
        data: Optional[dict] = None,
        ownership: Optional[Ownership] = None,
        created_at: Optional[int] = None,
        revision: int = 0,
        id: Optional[str] = None,
    ) -> None:
        """Add one directed edge. A no-op if an edge with the same source/type/target already exists."""
        _validate_id(source, "edge source")
        _validate_id(edge_type, "edge type")
        _validate_id(target, "edge target")
        key = id or _edge_key(source, edge_type, target)
        _validate_id(key, "edge id")
        if key in self._edges.get(source, {}):
            return
        full = dict(data or {})
        if ownership is not None:
            full[_OWNERSHIP_KEY] = ownership
        candidate = {
            "id": key,
            "source": source,
            "type": edge_type,
            "target": target,
            "data": full,
            "createdAt": created_at if created_at is not None else 0,
            "revision": int(revision),
        }
        self._validate_edge_schema(candidate)
        self._edges.setdefault(source, {})[key] = candidate
        incoming = self._incoming.setdefault(target, {})
        incoming[source] = incoming.get(source, 0) + 1
        self._removed_edge_ids.discard(key)

    def get_edges(self, source: str, edge_type: Optional[str] = None) -> list:
        edges = self._edges.get(source, {})
        out = []
        for e in edges.values():
            if edge_type is None or e["type"] == edge_type:
                out.append(dict(e, data=dict(e.get("data") or {})))
        return out

    def update_edge(
        self,
        id_: str,
        data: Optional[dict] = None,
        ownership: Optional[Ownership] = None,
        expected_revision: Optional[int] = None,
    ) -> Optional[dict]:
        """Update an edge conditionally by its independent ID."""
        for edges in self._edges.values():
            edge = edges.get(id_)
            if edge is None:
                continue
            actual = int(edge.get("revision", 0))
            if expected_revision is not None and actual != expected_revision:
                raise ConflictError(f"record {id_} has revision {actual}, expected {expected_revision}")
            candidate = dict(edge, data=dict(edge.get("data") or {}), revision=actual + 1)
            if data is not None:
                candidate["data"] = dict(data)
            if ownership is not None:
                candidate.setdefault("data", {})[_OWNERSHIP_KEY] = ownership
            self._validate_edge_schema(candidate)
            edge.clear()
            edge.update(candidate)
            return dict(edge, data=dict(edge.get("data") or {}))
        return None

    def remove_edge(self, id_: str, expected_revision: Optional[int] = None) -> bool:
        """Remove exactly one edge by independent ID, conditionally by revision."""
        for source, edges in list(self._edges.items()):
            edge = edges.get(id_)
            if edge is None:
                continue
            actual = int(edge.get("revision", 0))
            if expected_revision is not None and actual != expected_revision:
                raise ConflictError(f"record {id_} has revision {actual}, expected {expected_revision}")
            if self._ownership(edge) == "owned" and not self._has_other_owned_source(edge["target"], source):
                self.remove_node(edge["target"])
                return True
            del edges[id_]
            if not edges:
                self._edges.pop(source, None)
            self._decrement_incoming(edge["target"], source)
            self._removed_edge_ids.add(id_)
            return True
        return False

    def get_edge_targets(self, source: str, edge_type: str) -> list:
        return [e["target"] for e in self.get_edges(source, edge_type)]

    def get_edge_sources(self, target: str, edge_type: str) -> list:
        sources = []
        for src in self._incoming.get(target, {}):
            for e in self._edges.get(src, {}).values():
                if e["type"] == edge_type and e["target"] == target:
                    sources.append(src)
                    break
        return sources

    def remove_edges(self, source: str, edge_type: Optional[str] = None, target: Optional[str] = None) -> None:
        """Remove edges from `source` matching `edge_type`/`target` (all outgoing edges if both omitted).
        'owned' edges cascade-delete their target unless another source also owns it; 'shared' edges
        invoke `on_orphan` if the target becomes disconnected."""
        edges = self._edges.get(source, {})
        to_remove = [
            (key, e)
            for key, e in edges.items()
            if (edge_type is None or e["type"] == edge_type) and (target is None or e["target"] == target)
        ]
        if not to_remove:
            return
        for _, e in to_remove:
            if self._ownership(e) == "owned" and not self._has_other_owned_source(e["target"], source):
                self.remove_node(e["target"])
        for key, e in to_remove:
            # Cascade deletion may have already removed this edge (and its
            # incoming-index entry) via remove_node above.
            removed = self._edges.get(source, {}).pop(key, None)
            if removed is not None:
                self._decrement_incoming(e["target"], source)
                self._removed_edge_ids.add(key)
            if self._ownership(e) == "shared" and not self._has_incoming(e["target"], source):
                if self._on_orphan:
                    self._on_orphan(e["target"])
        if not self._edges.get(source):
            self._edges.pop(source, None)

    # ── queries ──

    def query(self) -> "GraphQuery":
        """Create a mutable `GraphQuery` over the currently loaded nodes."""
        return GraphQuery(self)

    def clear(self) -> None:
        """Clear in-memory state only — does not flush pending deletions or touch the attached store."""
        self._nodes.clear()
        self._edges.clear()
        self._incoming.clear()
        self.vectors.clear()
        self._removed_node_ids.clear()
        self._removed_edge_ids.clear()
        self._removed_vector_ids.clear()
        if self._store is None:
            self._secondary_index_data = {name: {} for name in self._indexes}

    # ── persistence (Rust storage state machine) ──

    @classmethod
    def open(cls, directory: str, read_only: bool = False) -> "PolyGraph":
        """Open a directory-backed binary store and load its graph."""
        graph = cls()
        graph.open_store(directory, read_only=read_only)
        return graph

    @classmethod
    def restore(cls, source: str, destination: str) -> "PolyGraph":
        """Restore a directory backup into ``destination`` and open it."""
        source_path = Path(source)
        destination_path = Path(destination)
        if not source_path.is_dir():
            raise PolypackStorageError(f"backup source does not exist: {source}")
        destination_path.mkdir(parents=True, exist_ok=True)
        for name in ("snapshot.msgpack", "wal.msgpack", "mutations.jsonl", "indexes.json"):
            source_file = source_path / name
            if source_file.exists():
                shutil.copy2(source_file, destination_path / name)
        return cls.open(str(destination_path))

    def open_store(self, directory: str, read_only: bool = False) -> None:
        """Attach a directory-backed store and load any existing state."""
        self._store = _NativeStore(DirectoryStorage(directory, read_only=read_only))
        self._store_directory = Path(directory)
        self._store_read_only = read_only
        self._load_index_metadata()
        self._load_from_store()
        try:
            self._validate_all_indexes()
        except Exception:
            self._store.close()
            self._store = None
            self._store_directory = None
            self._store_read_only = False
            raise

    def checkpoint(self) -> None:
        """Persist pending mutations and compact the WAL into a snapshot."""
        if self._store is None:
            raise PolypackStorageError("no store open; call open_store(path) first")
        if self._store_read_only:
            raise PolypackStorageError("store was opened read-only")
        self.save()
        self._store.compact()

    def backup(self, destination: str) -> None:
        """Create a consistent directory backup after checkpointing the store."""
        if self._store is None or self._store_directory is None:
            raise PolypackStorageError("no store open; call open_store(path) first")
        if self._store_read_only:
            raise PolypackStorageError("store was opened read-only")
        self.checkpoint()
        destination_path = Path(destination)
        destination_path.mkdir(parents=True, exist_ok=True)
        for name in ("snapshot.msgpack", "wal.msgpack", "mutations.jsonl", "indexes.json"):
            source_file = self._store_directory / name
            if source_file.exists():
                shutil.copy2(source_file, destination_path / name)

    def close_store(self) -> None:
        """Persist any unsaved changes, then compact and close the attached
        store. Safe to call repeatedly. Mirrors the Rust `Graph::close`,
        which flushes before closing for the same reason: without this,
        mutations made since the last explicit `save()` — including inside
        a `with PolyGraph.open(...) as g:` block — would be silently lost."""
        if self._store is not None:
            if not self._store_read_only:
                self.save()
            self._store.close()
            self._store = None
            self._store_directory = None
            self._store_read_only = False

    def save(self) -> None:
        """Persist the full current graph state through the attached store.

        Also flushes deletions recorded since the last successful save, so
        nodes/edges/vectors removed via remove_node/remove_edges do not
        resurrect on the next open().
        """
        if self._store is None:
            raise PolypackStorageError("no store open; call open_store(path) first")
        nodes = []
        for node in self._nodes.values():
            nodes.append(_copy_node(node))
        edges = []
        for edge_map in self._edges.values():
            for e in edge_map.values():
                edges.append(
                    {
                        "id": e["id"] if "id" in e else _edge_key(e["source"], e["type"], e["target"]),
                        "source": e["source"],
                        "target": e["target"],
                        "type": e["type"],
                        "data": dict(e.get("data") or {}),
                        "createdAt": e.get("createdAt", 0),
                        "revision": int(e.get("revision", 0)),
                    }
                )
        vectors = [{"id": id_, "vector": vector} for id_, vector in self.vectors.entries()]
        self._store.apply(
            put_nodes=nodes,
            delete_node_ids=list(self._removed_node_ids),
            put_edges=edges,
            delete_edge_ids=list(self._removed_edge_ids),
            put_vectors=vectors,
            delete_vector_ids=list(self._removed_vector_ids),
            transaction_id=self._active_transaction.id if self._active_transaction is not None else None,
            operation_id=self._active_transaction.operation_id if self._active_transaction is not None else None,
            actor=self._active_transaction.actor if self._active_transaction is not None else None,
            base_revision=self._active_transaction.base_revision if self._active_transaction is not None else None,
            metadata=self._active_transaction.metadata if self._active_transaction is not None else None,
        )
        self._removed_node_ids.clear()
        self._removed_edge_ids.clear()
        self._removed_vector_ids.clear()

    def _load_from_store(self) -> None:
        if self._store is None:
            return
        for id_ in self._store.all_node_ids():
            node = self._store.get_node(id_)
            if node:
                self.add_node(node)
        for edge in self._store.all_edges():
            data = dict(edge.get("data") or {})
            self.add_edge(
                edge["source"],
                edge["type"],
                edge["target"],
                data,
                created_at=edge.get("createdAt"),
                revision=edge.get("revision", 0),
                id=edge.get("id"),
            )
        for id_, vector in self._store.all_vectors():
            if id_ not in self._nodes:
                self.vectors.add(id_, vector)

    @property
    def size(self) -> int:
        return len(self._nodes)

    # ── activation (mirrors PolyGraph.reinforceNode etc.) ──

    def reinforce_node(self, id_: str, amount: float, reason: Optional[str] = None) -> Optional[Node]:
        """Apply a durable reinforcement delta. The prior state is decay-corrected
        to now, `amount` is added to `score`, a fraction folds into `importance`,
        the counter increments, and the decay anchor re-sets to now. Returns the
        updated node, or `None` if the node isn't loaded. (Python has no change
        events, so no notification is emitted.)"""
        if not isinstance(amount, (int, float)) or not math.isfinite(float(amount)):
            raise PolypackValueError("reinforcement amount must be finite")
        node = self._nodes.get(id_)
        if node is None:
            return None
        now = int(time.time() * 1000)
        node["activation"] = _reinforce_activation(node.get("activation"), float(amount), now)
        node["updatedAt"] = now
        return _copy_node(node)

    def reinforce_node_safe(self, id_: str, amount: float, reason: Optional[str] = None) -> Optional[Node]:
        """Alias for `reinforce_node` — the Python graph has no hot cache or
        eviction, so there is nothing to restore."""
        return self.reinforce_node(id_, amount, reason)

    def get_activation(self, id_: str, half_life_ms: Optional[float] = None) -> float:
        """Current decayed activation score of a node (0 when it has none)."""
        node = self._nodes.get(id_)
        return 0.0 if node is None else _activation_score_of(node, half_life_ms=half_life_ms)

    def get_activation_state(self, id_: str) -> Optional[dict]:
        """Decay-corrected view of a node's durable activation, or `None`."""
        node = self._nodes.get(id_)
        if node is None or node.get("activation") is None:
            return None
        return _decay_activation_state(node["activation"], int(time.time() * 1000))

    def top_activated(self, limit: int, min_score: float = 0.0) -> list:
        """Loaded nodes with the highest current activation, descending."""
        if not isinstance(limit, int) or limit < 0:
            raise PolypackValueError("limit must be a non-negative integer")
        if limit == 0:
            return []
        now = int(time.time() * 1000)
        scored = []
        for node in self._nodes.values():
            score = _activation_score_of(node, now)
            if score > min_score:
                scored.append((score, _copy_node(node)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:limit]]

    def decay(self, now: Optional[int] = None) -> None:
        """Materialize decay for every loaded node: rewrite stored activation to
        its current decayed state and re-anchor to `now`. Reads already decay
        lazily, so this only matters for persisting fresh values."""
        now = now if now is not None else int(time.time() * 1000)
        for node in self._nodes.values():
            if node.get("activation") is None:
                continue
            corrected = _decay_activation_state(node["activation"], now)
            node["activation"] = {
                "score": corrected["score"],
                "importance": corrected["importance"],
                "reinforcementCount": node["activation"]["reinforcementCount"],
                "lastMeaningfulActivation": now,
            }
            node["updatedAt"] = now

    # ── ownership internals ──

    def _ownership(self, edge: dict) -> Ownership:
        return edge.get("data", {}).get(_OWNERSHIP_KEY, "reference")

    def _has_other_owned_source(self, target: str, exclude: str) -> bool:
        for src in self._incoming.get(target, {}):
            if src == exclude:
                continue
            for e in self._edges.get(src, {}).values():
                if e["target"] == target and self._ownership(e) == "owned":
                    return True
        return False

    def _has_incoming(self, target: str, exclude: str) -> bool:
        for src in self._incoming.get(target, {}):
            if src == exclude:
                continue
            if any(e["target"] == target for e in self._edges.get(src, {}).values()):
                return True
        return False

    def _decrement_incoming(self, target: str, source: str) -> None:
        counts = self._incoming.get(target)
        if not counts or source not in counts:
            return
        counts[source] -= 1
        if counts[source] <= 0:
            del counts[source]
        if not counts:
            self._incoming.pop(target, None)

    def _cleanup_edges(self, id_: str) -> None:
        # Remove edges where id_ is the target (from every incoming source).
        for src in list(self._incoming.get(id_, {}).keys()):
            edges = self._edges.get(src, {})
            for key in [k for k, e in edges.items() if e["target"] == id_]:
                del edges[key]
                self._removed_edge_ids.add(key)
            if not edges:
                self._edges.pop(src, None)
        self._incoming.pop(id_, None)
        # Remove edges where id_ is the source, and drop id_ from the
        # incoming index of each of those edges' targets.
        outgoing = self._edges.pop(id_, None)
        if outgoing:
            for key, e in outgoing.items():
                self._decrement_incoming(e["target"], id_)
                self._removed_edge_ids.add(key)


# ── Query builder ──


class GraphQuery:
    def __init__(self, graph: PolyGraph) -> None:
        self._graph = graph
        self._node_types: Optional[list] = None
        self._attributes: list = []
        self._order_by: Optional[tuple] = None
        self._offset: Optional[int] = None
        self._limit: Optional[int] = None
        self._traversal: list = []
        self._joins: list = []
        self._similarity: Optional[dict] = None
        self._activation_above: Optional[float] = None
        self._activation_order: Optional[str] = None
        self._limits = graph.resource_limit_config

    def where_type(self, *types: str) -> "GraphQuery":
        self._node_types = list(types)
        return self

    def where(self, field: str, value: Any) -> "GraphQuery":
        self._attributes.append(("eq", field, value))
        return self

    def where_range(self, field: str, above: Optional[float] = None, below: Optional[float] = None) -> "GraphQuery":
        if above is not None and not math.isfinite(float(above)):
            raise PolypackValueError("range above must be finite")
        if below is not None and not math.isfinite(float(below)):
            raise PolypackValueError("range below must be finite")
        self._attributes.append(("range", field, (above, below)))
        return self

    def order_by(self, field: str, direction: str = "asc") -> "GraphQuery":
        self._order_by = (field, direction)
        return self

    def where_activated(self, above: float) -> "GraphQuery":
        """Keep only nodes whose current (decay-corrected) activation exceeds `above`."""
        if not math.isfinite(float(above)):
            raise PolypackValueError("above must be finite")
        self._activation_above = float(above)
        return self

    def order_by_activation(self, direction: str = "desc") -> "GraphQuery":
        """Order results by current (decay-corrected) activation instead of a data field."""
        if direction not in ("asc", "desc"):
            raise PolypackValueError("direction must be 'asc' or 'desc'")
        self._activation_order = direction
        return self

    def offset(self, n: int) -> "GraphQuery":
        if not isinstance(n, int) or n < 0:
            raise PolypackValueError("offset must be a non-negative integer")
        self._offset = n
        return self

    def limit(self, n: int) -> "GraphQuery":
        if not isinstance(n, int) or n < 0:
            raise PolypackValueError("limit must be a non-negative integer")
        self._limit = n
        return self

    def traverse(self, edge_type: str, depth: int, direction: str = "out") -> "GraphQuery":
        _validate_id(edge_type, "edge type")
        if not isinstance(depth, int) or depth < 0:
            raise PolypackValueError("depth must be a non-negative integer")
        maximum = self._limits.get("maxTraversalDepth")
        if maximum is not None and depth > maximum:
            raise ResourceLimitError("maxTraversalDepth", maximum)
        self._traversal.append((edge_type, depth, direction))
        return self

    def join(self, edge_type: str, direction: str = "out") -> "GraphQuery":
        _validate_id(edge_type, "edge type")
        self._joins.append((edge_type, direction))
        return self

    def similar_to(self, vector: Any, threshold: float = 0.0, top_k: Optional[int] = None) -> "GraphQuery":
        q = _validate_vector(vector, "query vector")
        if not math.isfinite(float(threshold)):
            raise PolypackValueError("threshold must be finite")
        self._similarity = {"vector": q, "threshold": float(threshold), "top_k": top_k}
        return self

    def explain(self) -> dict:
        """Describe the query stages and a coarse execution-cost estimate."""
        fields = {field for op, field, _ in self._attributes}
        node_type = self._node_types[0] if self._node_types and len(self._node_types) == 1 else None
        selected = next(
            (definition for definition in self._graph._indexes.values()
             if (not definition["nodeType"] or definition["nodeType"] == node_type)
             and all(field in fields for field in definition["fields"])),
            None,
        )
        stages = [f"property-index({selected['name']})" if selected else "record-scan"]
        if self._node_types:
            stages.append(f"type-filter({','.join(self._node_types)})")
        if self._attributes:
            stages.append("property-filter")
        if self._joins:
            stages.append(f"join(count={len(self._joins)})")
        if self._traversal:
            stages.append(f"traversal(depth={max(step[1] for step in self._traversal)})")
        if self._order_by:
            stages.append(f"order({self._order_by[0]},{self._order_by[1]})")
        if self._limit is not None:
            stages.append(f"limit({self._limit})")
        loaded = len(self._graph._nodes)
        return {
            "index": selected["name"] if selected else None,
            "stages": stages,
            "loadedRecords": loaded,
            "estimatedCost": max(1, loaded * (0.25 if selected else 1)),
        }

    def _match(self, node: Node) -> bool:
        if self._node_types is not None and node["type"] not in self._node_types:
            return False
        data = node.get("data") or {}
        for op, field, value in self._attributes:
            if op == "eq":
                actual = node["type"] if field == "type" else data.get(field)
                if actual != value:
                    return False
            else:
                above, below = value
                actual = data.get(field)
                if actual is None:
                    return False
                if above is not None and actual <= above:
                    return False
                if below is not None and actual >= below:
                    return False
        if self._activation_above is not None and _activation_score_of(node) < self._activation_above:
            return False
        return True

    def _connected(self, node: Node, edge_type: str, direction: str) -> bool:
        if direction == "out":
            return any(e["type"] == edge_type for e in self._graph.get_edges(node["id"]))
        return any(e["type"] == edge_type for e in self._graph.get_edge_sources(node["id"], edge_type))

    def _bfs(self, seeds: list, edge_type: str, depth: int, direction: str) -> list:
        """Breadth-first expansion; returns nodes in discovery order (seeds first)."""
        visited = list(seeds)
        seen = set(seeds)
        maximum = self._limits.get("maxNodesVisited")
        if maximum is not None and len(seen) > maximum:
            raise ResourceLimitError("maxNodesVisited", maximum)
        frontier = list(seeds)
        for _ in range(depth):
            if not frontier:
                break
            nxt = []
            for node_id in frontier:
                if direction == "out":
                    targets = [e["target"] for e in self._graph.get_edges(node_id, edge_type)]
                    for t in targets:
                        if t not in seen:
                            seen.add(t)
                            visited.append(t)
                            if maximum is not None and len(seen) > maximum:
                                raise ResourceLimitError("maxNodesVisited", maximum)
                            nxt.append(t)
                else:
                    sources = self._graph.get_edge_sources(node_id, edge_type)
                    for s in sources:
                        if s not in seen:
                            seen.add(s)
                            visited.append(s)
                            if maximum is not None and len(seen) > maximum:
                                raise ResourceLimitError("maxNodesVisited", maximum)
                            nxt.append(s)
            frontier = nxt
        return visited

    def _to_plan(self) -> dict:
        """Express this query as the shared query-plan IR."""
        plan: dict = {}
        if self._node_types:
            plan["nodeTypes"] = list(self._node_types)
        if self._attributes:
            attrs = []
            for op, field, value in self._attributes:
                if op == "eq":
                    attrs.append({"field": field, "operator": "eq", "value": value})
                else:
                    above, below = value
                    entry = {"field": field, "operator": "range"}
                    if above is not None:
                        entry["above"] = above
                    if below is not None:
                        entry["below"] = below
                    attrs.append(entry)
            plan["attributes"] = attrs
        if self._traversal:
            plan["traversal"] = [
                {"edgeType": et, "direction": d, "depth": dep} for et, dep, d in self._traversal
            ]
        if self._joins:
            plan["joins"] = [{"edgeType": et, "direction": d} for et, d in self._joins]
        if self._similarity:
            sim = {
                "vector": self._similarity["vector"],
                "threshold": self._similarity["threshold"],
            }
            if self._similarity["top_k"] is not None:
                sim["topK"] = self._similarity["top_k"]
            plan["similarity"] = sim
        if self._order_by:
            field, direction = self._order_by
            plan["order"] = {"field": field, "direction": direction}
        if self._offset is not None:
            plan["offset"] = self._offset
        if self._limit is not None:
            plan["limit"] = self._limit
        return plan

    def _native_ids(self, plan: dict) -> Optional[list]:
        """Run the plan through the Rust query executor; None on failure."""
        if self._activation_above is not None or self._activation_order is not None:
            # The native executor doesn't understand activation filters — force
            # the pure-Python pipeline so results can't silently diverge.
            return None
        try:
            nodes_source = self._indexed_candidates() if not self._joins and not self._traversal else self._graph._nodes.values()
            nodes = [dict(n) for n in nodes_source]
            edges = []
            for edge_map in self._graph._edges.values():
                for e in edge_map.values():
                    edges.append(
                        {
                            "id": e["id"] if "id" in e else _edge_key(e["source"], e["type"], e["target"]),
                            "source": e["source"],
                            "target": e["target"],
                            "type": e["type"],
                            "data": dict(e.get("data") or {}),
                            "createdAt": e.get("createdAt", 0),
                            "revision": int(e.get("revision", 0)),
                        }
                    )
            return list(_execute_query_plan(nodes, edges, plan))
        except (TypeError, PolypackValueError):
            # Expected fallback: e.g. a non-JSON-serialisable filter value.
            # Other exceptions (native bugs, corrupt state) propagate.
            return None

    def _collect(self) -> list:
        started = time.perf_counter()
        plan = self._to_plan()
        scanned_records = len(self._graph._nodes) if self._joins or self._traversal else len(self._indexed_candidates())
        native_ids = self._native_ids(plan)
        if native_ids is not None:
            maximum = self._limits.get("maxNodesVisited")
            if maximum is not None and len(native_ids) > maximum:
                raise ResourceLimitError("maxNodesVisited", maximum)
            by_id = {n["id"]: n for n in self._graph._nodes.values()}
            results = [by_id[i] for i in native_ids if i in by_id]
            maximum = self._limits.get("maxResults")
            if maximum is not None and len(results) > maximum:
                raise ResourceLimitError("maxResults", maximum)
            self._graph._record_query((time.perf_counter() - started) * 1000, scanned_records, self.explain()["index"])
            return results
        # Fallback: pure-Python pipeline (only reached if the native path
        # failed, e.g. a non-JSON-serialisable filter value).
        results = self._collect_python(plan)
        self._graph._record_query((time.perf_counter() - started) * 1000, scanned_records, self.explain()["index"])
        return results

    def _collect_python(self, plan: dict) -> list:
        candidates = self._indexed_candidates()
        results = [n for n in candidates if self._match(n)]
        if self._joins:
            results = [n for n in results if all(self._connected(n, et, d) for et, d in self._joins)]
        for edge_type, depth, direction in self._traversal:
            ids = [n["id"] for n in results]
            order = self._bfs(ids, edge_type, depth, direction)
            by_id = {n["id"]: n for n in self._graph._nodes.values()}
            results = [by_id[i] for i in order if i in by_id]
        if self._order_by:
            field, direction = self._order_by
            results = sorted(
                results,
                key=lambda n: (n.get("data") or {}).get(field, 0) or 0,
                reverse=(direction == "desc"),
            )
        if self._activation_order is not None:
            results = sorted(
                results,
                key=lambda n: _activation_score_of(n),
                reverse=(self._activation_order == "desc"),
            )
        if self._similarity:
            sim = self._similarity
            scored = []
            for n in results:
                if n.get("vector") is None:
                    continue
                score = _cosine(sim["vector"], n["vector"])
                if score >= sim["threshold"]:
                    scored.append((score, n))
            scored.sort(key=lambda x: x[0], reverse=True)
            if sim["top_k"] is not None:
                scored = scored[: sim["top_k"]]
            results = [n for _, n in scored]
        if self._offset is not None:
            results = results[self._offset :]
        if self._limit is not None:
            results = results[: self._limit]
        maximum = self._limits.get("maxResults")
        if maximum is not None and len(results) > maximum:
            raise ResourceLimitError("maxResults", maximum)
        return results

    def _indexed_candidates(self) -> list:
        explanation = self.explain()
        selected = explanation["index"]
        if selected is None or selected == "type-index":
            return list(self._graph._nodes.values())
        definition = self._graph._indexes[selected]
        equalities = {field: value for op, field, value in self._attributes if op == "eq"}
        buckets = self._graph._secondary_index_data.get(selected, {})
        if all(field in equalities for field in definition["fields"]):
            expected = repr([equalities[field] for field in definition["fields"]])
            ids = buckets.get(expected, set())
            return [self._graph._nodes[id_] for id_ in ids if id_ in self._graph._nodes]
        if len(definition["fields"]) == 1:
            field = definition["fields"][0]
            ranges = [(above, below) for op, name, (above, below) in self._attributes if op == "range" and name == field]
            if ranges:
                above, below = ranges[-1]
                ids: set[str] = set()
                for encoded, bucket in buckets.items():
                    try:
                        value = ast.literal_eval(encoded)[0]
                    except (ValueError, SyntaxError, IndexError, TypeError):
                        continue
                    if not isinstance(value, (int, float)) or isinstance(value, bool):
                        continue
                    if (above is not None and value <= above) or (below is not None and value >= below):
                        continue
                    ids.update(bucket)
                return [self._graph._nodes[id_] for id_ in ids if id_ in self._graph._nodes]
        return list(self._graph._nodes.values())

    def to_list(self) -> list:
        return [_copy_node(n) for n in self._collect()]

    def ids(self) -> list:
        return [n["id"] for n in self._collect()]

    def first(self) -> Optional[Node]:
        nodes = self.to_list()
        return nodes[0] if nodes else None

    def count(self) -> int:
        return len(self._collect())

    def aggregate(self, field: str, op: str) -> dict:
        values = [
            (n.get("data") or {}).get(field)
            for n in self._collect()
            if isinstance((n.get("data") or {}).get(field), (int, float))
            and math.isfinite(float((n.get("data") or {}).get(field)))
        ]
        if not values:
            return {"value": 0, "count": 0}
        if op == "sum":
            value = sum(values)
        elif op == "avg":
            value = sum(values) / len(values)
        elif op == "min":
            value = min(values)
        elif op == "max":
            value = max(values)
        elif op == "count":
            value = len(values)
        else:
            raise PolypackValueError(f"unknown aggregate op {op}")
        return {"value": value, "count": len(values)}


# ── ActivationEngine (adaptive memory, mirrors the Rust / TypeScript engine) ──


class ActivationEngine:
    """Adaptive-memory layer over a `PolyGraph`.

    Two tiers: durable reinforcement (persisted through the store) and
    transient attention (local to this engine, never serialized). `spread`
    implements spreading activation, `pulse`/`absorb` the semantic +
    relational + recency + usage composite, and `working_memory` materializes
    the current "mental state". `pulse`/`absorb` take a raw query vector —
    embed text first with your own embedding provider.
    """

    def __init__(self, graph: "PolyGraph", config: Optional[dict] = None) -> None:
        self.graph = graph
        cfg: dict = {
            "scoreHalfLifeMs": DEFAULT_ACTIVATION["scoreHalfLifeMs"],
            "importanceHalfLifeMs": DEFAULT_ACTIVATION["importanceHalfLifeMs"],
            "importanceGain": DEFAULT_ACTIVATION["importanceGain"],
            "spreadDecay": 0.5,
            "spreadDepth": 2,
            "recencyHalfLifeMs": 7 * _DAY,
            "weights": {"semantic": 1, "graph": 1, "recency": 1, "usage": 1},
            "minReinforceDelta": 0.05,
            "pulseThreshold": 0.0,
            "absorbThreshold": 0.3,
            "absorbGain": 0.05,
        }
        cfg.update(config or {})
        weights = dict(cfg["weights"])
        cfg["weights"] = weights
        self.config = cfg
        self.attention: dict = {}

    def dispose(self) -> None:
        """Drop transient attention. Durable state is untouched."""
        self.attention.clear()

    # ── transient attention (local, never synced) ──

    def bump_attention(self, id_: str, amount: float) -> None:
        """Accumulate runtime-only attention, promoting to durable reinforcement
        once it clears `minReinforceDelta`."""
        next_ = _clamp01(self.attention.get(id_, 0.0) + amount)
        self.attention[id_] = next_
        if next_ >= self.config["minReinforceDelta"]:
            self.attention.pop(id_, None)
            self.graph.reinforce_node(id_, next_, "attention")

    def attention_of(self, id_: str) -> float:
        return self.attention.get(id_, 0.0)

    def effective(self, id_: str) -> float:
        """Durable decayed score (using `scoreHalfLifeMs`) plus transient attention."""
        node = self.graph.get_node(id_)
        durable = 0.0
        if node and node.get("activation"):
            act = node["activation"]
            now = int(time.time() * 1000)
            durable = _clamp01(
                act["score"] * decay_factor(now - act["lastMeaningfulActivation"], self.config["scoreHalfLifeMs"])
            )
        return _clamp01(durable + self.attention_of(id_))

    # ── durable reinforcement ──

    def reinforce(self, id_: str, amount: float, reason: Optional[str] = None) -> Optional[Node]:
        return self.graph.reinforce_node(id_, amount, reason)

    def reinforce_all(self, entries: Iterable[tuple]) -> None:
        for id_, amount, reason in entries:
            self.graph.reinforce_node(id_, amount, reason)

    # ── relational spreading activation ──

    def spread(
        self,
        seeds: Iterable[str],
        depth: Optional[int] = None,
        decay: Optional[float] = None,
        edge_types: Optional[Sequence[str]] = None,
    ) -> dict:
        """Spread activation outward from `seeds` across outgoing edges. Each hop
        attenuates the contribution by `decay`; multiple paths to a node sum."""
        depth = depth if depth is not None else self.config["spreadDepth"]
        decay = decay if decay is not None else self.config["spreadDecay"]
        contributions: dict = {}
        visited = set(seeds)
        frontier = list(seeds)
        for hop in range(depth):
            if not frontier:
                break
            nxt = []
            for id_ in frontier:
                for e in self.graph.get_edges(id_):
                    if edge_types is not None and e["type"] not in edge_types:
                        continue
                    if e["target"] in visited:
                        continue
                    visited.add(e["target"])
                    contributions[e["target"]] = contributions.get(e["target"], 0.0) + decay ** (hop + 1)
                    nxt.append(e["target"])
            frontier = nxt
        return contributions

    # ── semantic pulse / absorb ──

    def pulse(
        self,
        vector: Any,
        top_k: Optional[int] = None,
        semantic_threshold: float = 0.0,
        pulse_threshold: Optional[float] = None,
        depth: Optional[int] = None,
        decay: Optional[float] = None,
        edge_types: Optional[Sequence[str]] = None,
    ) -> list:
        """Score the region of the graph around `vector`: semantic seeds via vector
        similarity (nodes with zero similarity never seed the region), outward
        spreading activation, folded with recency and usage. Returns ranked
        `(node_id, composite)` pairs."""
        threshold = pulse_threshold if pulse_threshold is not None else self.config["pulseThreshold"]
        now = int(time.time() * 1000)
        q = _validate_vector(vector, "query vector")
        top_k = top_k if top_k is not None else len(self.graph.vectors)
        semantic: dict = {}
        for id_, score in self.graph.vectors.query(q, top_k, 0.0):
            if score > max(semantic_threshold, 0.0):
                semantic[id_] = score
        graph_contrib = self.spread(list(semantic.keys()), depth=depth, decay=decay, edge_types=edge_types)
        w = self.config["weights"]
        scores: dict = {}
        for id_ in set(semantic) | set(graph_contrib):
            node = self.graph.get_node(id_)
            if node is None:
                continue
            s = semantic.get(id_, 0.0)
            g = graph_contrib.get(id_, 0.0)
            recency = decay_factor(now - node["insertedAt"], self.config["recencyHalfLifeMs"])
            activation = node.get("activation")
            usage = activation["importance"] if activation else 0.0
            scores[id_] = w["semantic"] * s + w["graph"] * g + w["recency"] * recency + w["usage"] * usage
        return sorted(
            ((id_, score) for id_, score in scores.items() if score > threshold),
            key=lambda x: x[1],
            reverse=True,
        )

    def absorb(self, vector: Any, **options) -> list:
        """`pulse` plus reinforcement: nodes whose composite clears
        `absorbThreshold` receive durable reinforcement of `absorbGain * score`."""
        scores = self.pulse(vector, **options)
        for id_, score in scores:
            if score >= self.config["absorbThreshold"]:
                self.graph.reinforce_node(id_, _clamp01(self.config["absorbGain"] * score), "pulse")
        return scores

    # ── working memory ──

    def working_memory(self, limit: int = 10, min_score: float = 0.0) -> list:
        """The current "mental state": loaded nodes ranked by `effective`
        activation descending, top `limit`."""
        scored = []
        for node in self.graph._nodes.values():
            score = self.effective(node["id"])
            if score > min_score:
                scored.append((score, _copy_node(node)))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [n for _, n in scored[:limit]]


# ── Change batch (v1: structural contract; execution is Phase 5) ──


@dataclass
class ChangeBatch:
    put_nodes: list = field(default_factory=list)
    delete_node_ids: list = field(default_factory=list)
    put_edges: list = field(default_factory=list)
    delete_edge_ids: list = field(default_factory=list)
    put_vectors: list = field(default_factory=list)
    delete_vector_ids: list = field(default_factory=list)

    def validate(self) -> None:
        for n in self.put_nodes:
            _validate_id(n.get("id"), "node id")
            _validate_id(n.get("type"), "node type")
        for e in self.put_edges:
            _validate_id(e.get("source"), "edge source")
            _validate_id(e.get("type"), "edge type")
            _validate_id(e.get("target"), "edge target")
