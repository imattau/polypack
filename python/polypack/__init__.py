"""Polypack — embedded local-first property graph with vector search.

Python-native graph layer over the Rust vector core (`polypack._core`).
Semantics mirror `specification/data-model.md` and the TypeScript reference.
"""

from __future__ import annotations

import copy
import ast
import json
import math
import os
import shutil
import uuid
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence

if os.name == "nt":
    import msvcrt
else:
    import fcntl

import numpy as np

from ._core import (
    ExactIndex as _NativeExactIndex,
    HnswIndex as _NativeHnswIndex,
    NativeStore as _NativeStore,
    engine_info as _engine_info,
    execute_query_plan as _execute_query_plan,
)
from .sync import FileSyncOperationLog, SyncServer, sync_checksum, sync_identity_checksum, validate_sync_batch, validate_sync_operation
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


class AdapterCapabilityError(PolypackError):
    """Raised when an operation requires an adapter guarantee it does not have."""

    def __init__(self, capability: str) -> None:
        super().__init__(f"Persistence adapter does not support capability: {capability}")
        self.capability = capability


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
            originals = nodes[start:start + batch_size]
            batch = [migrate_node(copy.deepcopy(node)) for node in originals]
            for original, migrated in zip(originals, batch):
                if migrated["id"] != original["id"] or migrated["type"] != original["type"]:
                    raise MigrationError(f"node migration changed identity or type for {original['id']}")
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
            for before, migrated in zip(original, batch):
                if migrated["id"] != before["id"] or migrated["source"] != before["source"] or migrated["target"] != before["target"] or migrated["type"] != before["type"]:
                    raise MigrationError(f"edge migration changed identity or endpoints for {before['id']}")
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
    "GraphSnapshot",
    "GraphTransaction",
    "GraphQuery",
    "PersistedGraphQuery",
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
    "AdapterCapabilityError",
    "ResourceLimitError",
    "MigrationError",
    "MigrationRegistry",
    "UniqueConstraintError",
    "SyncServer",
    "FileSyncOperationLog",
    "sync_checksum",
    "sync_identity_checksum",
    "validate_sync_batch",
    "validate_sync_operation",
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
    parts = path.split(".")
    if parts[0] == "data":
        parts = parts[1:]
    if not parts or any(not part for part in parts):
        raise PolypackValueError("patch paths must target node data")
    return parts


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


def _patch_has(root: dict, path: str) -> bool:
    current: Any = root
    for part in _patch_parts(path):
        if not isinstance(current, dict) or part not in current:
            return False
        current = current[part]
    return True


def _default_node_similarity(a: dict, b: dict) -> float:
    """Cosine similarity between two nodes' vectors, or 0 when either lacks one."""
    av, bv = a.get("vector"), b.get("vector")
    if not av or not bv:
        return 0.0
    return _cosine(av, bv)


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
    "inhibitionHalfLifeMs": 12 * _HOUR,
    "contextHalfLifeMs": _DAY,
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


def _decay_context_entry(entry: dict, now: int, half_life_ms: float) -> dict:
    return {
        "score": _clamp01(entry["score"] * decay_factor(now - entry["lastMeaningfulActivation"], half_life_ms)),
        "lastMeaningfulActivation": entry["lastMeaningfulActivation"],
    }


def _decay_activation_state(
    activation: dict,
    now: int,
    score_half_life_ms: float = DEFAULT_ACTIVATION["scoreHalfLifeMs"],
    importance_half_life_ms: float = DEFAULT_ACTIVATION["importanceHalfLifeMs"],
    inhibition_half_life_ms: float = DEFAULT_ACTIVATION["inhibitionHalfLifeMs"],
    context_half_life_ms: float = DEFAULT_ACTIVATION["contextHalfLifeMs"],
) -> dict:
    result = {
        "score": _clamp01(activation["score"] * decay_factor(now - activation["lastMeaningfulActivation"], score_half_life_ms)),
        "importance": _clamp01(
            activation["importance"] * decay_factor(now - activation["lastMeaningfulActivation"], importance_half_life_ms)
        ),
        "reinforcementCount": activation["reinforcementCount"],
        "lastMeaningfulActivation": activation["lastMeaningfulActivation"],
    }
    if activation.get("inhibition") is not None:
        anchor = activation.get("lastInhibitedAt", activation["lastMeaningfulActivation"])
        result["inhibition"] = _clamp01(activation["inhibition"] * decay_factor(now - anchor, inhibition_half_life_ms))
        result["lastInhibitedAt"] = activation.get("lastInhibitedAt")
    if activation.get("context"):
        result["context"] = {
            key: _decay_context_entry(entry, now, context_half_life_ms) for key, entry in activation["context"].items()
        }
    return result


def _reinforce_activation(previous: Optional[dict], delta: float, now: int, context: Optional[str] = None) -> dict:
    decayed = _decay_activation_state(previous, now) if previous else None
    score = decayed["score"] if decayed else 0.0
    importance = decayed["importance"] if decayed else 0.0
    count = (previous.get("reinforcementCount", 0) + 1) if previous else 1

    result = {
        "score": _clamp01(score + delta),
        "importance": _clamp01(importance + DEFAULT_ACTIVATION["importanceGain"] * delta),
        "reinforcementCount": count,
        "lastMeaningfulActivation": now,
    }
    if decayed and decayed.get("inhibition") is not None:
        result["inhibition"] = decayed["inhibition"]
        result["lastInhibitedAt"] = decayed.get("lastInhibitedAt")

    context_map = dict((decayed or {}).get("context") or (previous or {}).get("context") or {})
    if context:
        entry = context_map.get(context, {"score": 0.0, "lastMeaningfulActivation": now})
        context_map[context] = {"score": _clamp01(entry["score"] + delta), "lastMeaningfulActivation": now}
    if context_map:
        result["context"] = context_map
    return result


def _suppress_activation(
    previous: Optional[dict],
    delta: float,
    now: int,
    inhibition_half_life_ms: float = DEFAULT_ACTIVATION["inhibitionHalfLifeMs"],
) -> dict:
    """Mirrors `_reinforce_activation` but for the `inhibition` axis: decay-correct
    the prior inhibition to `now`, add `delta`, clamp, and re-anchor `lastInhibitedAt`.
    A negative `delta` releases suppression. `score`/`importance` are only
    decay-corrected, not re-anchored."""
    decayed = (
        _decay_activation_state(
            previous,
            now,
            DEFAULT_ACTIVATION["scoreHalfLifeMs"],
            DEFAULT_ACTIVATION["importanceHalfLifeMs"],
            inhibition_half_life_ms,
        )
        if previous
        else None
    )
    result = {
        "score": decayed["score"] if decayed else 0.0,
        "importance": decayed["importance"] if decayed else 0.0,
        "reinforcementCount": previous.get("reinforcementCount", 0) if previous else 0,
        "lastMeaningfulActivation": decayed["lastMeaningfulActivation"] if decayed else now,
        "inhibition": _clamp01((decayed.get("inhibition", 0.0) if decayed else 0.0) + delta),
        "lastInhibitedAt": now,
    }
    if decayed and decayed.get("context"):
        result["context"] = decayed["context"]
    return result


def merge_activation(existing: dict, incoming: dict, now: Optional[int] = None) -> dict:
    """Merge two durable activation records (total-state max-merge). Decay-corrects
    both to `now`, keeps the stronger component of each, and re-anchors to `now`.
    Concurrent deltas accumulate additively instead — activation is accumulated
    knowledge, not last-write-wins data."""
    now = now if now is not None else int(time.time() * 1000)
    ex = _decay_activation_state(existing, now)
    inc = _decay_activation_state(incoming, now)
    result = {
        "score": max(ex["score"], inc["score"]),
        "importance": max(ex["importance"], inc["importance"]),
        "reinforcementCount": max(existing["reinforcementCount"], incoming["reinforcementCount"]),
        "lastMeaningfulActivation": now,
    }
    if ex.get("inhibition") is not None or inc.get("inhibition") is not None:
        result["inhibition"] = max(ex.get("inhibition") or 0.0, inc.get("inhibition") or 0.0)
        result["lastInhibitedAt"] = now
    if ex.get("context") or inc.get("context"):
        context: dict = {}
        for key in set((ex.get("context") or {}).keys()) | set((inc.get("context") or {}).keys()):
            a = (ex.get("context") or {}).get(key, {}).get("score", 0.0)
            b = (inc.get("context") or {}).get(key, {}).get("score", 0.0)
            context[key] = {"score": max(a, b), "lastMeaningfulActivation": now}
        result["context"] = context
    return result


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
    result = {
        "score": float(activation["score"]),
        "importance": float(activation["importance"]),
        "reinforcementCount": int(count),
        "lastMeaningfulActivation": int(anchor),
    }
    inhibition = activation.get("inhibition")
    if inhibition is not None:
        if not isinstance(inhibition, (int, float)) or not math.isfinite(float(inhibition)) or not 0 <= float(inhibition) <= 1:
            raise PolypackValueError("activation.inhibition must be a finite number in [0, 1]")
        last_inhibited_at = activation.get("lastInhibitedAt")
        if not isinstance(last_inhibited_at, (int, float)) or not math.isfinite(float(last_inhibited_at)) or last_inhibited_at < 0:
            raise PolypackValueError("activation.lastInhibitedAt must be a finite non-negative number when inhibition is set")
        result["inhibition"] = float(inhibition)
        result["lastInhibitedAt"] = int(last_inhibited_at)
    context = activation.get("context")
    if context:
        if not isinstance(context, dict):
            raise PolypackValueError("activation.context must be an object")
        validated_context = {}
        for key, entry in context.items():
            score = entry.get("score") if isinstance(entry, dict) else None
            if not isinstance(score, (int, float)) or not math.isfinite(float(score)) or not 0 <= float(score) <= 1:
                raise PolypackValueError(f"activation.context[{key}].score must be a finite number in [0, 1]")
            entry_anchor = entry.get("lastMeaningfulActivation")
            if not isinstance(entry_anchor, (int, float)) or not math.isfinite(float(entry_anchor)) or entry_anchor < 0:
                raise PolypackValueError(f"activation.context[{key}].lastMeaningfulActivation must be a finite non-negative number")
            validated_context[key] = {"score": float(score), "lastMeaningfulActivation": int(entry_anchor)}
        result["context"] = validated_context
    return result


MEMORY_CLASSES = ("episodic", "semantic", "procedural", "entity")

# Provenance/memory-class fields settable directly via `patch_node` (not nested under `data.`).
TOP_LEVEL_PATCHABLE_FIELDS = ("memoryClass", "confidence", "source", "observedAt", "derivedFrom", "supersedes", "contradicts")

# Edge type used by `PolyGraph.supersede` for the contradiction axis.
SUPERSEDED_BY_EDGE = "SUPERSEDED_BY"

# Edge type used by `PolyGraph.consolidate` linking a consolidated node to its sources.
CONSOLIDATED_FROM_EDGE = "CONSOLIDATED_FROM"

# Default per-class score/importance half-lives. Episodic memories fade
# fastest unless reinforced; semantic/procedural facts are far more durable;
# entities barely decay. Only used for nodes whose resolved memoryClass (see
# ActivationEngine.resolve_half_lives) has no override in `classHalfLives`.
DEFAULT_CLASS_HALF_LIVES = {
    "episodic": {"scoreHalfLifeMs": 12 * _HOUR, "importanceHalfLifeMs": 7 * _DAY},
    "semantic": {"scoreHalfLifeMs": 7 * _DAY, "importanceHalfLifeMs": 90 * _DAY},
    "procedural": {"scoreHalfLifeMs": 7 * _DAY, "importanceHalfLifeMs": 60 * _DAY},
    "entity": {"scoreHalfLifeMs": 30 * _DAY, "importanceHalfLifeMs": float("inf")},
}


def _validate_provenance(node: dict) -> dict:
    """Validate memory-class and confidence/provenance fields. Node-level
    metadata, not activation state, so this is a sibling to `_validate_activation`,
    not an extension of it. `derivedFrom`/`supersedes`/`contradicts` are soft
    references — the referenced ids are not required to exist."""
    result: dict = {}
    memory_class = node.get("memoryClass")
    if memory_class is not None:
        if memory_class not in MEMORY_CLASSES:
            raise PolypackValueError(f"memoryClass must be one of {', '.join(MEMORY_CLASSES)}")
        result["memoryClass"] = memory_class
    confidence = node.get("confidence")
    if confidence is not None:
        if not isinstance(confidence, (int, float)) or not math.isfinite(float(confidence)) or not 0 <= float(confidence) <= 1:
            raise PolypackValueError("confidence must be a finite number in [0, 1]")
        result["confidence"] = float(confidence)
    observed_at = node.get("observedAt")
    if observed_at is not None:
        if not isinstance(observed_at, (int, float)) or not math.isfinite(float(observed_at)) or observed_at < 0:
            raise PolypackValueError("observedAt must be a finite non-negative number")
        result["observedAt"] = int(observed_at)
    source = node.get("source")
    if source is not None:
        if source == "":
            raise PolypackValueError("source must not be empty")
        result["source"] = source
    supersedes = node.get("supersedes")
    if supersedes is not None:
        if supersedes == "":
            raise PolypackValueError("supersedes must not be empty")
        result["supersedes"] = supersedes
    for field in ("derivedFrom", "contradicts"):
        value = node.get(field)
        if value is not None:
            if any(id_ == "" for id_ in value):
                raise PolypackValueError(f"{field} must contain only non-empty strings")
            result[field] = list(value)
    return result


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


def _lock_store_file(lock_file: Any, read_only: bool) -> None:
    """Acquire the platform-specific lock for a directory-backed store."""
    if os.name == "nt":
        # msvcrt.locking requires a byte to exist and locks from the current
        # file position. Windows has no shared-lock equivalent in msvcrt, so
        # read-only opens use a non-blocking exclusive lock as well.
        lock_file.seek(0)
        if lock_file.read(1) == b"":
            lock_file.seek(0)
            lock_file.write(b"\0")
            lock_file.flush()
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_NBLCK, 1)
    else:
        fcntl.flock(lock_file.fileno(), (fcntl.LOCK_SH if read_only else fcntl.LOCK_EX) | fcntl.LOCK_NB)


def _unlock_store_file(lock_file: Any) -> None:
    if os.name == "nt":
        lock_file.seek(0)
        msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
    else:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


class DirectoryStorage:
    """Host byte-storage adapter over a directory (snapshot.msgpack / wal.msgpack)."""

    def __init__(self, directory: str, read_only: bool = False) -> None:
        self._dir = Path(directory)
        self._dir.mkdir(parents=True, exist_ok=True)
        self._read_only = read_only
        self._lock_file = open(self._dir / "store.lock", "a+b")
        try:
            _lock_store_file(self._lock_file, read_only)
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
            _unlock_store_file(self._lock_file)
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
            "changeFeed": True,
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
        # intact instead of a torn file. Fsyncing the tmp file before the
        # rename is the Rust `Store`'s call via the explicit `sync()`/
        # `sync_dir()` it issues only under `Durability::Fsync` — doing it
        # unconditionally here silently upgraded `Durability::Process`
        # ("written to the OS, not fsynced") into an always-fsync adapter.
        target = self._path(name)
        tmp_path = self._dir / f"{name}.tmp"
        with open(tmp_path, "wb") as f:
            f.write(bytes(data))
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

    def sync(self, name: str) -> None:
        # Invoked by the Rust `Store` only under `Durability::Fsync`. Mirrors
        # the Rust `FileStorage::sync`: fsync the named file so its contents
        # survive a power loss, not just a process crash.
        try:
            fd = os.open(self._path(name), os.O_RDONLY)
        except FileNotFoundError:
            return
        try:
            os.fsync(fd)
        finally:
            os.close(fd)

    def sync_dir(self) -> None:
        # Best-effort directory fsync so a rename (see `write`) is durable
        # too; unsupported on some platforms (e.g. Windows), same as Rust's
        # `FileStorage::sync_dir`.
        try:
            fd = os.open(self._dir, os.O_RDONLY)
        except OSError:
            return
        try:
            os.fsync(fd)
        except OSError:
            pass
        finally:
            os.close(fd)

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
    for field in ("memoryClass", "confidence", "source", "observedAt", "supersedes"):
        if node.get(field) is not None:
            out[field] = node[field]
    for field in ("derivedFrom", "contradicts"):
        if node.get(field) is not None:
            out[field] = list(node[field])
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
            "dirty_nodes": set(graph._dirty_node_ids),
            "dirty_edges": set(graph._dirty_edge_ids),
            "dirty_vectors": set(graph._dirty_vector_ids),
            "vectors": list(graph.vectors.entries()),
            "dirty": graph._dirty,
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
        self.graph._dirty_node_ids = self._snapshot["dirty_nodes"]
        self.graph._dirty_edge_ids = self._snapshot["dirty_edges"]
        self.graph._dirty_vector_ids = self._snapshot["dirty_vectors"]
        self.graph._dirty = self._snapshot["dirty"]
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
        self._dirty_node_ids: set = set()
        self._dirty_edge_ids: set = set()
        self._dirty_vector_ids: set = set()
        self._dirty = False
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
            "changeFeed": True,
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
                raise AdapterCapabilityError(name)

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
        memory_class: Optional[str] = None,
    ) -> None:
        if self._store_read_only:
            raise PolypackStorageError("store was opened read-only")
        _validate_id(node_type, "node type")
        required = tuple(required_fields or ())
        types = dict(data_types or {})
        if len(set(required)) != len(required) or any(not isinstance(field, str) or not field for field in required):
            raise PolypackValueError("required fields must be unique and non-empty")
        if any(not isinstance(field, str) or not field or expected not in {"string", "number", "integer", "boolean", "object", "array"} for field, expected in types.items()):
            raise PolypackValueError("invalid node data type definition")
        if memory_class is not None and memory_class not in MEMORY_CLASSES:
            raise PolypackValueError(f"memoryClass must be one of {', '.join(MEMORY_CLASSES)}")
        previous = self._node_type_definitions.get(node_type)
        self._node_type_definitions[node_type] = {
            "validate": validate,
            "requiredFields": required,
            "dataTypes": types,
            "memoryClass": memory_class,
        }
        try:
            for node in self._nodes.values():
                if node["type"] == node_type:
                    self._validate_node_schema(node)
            self._persist_schema_metadata()
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

    def _record_query(self, duration_ms: float, scanned_records: int, index: Optional[str], indexes: Optional[Iterable[str]] = None) -> None:
        self._query_count += 1
        self._query_duration_ms += duration_ms
        self._query_scanned_records += scanned_records
        selected = list(indexes or (() if index is None else (index,)))
        for name in selected:
            self._query_index_usage[name] = self._query_index_usage.get(name, 0) + 1

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

    def _persist_schema_metadata(self) -> None:
        if self._store_directory is None:
            return
        if self._store_read_only:
            raise PolypackStorageError("store was opened read-only")
        metadata = {
            "nodeTypes": [
                {
                    "nodeType": name,
                    "requiredFields": list(definition["requiredFields"]),
                    "dataTypes": dict(definition["dataTypes"]),
                    "memoryClass": definition.get("memoryClass"),
                }
                for name, definition in self._node_type_definitions.items()
            ],
            "edgeTypes": [
                {
                    "edgeType": name,
                    "sourceTypes": sorted(definition["sourceTypes"]),
                    "targetTypes": sorted(definition["targetTypes"]),
                    "cardinality": definition["cardinality"],
                    "requiredFields": list(definition["requiredFields"]),
                    "dataTypes": dict(definition["dataTypes"]),
                }
                for name, definition in self._edge_type_definitions.items()
            ],
        }
        target = self._store_directory / "schemas.json"
        temporary = self._store_directory / "schemas.json.tmp"
        temporary.write_text(json.dumps(metadata, sort_keys=True), encoding="utf-8")
        os.replace(temporary, target)

    def _load_schema_metadata(self) -> None:
        if self._store_directory is None:
            return
        source = self._store_directory / "schemas.json"
        if not source.exists():
            return
        try:
            metadata = json.loads(source.read_text(encoding="utf-8"))
            if not isinstance(metadata, dict) or not isinstance(metadata.get("nodeTypes"), list) or not isinstance(metadata.get("edgeTypes"), list):
                raise ValueError("schema metadata must contain nodeTypes and edgeTypes arrays")
            valid_types = {"string", "number", "integer", "boolean", "object", "array"}
            nodes: dict[str, dict] = {}
            edges: dict[str, dict] = {}
            for definition in metadata["nodeTypes"]:
                if not isinstance(definition, dict):
                    raise ValueError("node type definition must be an object")
                name = definition.get("nodeType")
                required = definition.get("requiredFields", [])
                types = definition.get("dataTypes", {})
                memory_class = definition.get("memoryClass")
                if not isinstance(name, str) or not name or name in nodes or not isinstance(required, list) or len(set(required)) != len(required) or any(not isinstance(field, str) or not field for field in required) or not isinstance(types, dict) or any(not isinstance(field, str) or not field or expected not in valid_types for field, expected in types.items()) or (memory_class is not None and memory_class not in MEMORY_CLASSES):
                    raise ValueError("invalid node type definition")
                nodes[name] = {"validate": None, "requiredFields": tuple(required), "dataTypes": dict(types), "memoryClass": memory_class}
            for definition in metadata["edgeTypes"]:
                if not isinstance(definition, dict):
                    raise ValueError("edge type definition must be an object")
                name = definition.get("edgeType")
                required = definition.get("requiredFields", [])
                types = definition.get("dataTypes", {})
                cardinality = definition.get("cardinality")
                if not isinstance(name, str) or not name or name in edges or not isinstance(required, list) or len(set(required)) != len(required) or any(not isinstance(field, str) or not field for field in required) or not isinstance(types, dict) or any(not isinstance(field, str) or not field or expected not in valid_types for field, expected in types.items()) or cardinality not in (None, "one-to-one", "one-to-many", "many-to-one", "many-to-many"):
                    raise ValueError("invalid edge type definition")
                source_types = definition.get("sourceTypes", [])
                target_types = definition.get("targetTypes", [])
                if not isinstance(source_types, list) or not isinstance(target_types, list) or any(not isinstance(value, str) or not value for value in source_types + target_types):
                    raise ValueError("invalid edge type definition")
                edges[name] = {"sourceTypes": frozenset(source_types), "targetTypes": frozenset(target_types), "validate": None, "cardinality": cardinality, "requiredFields": tuple(required), "dataTypes": dict(types)}
            self._node_type_definitions = nodes
            self._edge_type_definitions = edges
        except (OSError, ValueError, TypeError) as exc:
            raise PolypackStorageError(f"invalid schema metadata: {exc}") from exc

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
        if self._store_read_only:
            raise PolypackStorageError("store was opened read-only")
        _validate_id(edge_type, "edge type")
        if cardinality not in (None, "one-to-one", "one-to-many", "many-to-one", "many-to-many"):
            raise PolypackValueError("invalid edge cardinality")
        required = tuple(required_fields or ())
        types = dict(data_types or {})
        if len(set(required)) != len(required) or any(not isinstance(field, str) or not field for field in required):
            raise PolypackValueError("required fields must be unique and non-empty")
        if any(not isinstance(field, str) or not field or expected not in {"string", "number", "integer", "boolean", "object", "array"} for field, expected in types.items()):
            raise PolypackValueError("invalid edge data type definition")
        previous = self._edge_type_definitions.get(edge_type)
        self._edge_type_definitions[edge_type] = {
            "sourceTypes": frozenset(source_types or ()),
            "targetTypes": frozenset(target_types or ()),
            "validate": validate,
            "cardinality": cardinality,
            "requiredFields": required,
            "dataTypes": types,
        }
        try:
            for edges in self._edges.values():
                for edge in edges.values():
                    if edge["type"] == edge_type:
                        self._validate_edge_schema(edge)
            self._persist_schema_metadata()
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

    def _validate_node_resource_limits(self, node: Node) -> None:
        max_dimensions = self._resource_limits.get("maxVectorDimensions")
        vector = node.get("vector")
        if max_dimensions is not None and vector is not None and len(vector) > max_dimensions:
            raise ResourceLimitError("maxVectorDimensions", max_dimensions)
        max_payload = self._resource_limits.get("maxNodePayloadBytes")
        if max_payload is not None:
            payload = json.dumps(node.get("data") or {}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            if len(payload) > max_payload:
                raise ResourceLimitError("maxNodePayloadBytes", max_payload)

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
        stored.update(_validate_provenance(node))
        if node.get("vector") is not None:
            stored["vector"] = _validate_vector(node["vector"])
        else:
            stored["vector"] = None
        self._validate_node_resource_limits(stored)
        self._validate_node_schema(stored)
        self._validate_index_candidate(stored)
        if previous is not None:
            self._remove_secondary_index_entry(previous)
        if stored["vector"] is None:
            self.vectors.remove(stored["id"])
        self._nodes[stored["id"]] = stored
        self._dirty_node_ids.add(stored["id"])
        self._dirty = True
        self._removed_node_ids.discard(stored["id"])
        if stored["vector"] is not None:
            self.vectors.add(stored["id"], stored["vector"])
            self._dirty_vector_ids.add(stored["id"])
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
        self._validate_node_resource_limits(candidate)
        self._validate_node_schema(candidate)
        _validate_provenance(candidate)
        self._validate_index_candidate(candidate)
        self._remove_secondary_index_entry(node)
        node.clear()
        node.update(candidate)
        self._add_secondary_index_entry(node)
        self._dirty_node_ids.add(id_)
        self._dirty = True
        if vector is not None:
            self.vectors.add(id_, node["vector"])
            self._dirty_vector_ids.add(id_)
            self._removed_vector_ids.discard(id_)
        self._dirty = True
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
            "dirtyRecordCount": len(self._dirty_node_ids) + len(self._dirty_edge_ids) + len(self._dirty_vector_ids) + len(self._removed_node_ids) + len(self._removed_edge_ids) + len(self._removed_vector_ids),
            "pendingPersistence": bool(self._dirty or self._dirty_node_ids or self._dirty_edge_ids or self._dirty_vector_ids or self._removed_node_ids or self._removed_edge_ids or self._removed_vector_ids),
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
        compare_and_set: Optional[dict] = None,
    ) -> Optional[Node]:
        """Apply dotted-path patch operations atomically to a node."""
        node = self._nodes.get(id_)
        if node is None:
            return None
        actual = int(node.get("revision", 0))
        if expected_revision is not None and actual != expected_revision:
            raise ConflictError(f"record {id_} has revision {actual}, expected {expected_revision}")

        candidate = copy.deepcopy(node.get("data") or {})
        candidate_node = _copy_node(node)
        for path, operation in (compare_and_set or {}).items():
            if not isinstance(operation, dict) or "expected" not in operation or "value" not in operation:
                raise PolypackValueError("compare_and_set entries require expected and value")
            if path in TOP_LEVEL_PATCHABLE_FIELDS:
                present = path in candidate_node
                current = candidate_node.get(path)
                if (not present and operation["expected"] is not None) or (present and current != operation["expected"]):
                    raise ConflictError(f"compare-and-set failed for record {id_} at {path}")
                candidate_node[path] = copy.deepcopy(operation["value"])
                continue
            present = _patch_has(candidate, path)
            current = _patch_get(candidate, path)
            if (not present and operation["expected"] is not None) or (present and current != operation["expected"]):
                raise ConflictError(f"compare-and-set failed for record {id_} at {path}")
            _patch_set(candidate, path, operation["value"])
        for path, value in (set or {}).items():
            if path in TOP_LEVEL_PATCHABLE_FIELDS:
                candidate_node[path] = copy.deepcopy(value)
                continue
            _patch_set(candidate, path, value)
        for path in unset or ():
            if path in TOP_LEVEL_PATCHABLE_FIELDS:
                candidate_node.pop(path, None)
                continue
            _patch_unset(candidate, path)
        for path, delta in (increment or {}).items():
            if not isinstance(delta, (int, float)) or not math.isfinite(float(delta)):
                raise PolypackValueError("increment values must be finite numbers")
            if path in TOP_LEVEL_PATCHABLE_FIELDS:
                current = candidate_node.get(path)
                if current is None:
                    current = 0
                if not isinstance(current, (int, float)) or isinstance(current, bool) or not math.isfinite(float(current)):
                    raise PolypackValueError("increment targets must be numeric")
                candidate_node[path] = float(current) + float(delta)
                continue
            current = _patch_get(candidate, path)
            if current is None:
                current = 0
            if not isinstance(current, (int, float)) or isinstance(current, bool) or not math.isfinite(float(current)):
                raise PolypackValueError("increment targets must be numeric")
            _patch_set(candidate, path, float(current) + float(delta))

        candidate_node["data"] = candidate
        candidate_node["updatedAt"] = int(time.time() * 1000)
        candidate_node["revision"] = actual + 1
        self._validate_node_resource_limits(candidate_node)
        self._validate_node_schema(candidate_node)
        candidate_node.update(_validate_provenance(candidate_node))
        node.clear()
        node.update(candidate_node)
        self._dirty_node_ids.add(id_)
        self._dirty = True
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
        self._dirty_node_ids.discard(id_)
        self.vectors.remove(id_)
        self._dirty_vector_ids.discard(id_)
        self._removed_node_ids.add(id_)
        self._removed_vector_ids.add(id_)
        self._dirty = True

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
        self._dirty_edge_ids.add(key)
        incoming = self._incoming.setdefault(target, {})
        incoming[source] = incoming.get(source, 0) + 1
        self._removed_edge_ids.discard(key)
        self._dirty = True

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
            self._dirty_edge_ids.add(id_)
            self._dirty = True
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
            self._dirty_edge_ids.discard(id_)
            self._removed_edge_ids.add(id_)
            self._dirty = True
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
                self._dirty_edge_ids.discard(key)
                self._removed_edge_ids.add(key)
                self._dirty = True
            if self._ownership(e) == "shared" and not self._has_incoming(e["target"], source):
                if self._on_orphan:
                    self._on_orphan(e["target"])
        if not self._edges.get(source):
            self._edges.pop(source, None)

    # ── queries ──

    def snapshot(self) -> "GraphSnapshot":
        """Capture a detached, queryable view of the currently loaded graph."""
        return GraphSnapshot(self)

    def query(self) -> "GraphQuery":
        """Create a mutable `GraphQuery` over the currently loaded nodes."""
        return GraphQuery(self)

    def query_persisted(self) -> "PersistedGraphQuery":
        """Create a storage-level query over persisted nodes.

        Results are filtered, ordered, and paginated inside the native store;
        only the final result page crosses the Python boundary. Pending graph
        mutations are saved before the query so persisted visibility matches
        the rest of the binding's durable API.
        """
        if self._store is None:
            raise PolypackStorageError("no store open; call open_store(path) first")
        if self._dirty:
            if self._store_read_only:
                raise PolypackStorageError("store was opened read-only and has pending changes")
            self.save()
        return PersistedGraphQuery(self)

    def clear(self) -> None:
        """Clear in-memory state only — does not flush pending deletions or touch the attached store."""
        self._nodes.clear()
        self._edges.clear()
        self._incoming.clear()
        self.vectors.clear()
        self._removed_node_ids.clear()
        self._removed_edge_ids.clear()
        self._removed_vector_ids.clear()
        self._dirty_node_ids.clear()
        self._dirty_edge_ids.clear()
        self._dirty_vector_ids.clear()
        self._dirty = False
        if self._store is None:
            self._secondary_index_data = {name: {} for name in self._indexes}

    # ── persistence (Rust storage state machine) ──

    @classmethod
    def open(cls, directory: str, read_only: bool = False, compact_threshold: int = 10_000) -> "PolyGraph":
        """Open a directory-backed binary store and load its graph."""
        graph = cls()
        graph.open_store(directory, read_only=read_only, compact_threshold=compact_threshold)
        return graph

    @classmethod
    def restore(cls, source: str, destination: str) -> "PolyGraph":
        """Restore a directory backup into ``destination`` and open it."""
        source_path = Path(source)
        destination_path = Path(destination)
        if not source_path.is_dir():
            raise PolypackStorageError(f"backup source does not exist: {source}")
        destination_path.mkdir(parents=True, exist_ok=True)
        for name in ("snapshot.msgpack", "wal.msgpack", "mutations.jsonl", "indexes.json", "schemas.json"):
            source_file = source_path / name
            if source_file.exists():
                shutil.copy2(source_file, destination_path / name)
        return cls.open(str(destination_path))

    def open_store(self, directory: str, read_only: bool = False, compact_threshold: int = 10_000) -> None:
        """Attach a directory-backed store and load any existing state."""
        if not isinstance(compact_threshold, int) or compact_threshold < 1:
            raise PolypackValueError("compact_threshold must be a positive integer")
        had_pending_state = bool(
            self._dirty
            or self._nodes
            or self._edges
            or list(self.vectors.entries())
            or self._removed_node_ids
            or self._removed_edge_ids
            or self._removed_vector_ids
        )
        pending_dirty_nodes = set(self._dirty_node_ids)
        pending_dirty_edges = set(self._dirty_edge_ids)
        pending_dirty_vectors = set(self._dirty_vector_ids)
        self._dirty_node_ids.clear()
        self._dirty_edge_ids.clear()
        self._dirty_vector_ids.clear()
        self._store = _NativeStore(DirectoryStorage(directory, read_only=read_only), compact_threshold)
        self._store_directory = Path(directory)
        self._store_read_only = read_only
        self._dirty = False
        self._load_index_metadata()
        self._load_schema_metadata()
        self._load_from_store()
        self._dirty_node_ids = pending_dirty_nodes if had_pending_state else set()
        self._dirty_edge_ids = pending_dirty_edges if had_pending_state else set()
        self._dirty_vector_ids = pending_dirty_vectors if had_pending_state else set()
        self._dirty = had_pending_state
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
        for name in ("snapshot.msgpack", "wal.msgpack", "mutations.jsonl", "indexes.json", "schemas.json"):
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
            if not self._store_read_only and self._dirty:
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
        if not self._dirty:
            return
        nodes = [_copy_node(self._nodes[id_]) for id_ in self._dirty_node_ids if id_ in self._nodes]
        edge_by_id = {
            key: edge
            for edge_map in self._edges.values()
            for key, edge in edge_map.items()
        }
        edges = []
        for id_ in self._dirty_edge_ids:
            e = edge_by_id.get(id_)
            if e is not None:
                edges.append({
                    "id": e["id"] if "id" in e else _edge_key(e["source"], e["type"], e["target"]),
                    "source": e["source"],
                    "target": e["target"],
                    "type": e["type"],
                    "data": dict(e.get("data") or {}),
                    "createdAt": e.get("createdAt", 0),
                    "revision": int(e.get("revision", 0)),
                })
        vector_by_id = dict(self.vectors.entries())
        # `graph.vectors` is public and recovery/conformance callers may add
        # vectors directly. If a node load tentatively marked a vector for
        # deletion but the current index contains it, the current index wins.
        for id_ in tuple(self._removed_vector_ids):
            if id_ in vector_by_id:
                self._dirty_vector_ids.add(id_)
                self._removed_vector_ids.discard(id_)
        vectors = [{"id": id_, "vector": vector_by_id[id_]} for id_ in self._dirty_vector_ids if id_ in vector_by_id]
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
        self._dirty_node_ids.clear()
        self._dirty_edge_ids.clear()
        self._dirty_vector_ids.clear()
        self._dirty = False

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

    def reinforce_node(self, id_: str, amount: float, reason: Optional[str] = None, context: Optional[str] = None) -> Optional[Node]:
        """Apply a durable reinforcement delta. The prior state is decay-corrected
        to now, `amount` is added to `score`, a fraction folds into `importance`,
        the counter increments, and the decay anchor re-sets to now. When
        `context` is given, the same delta additionally reinforces
        `activation.context[context]` — an independently-decaying, additional
        lens, not a replacement for the global score. Returns the updated node,
        or `None` if the node isn't loaded. (Python has no change events, so no
        notification is emitted.)"""
        if not isinstance(amount, (int, float)) or not math.isfinite(float(amount)):
            raise PolypackValueError("reinforcement amount must be finite")
        node = self._nodes.get(id_)
        if node is None:
            return None
        now = int(time.time() * 1000)
        node["activation"] = _reinforce_activation(node.get("activation"), float(amount), now, context)
        node["updatedAt"] = now
        self._dirty_node_ids.add(id_)
        self._dirty = True
        return _copy_node(node)

    def reinforce_node_safe(self, id_: str, amount: float, reason: Optional[str] = None, context: Optional[str] = None) -> Optional[Node]:
        """Alias for `reinforce_node` — the Python graph has no hot cache or
        eviction, so there is nothing to restore."""
        return self.reinforce_node(id_, amount, reason, context)

    def suppress_node(self, id_: str, amount: float, reason: Optional[str] = None) -> Optional[Node]:
        """Apply a durable suppression delta to a node's `inhibition` (mirrors
        `reinforce_node` but for the inhibition axis, which decays on its own,
        shorter-by-default half-life and is subtracted from `score` only at the
        final read/ranking step). A negative `amount` releases suppression.
        Returns the updated node, or `None` if the node isn't loaded."""
        if not isinstance(amount, (int, float)) or not math.isfinite(float(amount)):
            raise PolypackValueError("suppression amount must be finite")
        node = self._nodes.get(id_)
        if node is None:
            return None
        now = int(time.time() * 1000)
        node["activation"] = _suppress_activation(node.get("activation"), float(amount), now)
        node["updatedAt"] = now
        self._dirty_node_ids.add(id_)
        self._dirty = True
        return _copy_node(node)

    def suppress_node_safe(self, id_: str, amount: float, reason: Optional[str] = None) -> Optional[Node]:
        """Alias for `suppress_node` — the Python graph has no hot cache or
        eviction, so there is nothing to restore."""
        return self.suppress_node(id_, amount, reason)

    def supersede(self, id_: str, superseded_id: str, amount: float = 1.0, reason: Optional[str] = "superseded") -> Optional[Node]:
        """Mark `id_` as superseding `superseded_id`: records `id_.supersedes =
        superseded_id`, adds a `SUPERSEDED_BY` edge from `id_` to
        `superseded_id` (ownership "reference" — deleting either node never
        cascades to the other), and suppresses the superseded node (see
        `suppress_node`) so retrieval prefers the newer node without deleting
        the old one. Both the historical relationship and the stale node
        remain in the graph — this is "contradiction", not deletion. Returns
        `None` if either node isn't loaded."""
        if id_ not in self._nodes or superseded_id not in self._nodes:
            return None
        self.patch_node(id_, set={"supersedes": superseded_id})
        self.add_edge(id_, SUPERSEDED_BY_EDGE, superseded_id, ownership="reference")
        self.suppress_node(superseded_id, amount, reason)
        return self.get_node(id_)

    def consolidate(
        self,
        node: dict,
        source_ids: Sequence[str],
        amount: float = 1.0,
        reason: Optional[str] = "consolidated",
    ) -> Optional[Node]:
        """Consolidate `source_ids` into `node`: writes `node` via `add_node`
        (insert-or-replace — pass an existing id to extend a previous
        consolidation), merging `source_ids` into its `derivedFrom`
        (deduplicated against both the caller-supplied value and any
        already-stored node, not overwritten — re-consolidating the same node
        as more evidence accumulates is a normal use), adds a
        `CONSOLIDATED_FROM` edge from `node` to each source (ownership
        "reference" — no cascade delete either way), and suppresses each
        source (see `suppress_node`) so retrieval prefers the consolidated
        node without deleting the sources. Polypack only provides this
        mechanism — `node`'s content (including `memoryClass`, which is never
        forced) and which sources belong together are entirely caller-decided
        policy. Returns `None` (writing nothing) if any source isn't loaded,
        or raises if `source_ids` is empty."""
        if not source_ids:
            raise PolypackValueError("consolidate requires at least one source id")
        if any(sid not in self._nodes for sid in source_ids):
            return None
        node_id = node["id"]
        existing = self._nodes.get(node_id, {}).get("derivedFrom") or node.get("derivedFrom") or []
        derived_from = list(dict.fromkeys([*existing, *source_ids]))
        node = {**node, "derivedFrom": derived_from}
        self.add_node(node)
        for sid in source_ids:
            self.add_edge(node_id, CONSOLIDATED_FROM_EDGE, sid, ownership="reference")
            self.suppress_node(sid, amount, reason)
        return self.get_node(node_id)

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

    def get_context_activation(self, id_: str, context: str) -> float:
        """Decay-corrected score of a node within one named context, or 0 when
        the node has no history in that context. Unlike `get_activation`, this
        never falls back to the global score."""
        state = self.get_activation_state(id_)
        if not state:
            return 0.0
        return (state.get("context") or {}).get(context, {}).get("score", 0.0)

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
            self._dirty_node_ids.add(node["id"])
            self._dirty = True

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
                self._dirty_edge_ids.discard(key)
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
                self._dirty_edge_ids.discard(key)
                self._removed_edge_ids.add(key)


# ── Query builder ──


class PersistedGraphQuery:
    """Fluent query executed directly by the native persisted store.

    This deliberately covers the storage-level query contract. Similarity,
    joins, and traversal remain hot-graph operations until their corresponding
    store indexes can execute without materialising records in Python.
    """

    def __init__(self, graph: PolyGraph) -> None:
        self._graph = graph
        self._node_types: Optional[list[str]] = None
        self._attributes: dict[str, Any] = {}
        self._ranges: dict[str, dict[str, float]] = {}
        self._order: Optional[dict[str, str]] = None
        self._offset: Optional[int] = None
        self._limit: Optional[int] = None

    def where_type(self, *types: str) -> "PersistedGraphQuery":
        self._node_types = list(types)
        return self

    def where(self, field: str, value: Any) -> "PersistedGraphQuery":
        self._attributes[field] = value
        return self

    def where_range(self, field: str, above: Optional[float] = None, below: Optional[float] = None) -> "PersistedGraphQuery":
        if above is not None and not math.isfinite(float(above)):
            raise PolypackValueError("range above must be finite")
        if below is not None and not math.isfinite(float(below)):
            raise PolypackValueError("range below must be finite")
        entry: dict[str, float] = {}
        if above is not None:
            entry["above"] = float(above)
        if below is not None:
            entry["below"] = float(below)
        self._ranges[field] = entry
        return self

    def order_by(self, field: str, direction: str = "asc") -> "PersistedGraphQuery":
        if direction not in ("asc", "desc"):
            raise PolypackValueError("direction must be 'asc' or 'desc'")
        self._order = {"field": field, "direction": direction}
        return self

    def offset(self, n: int) -> "PersistedGraphQuery":
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise PolypackValueError("offset must be a non-negative integer")
        self._offset = n
        return self

    def limit(self, n: int) -> "PersistedGraphQuery":
        if not isinstance(n, int) or isinstance(n, bool) or n < 0:
            raise PolypackValueError("limit must be a non-negative integer")
        self._limit = n
        return self

    def _query(self, paginate: bool = True) -> dict:
        query: dict[str, Any] = {}
        if self._node_types is not None:
            query["nodeTypes"] = list(self._node_types)
        if self._attributes:
            query["attributes"] = dict(self._attributes)
        if self._ranges:
            query["attributeRanges"] = copy.deepcopy(self._ranges)
        if self._order is not None:
            query["orderBy"] = dict(self._order)
        if paginate:
            if self._offset is not None:
                query["offset"] = self._offset
            if self._limit is not None:
                query["limit"] = self._limit
        return query

    def to_list(self) -> list:
        return self._graph._store.query_nodes(self._query())

    def ids(self) -> list[str]:
        return [node["id"] for node in self.to_list()]

    def count(self) -> int:
        return self._graph._store.count_nodes(self._query(paginate=False))


class GraphSnapshot:
    """Read-only graph state captured at one point in time.

    A snapshot copies nodes, edges, and index buckets so later graph writes do
    not affect queries created from it. It intentionally exposes only reads
    and query construction; mutations remain methods on :class:`PolyGraph`.
    """

    def __init__(self, graph: PolyGraph) -> None:
        self._nodes = copy.deepcopy(graph._nodes)
        self._edges = copy.deepcopy(graph._edges)
        self._incoming = copy.deepcopy(graph._incoming)
        self._indexes = copy.deepcopy(graph._indexes)
        self._secondary_index_data = copy.deepcopy(graph._secondary_index_data)
        self._resource_limits = dict(graph._resource_limits)

    @property
    def resource_limit_config(self) -> dict:
        return dict(self._resource_limits)

    def _record_query(self, duration_ms: float, scanned_records: int, index: Optional[str], indexes: Optional[Iterable[str]] = None) -> None:
        # Query metrics belong to the live graph, not to this detached view.
        return None

    def query(self) -> "GraphQuery":
        return GraphQuery(self)

    def get_node(self, id_: str) -> Optional[Node]:
        node = self._nodes.get(id_)
        return None if node is None else _copy_node(node)

    def get_nodes(self, ids: Optional[Iterable[str]] = None) -> list:
        selected = self._nodes.keys() if ids is None else ids
        return [self._copy_node_by_id(id_) for id_ in selected if id_ in self._nodes]

    def _copy_node_by_id(self, id_: str) -> Node:
        return _copy_node(self._nodes[id_])

    def get_edges(self, source: Optional[str] = None, edge_type: Optional[str] = None) -> list:
        maps = self._edges.items() if source is None else ((source, self._edges.get(source, {})),)
        result = []
        for _, edges in maps:
            for edge in edges.values():
                if edge_type is None or edge["type"] == edge_type:
                    result.append(dict(edge, data=dict(edge.get("data") or {})))
        return result


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
        selected = self._selected_indexes()
        stages = [*(f"property-index({definition['name']})" for definition in selected)] or ["record-scan"]
        if len(selected) > 1:
            stages.append(f"index-intersection({len(selected)})")
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
            "index": selected[0]["name"] if selected else None,
            "indexes": [definition["name"] for definition in selected],
            "stages": stages,
            "loadedRecords": loaded,
            "estimatedCost": max(1, loaded * (0.25 / len(selected) if selected else 1)),
        }

    def _selected_indexes(self) -> list[dict]:
        equality_fields = {field for op, field, _ in self._attributes if op == "eq"}
        range_fields = {field for op, field, _ in self._attributes if op == "range"}
        node_type = self._node_types[0] if self._node_types and len(self._node_types) == 1 else None
        selected = []
        for definition in self._graph._indexes.values():
            if definition["nodeType"] and definition["nodeType"] != node_type:
                continue
            if all(field in equality_fields or field.removeprefix("data.") in equality_fields for field in definition["fields"]):
                selected.append(definition)
                continue
            if len(definition["fields"]) == 1 and definition["fields"][0] in range_fields:
                selected.append(definition)
        return selected

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

    def _native_ids(self, plan: dict, candidates: Optional[list[Node]] = None) -> Optional[list]:
        """Run expensive plans through Rust; return None for hot simple plans."""
        if self._activation_above is not None or self._activation_order is not None:
            # The native executor doesn't understand activation filters — force
            # the pure-Python pipeline so results can't silently diverge.
            return None
        # Crossing the Python/Rust boundary costs more than the local pipeline
        # for ordinary filters/order/pagination. Native execution is reserved
        # for similarity, joins, and traversal where the Rust executor can
        # amortise the conversion cost over substantially more work.
        if self._similarity is None and not self._joins and not self._traversal:
            return None
        try:
            nodes_source = candidates if candidates is not None and not self._joins and not self._traversal else self._graph._nodes.values()
            needs_vectors = self._similarity is not None
            nodes = [dict(n) if needs_vectors else {key: value for key, value in n.items() if key != "vector"} for n in nodes_source]
            edges = []
            if self._joins or self._traversal:
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
        candidates = None if self._joins or self._traversal else self._indexed_candidates()
        scanned_records = len(self._graph._nodes) if candidates is None else len(candidates)
        native_ids = self._native_ids(plan, candidates)
        if native_ids is not None:
            maximum = self._limits.get("maxNodesVisited")
            if maximum is not None and len(native_ids) > maximum:
                raise ResourceLimitError("maxNodesVisited", maximum)
            by_id = {n["id"]: n for n in self._graph._nodes.values()}
            results = [by_id[i] for i in native_ids if i in by_id]
            maximum = self._limits.get("maxResults")
            if maximum is not None and len(results) > maximum:
                raise ResourceLimitError("maxResults", maximum)
            explanation = self.explain()
            self._graph._record_query((time.perf_counter() - started) * 1000, scanned_records, explanation["index"], explanation["indexes"])
            return results
        # Fallback: pure-Python pipeline (only reached if the native path
        # failed, e.g. a non-JSON-serialisable filter value).
        results = self._collect_python(plan, candidates)
        explanation = self.explain()
        self._graph._record_query((time.perf_counter() - started) * 1000, scanned_records, explanation["index"], explanation["indexes"])
        return results

    def _collect_python(self, plan: dict, candidates: Optional[list[Node]] = None) -> list:
        if candidates is None:
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
        selected_indexes = self._selected_indexes()
        if not selected_indexes:
            return list(self._graph._nodes.values())
        equalities = {field: value for op, field, value in self._attributes if op == "eq"}
        candidate_sets = []
        for definition in selected_indexes:
            buckets = self._graph._secondary_index_data.get(definition["name"], {})
            values = [equalities.get(field, equalities.get(field.removeprefix("data."))) for field in definition["fields"]]
            if all(field in equalities or field.removeprefix("data.") in equalities for field in definition["fields"]):
                candidate_sets.append(set(buckets.get(repr(values), set())))
                continue
            field = definition["fields"][0]
            ranges = [(above, below) for op, name, (above, below) in self._attributes if op == "range" and name == field]
            if len(definition["fields"]) == 1 and ranges:
                above, below = ranges[-1]
                ids = set()
                for encoded, bucket in buckets.items():
                    try:
                        value = ast.literal_eval(encoded)[0]
                    except (ValueError, SyntaxError, IndexError, TypeError):
                        continue
                    if isinstance(value, (int, float)) and not isinstance(value, bool) and (above is None or value > above) and (below is None or value < below):
                        ids.update(bucket)
                candidate_sets.append(ids)
        if not candidate_sets:
            return list(self._graph._nodes.values())
        ids = set.intersection(*candidate_sets)
        return [self._graph._nodes[id_] for id_ in ids if id_ in self._graph._nodes]

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
            "classHalfLives": DEFAULT_CLASS_HALF_LIVES,
        }
        cfg.update(config or {})
        weights = dict(cfg["weights"])
        cfg["weights"] = weights
        self.config = cfg
        self.attention: dict = {}
        # Per-node signal breakdown from the most recent `pulse` scoring, consumed by `record_feedback`.
        self._last_signals: dict = {}

    def dispose(self) -> None:
        """Drop transient attention. Durable state is untouched."""
        self.attention.clear()
        self._last_signals.clear()

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

    def resolve_half_lives(self, node: dict) -> tuple[float, float]:
        """Resolve a node's effective score/importance half-lives: `node["memoryClass"]`
        if set, else the owning type's registered default, else no class — the
        flat `config["scoreHalfLifeMs"]`/`config["importanceHalfLifeMs"]`
        defaults, unaffected by `classHalfLives`."""
        memory_class = node.get("memoryClass")
        if memory_class is None:
            definition = self.graph._node_type_definitions.get(node.get("type"))
            memory_class = definition.get("memoryClass") if definition else None
        override = self.config["classHalfLives"].get(memory_class) if memory_class else None
        score_half_life_ms = (override or {}).get("scoreHalfLifeMs", self.config["scoreHalfLifeMs"])
        importance_half_life_ms = (override or {}).get("importanceHalfLifeMs", self.config["importanceHalfLifeMs"])
        return score_half_life_ms, importance_half_life_ms

    def effective(self, id_: str, context: Optional[str] = None) -> float:
        """Durable decayed score (using the node's resolved memory-class
        half-lives when it has one, else the flat config defaults), or the
        context-scoped score when `context` is given, plus transient
        attention, minus decayed inhibition. Inhibition is applied only here,
        never inside `pulse`'s composite, so a suppressed node stays
        re-evaluable."""
        node = self.graph.get_node(id_)
        if node is None or node.get("activation") is None:
            return self.attention_of(id_)
        # Decay the raw stored record exactly once, with the resolved
        # half-life — NOT `graph.get_activation_state`, which already
        # decay-corrects with the flat default and would double-decay a
        # class-resolved half-life.
        score_half_life_ms, importance_half_life_ms = self.resolve_half_lives(node)
        decayed = _decay_activation_state(
            node["activation"], int(time.time() * 1000),
            score_half_life_ms=score_half_life_ms, importance_half_life_ms=importance_half_life_ms,
        )
        base = (decayed.get("context") or {}).get(context, {}).get("score", 0.0) if context is not None else decayed["score"]
        inhibition = decayed.get("inhibition") or 0.0
        return _clamp01(base + self.attention_of(id_) - inhibition)

    def inhibition_of(self, id_: str) -> float:
        """Current decayed inhibition for `id_` (0 when none)."""
        state = self.graph.get_activation_state(id_)
        return (state or {}).get("inhibition") or 0.0

    # ── durable reinforcement ──

    def reinforce(self, id_: str, amount: float, reason: Optional[str] = None, context: Optional[str] = None) -> Optional[Node]:
        return self.graph.reinforce_node(id_, amount, reason, context)

    def reinforce_all(self, entries: Iterable[tuple]) -> None:
        for entry in entries:
            id_, amount, reason = entry[0], entry[1], entry[2] if len(entry) > 2 else None
            context = entry[3] if len(entry) > 3 else None
            self.graph.reinforce_node(id_, amount, reason, context)

    def suppress(self, id_: str, amount: float, reason: Optional[str] = None) -> Optional[Node]:
        """Suppress a node's durable `inhibition`. A negative `amount` releases
        suppression."""
        return self.graph.suppress_node(id_, amount, reason)

    # ── learned weights ──

    def record_feedback(self, id_: str, was_useful: bool, learning_rate: float = 0.05) -> None:
        """Record whether `id_` — previously scored by `pulse` — turned out
        useful, nudging the composite `weights` (used by `pulse`'s scoring)
        toward whichever signal was strongest for it: each weight moves by
        `learning_rate * direction * signal`, where `direction` is +1 for
        useful and -1 for not, so a signal that was high for a useful node
        gets reinforced and a signal that was high for a useless one gets
        discounted. Weights are clamped to stay non-negative. A no-op if
        `id_` has no cached signal breakdown (i.e. wasn't scored by a `pulse`
        call since it was last cleared/scored) — this is a simple
        exponential-moving-average-style nudge, not a full online learner.
        Weights are in-memory only — not persisted or synced."""
        signals = self._last_signals.get(id_)
        if signals is None:
            return
        direction = 1.0 if was_useful else -1.0
        weights = self.config["weights"]
        for key in ("semantic", "graph", "recency", "usage"):
            weights[key] = max(0.0, weights[key] + learning_rate * direction * signals[key])

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
            self._last_signals[id_] = {"semantic": s, "graph": g, "recency": recency, "usage": usage}
            scores[id_] = w["semantic"] * s + w["graph"] * g + w["recency"] * recency + w["usage"] * usage
        return sorted(
            ((id_, score) for id_, score in scores.items() if score > threshold),
            key=lambda x: x[1],
            reverse=True,
        )

    def absorb(self, vector: Any, context: Optional[str] = None, **options) -> list:
        """`pulse` plus reinforcement: nodes whose composite clears
        `absorbThreshold` receive durable reinforcement of `absorbGain * score`.
        When `context` is given, the same reinforcement additionally reinforces
        that context on every absorbed node."""
        scores = self.pulse(vector, **options)
        for id_, score in scores:
            if score >= self.config["absorbThreshold"]:
                self.graph.reinforce_node(id_, _clamp01(self.config["absorbGain"] * score), "pulse", context)
        return scores

    # ── working memory ──

    def working_memory(
        self,
        limit: int = 10,
        min_score: float = 0.0,
        context: Optional[str] = None,
        token_budget: Optional[float] = None,
        cost_of: Optional[Callable[[Node], float]] = None,
        diversity_lambda: float = 0.0,
        similarity_of: Optional[Callable[[Node, Node], float]] = None,
    ) -> list:
        """The current "mental state": loaded nodes ranked by `effective`
        activation (durable decayed score + transient attention, minus
        inhibition) descending, top `limit`.

        Passing `token_budget` and/or `diversity_lambda` enables a budgeted,
        diversity-aware selection — a memory-flavoured maximal-marginal-relevance
        pass suited to LLM context assembly: greedily picks the highest
        `relevance - diversity_lambda * similarity-to-already-selected`
        candidate under a token budget, so the result isn't many
        near-duplicate highly-activated neighbours."""
        cost_of = cost_of or (lambda _n: 1.0)
        similarity_of = similarity_of or _default_node_similarity

        candidates = []
        for node in self.graph._nodes.values():
            score = self.effective(node["id"], context)
            if score > min_score:
                candidates.append((score, _copy_node(node)))
        candidates.sort(key=lambda x: x[0], reverse=True)

        if diversity_lambda <= 0.0 and token_budget is None:
            return [n for _, n in candidates[:limit]]

        pool = candidates[: max(limit * 4, limit)]
        budget = token_budget if token_budget is not None else float("inf")
        remaining = list(pool)
        selected: list = []
        spent = 0.0
        while len(selected) < limit and remaining:
            best_index = 0
            best_mmr = float("-inf")
            for i, (score, node) in enumerate(remaining):
                max_similarity = max((similarity_of(node, s) for s in selected), default=0.0)
                mmr = (1 - diversity_lambda) * score - diversity_lambda * max_similarity if diversity_lambda > 0 else score
                if mmr > best_mmr:
                    best_mmr = mmr
                    best_index = i
            _, chosen = remaining.pop(best_index)
            cost = cost_of(chosen)
            if spent + cost > budget:
                break
            spent += cost
            selected.append(chosen)
        return selected


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
