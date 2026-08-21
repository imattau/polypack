"""Portable sync-envelope validation and checksum helpers."""

from __future__ import annotations

import json
import math
from typing import Any, Iterable


def sync_checksum(operations: Iterable[dict[str, Any]]) -> str:
    """Return the TypeScript-compatible FNV-1a checksum for an ordered batch."""
    encoded = json.dumps(list(operations), ensure_ascii=False, separators=(",", ":"))
    digest = 2166136261
    units = encoded.encode("utf-16-le")
    for index in range(0, len(units), 2):
        digest ^= units[index] | (units[index + 1] << 8)
        digest = (digest * 16777619) & 0xFFFFFFFF
    return f"{digest:08x}"


def sync_identity_checksum(operation_ids: Iterable[str], transaction_ids: Iterable[str]) -> str:
    return sync_checksum([{
        "seq": 0,
        "timestamp": 0,
        "clientId": "sync-identities",
        "kind": "addNode",
        "payload": {"operationIds": sorted(operation_ids), "transactionIds": sorted(transaction_ids)},
    }])


def validate_sync_operation(operation: dict[str, Any]) -> None:
    if not isinstance(operation, dict):
        raise ValueError("sync operation must be an object")
    if not isinstance(operation.get("seq"), int) or isinstance(operation["seq"], bool) or operation["seq"] < 1:
        raise ValueError("sync sequence must be a positive integer")
    if not isinstance(operation.get("timestamp"), (int, float)) or isinstance(operation["timestamp"], bool) or not math.isfinite(operation["timestamp"]):
        raise ValueError("sync timestamp must be finite")
    if not isinstance(operation.get("clientId"), str) or not operation["clientId"]:
        raise ValueError("sync clientId must be non-empty")
    if not isinstance(operation.get("kind"), str) or not operation["kind"]:
        raise ValueError("sync operation kind must be non-empty")
    if not isinstance(operation.get("payload"), dict):
        raise ValueError("sync payload must be an object")
    for field in ("operationId", "transactionId"):
        if field in operation and (not isinstance(operation[field], str) or not operation[field]):
            raise ValueError(f"sync {field} must be non-empty")
    if "baseRevision" in operation and (not isinstance(operation["baseRevision"], int) or isinstance(operation["baseRevision"], bool) or operation["baseRevision"] < 0):
        raise ValueError("sync baseRevision must be a non-negative integer")


def validate_sync_batch(operations: Iterable[dict[str, Any]]) -> str:
    batch = list(operations)
    for operation in batch:
        validate_sync_operation(operation)
    return sync_checksum(batch)
