"""Portable sync-envelope validation and checksum helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path
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


class FileSyncOperationLog:
    """Small durable JSON operation log for the synchronous Python server."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)

    def load(self) -> dict[str, Any]:
        if not self.path.exists():
            return {"baseCursor": 0, "ops": [], "operationIds": [], "transactionIds": []}
        state = json.loads(self.path.read_text())
        if not isinstance(state, dict) or not isinstance(state.get("ops", []), list):
            raise ValueError("invalid sync operation log")
        return state

    def append_batch(self, operations: list[dict[str, Any]], base_cursor: int, operation_ids: set[str], transaction_ids: set[str]) -> None:
        state = {"baseCursor": base_cursor, "ops": operations, "operationIds": sorted(operation_ids), "transactionIds": sorted(transaction_ids)}
        state["checksum"] = sync_checksum(state["ops"])
        state["identityChecksum"] = sync_identity_checksum(state["operationIds"], state["transactionIds"])
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(self.path.suffix + ".tmp")
        temporary.write_text(json.dumps(state, separators=(",", ":")))
        temporary.replace(self.path)


class SyncServer:
    """Transport-neutral synchronous sync server matching the TS protocol."""

    def __init__(self, *, protocol_version: int = 1, max_ops: int | None = None, max_batch_ops: int | None = None,
                 authorize: Any = None, conflict: Any = None, operation_log: FileSyncOperationLog | None = None) -> None:
        if protocol_version < 1 or (max_ops is not None and max_ops < 1) or (max_batch_ops is not None and max_batch_ops < 1):
            raise ValueError("invalid sync server limits")
        self.protocol_version = protocol_version
        self.max_ops = max_ops
        self.max_batch_ops = max_batch_ops
        self.authorize = authorize
        self.conflict = conflict
        self.operation_log = operation_log
        self.base_cursor = 0
        self.ops: list[dict[str, Any]] = []
        self._operation_ids: set[str] = set()
        self._transaction_ids: set[str] = set()
        self._clients: dict[str, tuple[Any, Any, dict[str, Any]]] = {}
        if operation_log is not None:
            state = operation_log.load()
            self.base_cursor = int(state.get("baseCursor", 0))
            self.ops = list(state.get("ops", []))
            self._operation_ids = set(state.get("operationIds", []))
            self._transaction_ids = set(state.get("transactionIds", []))

    @property
    def cursor(self) -> int:
        return self.base_cursor + len(self.ops)

    def add_client(self, client_id: str, send: Any, filter: Any = None, metadata: dict[str, Any] | None = None) -> Any:
        self._clients[client_id] = (send, filter, metadata or {})
        return lambda message: self.receive(message, client_id)

    def remove_client(self, client_id: str) -> bool:
        return self._clients.pop(client_id, None) is not None

    def receive(self, message: dict[str, Any], sender_id: str | None = None) -> dict[str, Any]:
        if message.get("protocolVersion", self.protocol_version) != self.protocol_version:
            return self._send_ack(message, sender_id, [{"code": "protocol_version", "message": "unsupported sync protocol version"}])
        if message.get("type") == "request-snapshot":
            response = self.recover(int(message.get("fromSeq", 0)), self.max_batch_ops or len(self.ops) or 1, sender_id)
            self._send(sender_id, response)
            return response
        operations = list(message.get("ops", []))
        if self.max_batch_ops is not None and len(operations) > self.max_batch_ops:
            return self._send_ack(message, sender_id, [{"code": "batch_too_large", "message": "sync batch exceeds maxBatchOps"}])
        errors: list[dict[str, Any]] = []
        groups: dict[str, list[dict[str, Any]]] = {}
        for index, operation in enumerate(operations):
            validate_sync_operation(operation)
            groups.setdefault(f"tx:{operation.get('clientId')}:{operation.get('transactionId', index)}", []).append(operation)
        accepted: list[dict[str, Any]] = []
        for group in groups.values():
            group_errors: list[dict[str, Any]] = []
            for operation in group:
                context = {"clientId": sender_id or operation["clientId"], "protocolVersion": self.protocol_version}
                if self.authorize is not None and not self.authorize(operation, context):
                    group_errors.append({"code": "unauthorized", "message": "operation was not authorized", "operationId": operation.get("operationId")})
                if self.conflict is not None:
                    result = self.conflict(operation, context)
                    if result is False or isinstance(result, dict) and not result.get("ok"):
                        group_errors.append({"code": "conflict", "message": result.get("message", "operation conflicts") if isinstance(result, dict) else "operation conflicts", "operationId": operation.get("operationId")})
            if group_errors:
                errors.extend(group_errors)
            else:
                accepted.extend(group)
        self._commit(accepted)
        return self._send_ack(message, sender_id, errors)

    def recover(self, from_cursor: int, limit: int, client_id: str | None = None) -> dict[str, Any]:
        valid = self.base_cursor <= from_cursor <= self.cursor
        requested = from_cursor if valid else 0
        offset = requested - self.base_cursor if requested else 0
        page = self.ops[offset:offset + limit]
        visible = page
        if client_id in self._clients and self._clients[client_id][1] is not None:
            _, predicate, metadata = self._clients[client_id]
            context = {"clientId": client_id, "protocolVersion": self.protocol_version, "metadata": metadata}
            visible = [operation for operation in page if predicate(operation, context)]
        return {"type": "snapshot" if not requested else "delta", "clientId": "server", "fromSeq": requested,
                "cursor": (self.base_cursor if not requested else requested) + len(page), "more": offset + len(page) < len(self.ops),
                "ops": visible, "checksum": sync_checksum(visible), "protocolVersion": self.protocol_version,
                "errors": None if valid else [{"code": "cursor_expired", "message": "requested cursor is no longer available"}]}

    def _commit(self, operations: list[dict[str, Any]]) -> None:
        accepted: list[dict[str, Any]] = []
        accepted_transactions: set[str] = set()
        for operation in operations:
            operation_key = f"{operation['clientId']}:{operation['operationId']}" if operation.get("operationId") else None
            transaction_key = f"{operation['clientId']}:{operation['transactionId']}" if operation.get("transactionId") else None
            if operation_key in self._operation_ids or (transaction_key in self._transaction_ids and transaction_key not in accepted_transactions):
                continue
            if transaction_key:
                accepted_transactions.add(transaction_key)
            if operation_key:
                self._operation_ids.add(operation_key)
            if transaction_key:
                self._transaction_ids.add(transaction_key)
            accepted.append(operation)
        self.ops.extend(accepted)
        if self.max_ops is not None and len(self.ops) > self.max_ops:
            removed = len(self.ops) - self.max_ops
            self.ops = self.ops[removed:]
            self.base_cursor += removed
        if self.operation_log is not None:
            self.operation_log.append_batch(self.ops, self.base_cursor, self._operation_ids, self._transaction_ids)
        for operation in accepted:
            for client_id, (send, predicate, metadata) in self._clients.items():
                if client_id == operation.get("clientId"):
                    continue
                context = {"clientId": client_id, "protocolVersion": self.protocol_version, "metadata": metadata}
                visible = [operation] if predicate is None or predicate(operation, context) else []
                if visible:
                    send({"type": "delta", "clientId": "server", "fromSeq": self.cursor - len(accepted), "cursor": self.cursor, "ops": visible, "checksum": sync_checksum(visible), "protocolVersion": self.protocol_version})

    def _send(self, client_id: str | None, message: dict[str, Any]) -> None:
        if client_id in self._clients:
            self._clients[client_id][0](message)

    def _send_ack(self, message: dict[str, Any], client_id: str | None, errors: list[dict[str, Any]]) -> dict[str, Any]:
        response = {"type": "ack", "clientId": message.get("clientId", "server"), "fromSeq": message.get("fromSeq", 0), "ops": [], "protocolVersion": self.protocol_version, "errors": errors or None}
        self._send(client_id, response)
        return response
