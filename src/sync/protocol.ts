import type { SyncOp } from './types.js'
import { syncChecksum } from './checksum.js'

/** Validate the portable fields required by every sync operation. */
export function validateSyncOperation(operation: SyncOp): void {
  if (!Number.isInteger(operation.seq) || operation.seq < 1) throw new TypeError('Sync sequence must be a positive integer')
  if (!Number.isFinite(operation.timestamp)) throw new TypeError('Sync timestamp must be finite')
  if (!operation.clientId) throw new TypeError('Sync clientId must be non-empty')
  if (!operation.kind) throw new TypeError('Sync operation kind must be non-empty')
  if (!operation.payload || typeof operation.payload !== 'object') throw new TypeError('Sync payload must be an object')
  if (operation.operationId !== undefined && !operation.operationId) throw new TypeError('Sync operationId must be non-empty')
  if (operation.transactionId !== undefined && !operation.transactionId) throw new TypeError('Sync transactionId must be non-empty')
  if (operation.baseRevision !== undefined && (!Number.isInteger(operation.baseRevision) || operation.baseRevision < 0)) throw new TypeError('Sync baseRevision must be a non-negative integer')
}

/** Validate an ordered batch and return its deterministic wire checksum. */
export function validateSyncBatch(operations: readonly SyncOp[]): string {
  operations.forEach(validateSyncOperation)
  return syncChecksum(operations)
}
