import type { SyncOp } from './types.js'

/** Deterministic, transport-friendly checksum for an ordered operation batch. */
export function syncChecksum(ops: readonly SyncOp[]): string {
  const input = JSON.stringify(ops, (_key, value: unknown) => typeof value === 'bigint' ? `${value}n` : value)
  let hash = 2166136261
  for (let i = 0; i < input.length; i++) {
    hash ^= input.charCodeAt(i)
    hash = Math.imul(hash, 16777619)
  }
  return (hash >>> 0).toString(16).padStart(8, '0')
}

/** Integrity marker for deduplication identities retained after compaction. */
export function syncIdentityChecksum(operationIds: readonly string[], transactionIds: readonly string[]): string {
  return syncChecksum([{ seq: 0, timestamp: 0, clientId: 'sync-identities', kind: 'addNode', payload: { operationIds: [...operationIds].sort(), transactionIds: [...transactionIds].sort() } }])
}
