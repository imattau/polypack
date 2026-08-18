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
