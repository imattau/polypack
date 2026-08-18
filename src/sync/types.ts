export const SYNC_PROTOCOL_VERSION = 1

export interface SyncContext {
  clientId: string
  protocolVersion: number
  metadata?: Record<string, unknown>
}

export interface SyncError {
  code: 'unauthorized' | 'conflict' | 'protocol_version' | 'cursor_expired'
  message: string
  operationId?: string
}

export type SyncConflictResult = boolean | { ok: boolean; message?: string }

/** Ordered graph mutation exchanged by sync clients. */
export interface SyncOp {
  seq: number
  timestamp: number
  clientId: string
  kind: 'addNode' | 'updateNode' | 'removeNode' | 'addEdge' | 'removeEdges' | 'activationUpdate'
  payload: Record<string, unknown>
  operationId?: string
  transactionId?: string
  baseRevision?: number
}

/**
 * Transport envelope. `fromSeq` is a client sequence for acknowledgements and
 * a server operation cursor for server deltas, snapshots, and recovery requests.
 */
export interface SyncMessage {
  type: 'delta' | 'snapshot' | 'ack' | 'request-snapshot'
  clientId: string
  fromSeq: number
  ops: SyncOp[]
  protocolVersion?: number
  errors?: SyncError[]
}
