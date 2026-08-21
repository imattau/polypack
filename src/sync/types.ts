export const SYNC_PROTOCOL_VERSION = 1

export interface SyncContext {
  clientId: string
  protocolVersion: number
  metadata?: Record<string, unknown>
}

export interface SyncError {
  code: 'unauthorized' | 'conflict' | 'protocol_version' | 'cursor_expired' | 'batch_too_large' | 'pending_too_large' | 'persistence_error' | 'checksum_mismatch'
  message: string
  operationId?: string
}

export type SyncConflictResult = boolean | { ok: boolean; message?: string }

/** Ordered graph mutation exchanged by sync clients. */
export interface SyncOp {
  seq: number
  timestamp: number
  clientId: string
  kind: 'addNode' | 'updateNode' | 'removeNode' | 'addEdge' | 'updateEdge' | 'removeEdges' | 'activationUpdate' | 'inhibitionUpdate'
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
  /** Global server cursor after applying this envelope. Required for filtered deltas. */
  cursor?: number
  /** More server operations are available after this envelope. */
  more?: boolean
  ops: SyncOp[]
  protocolVersion?: number
  checksum?: string
  errors?: SyncError[]
}
