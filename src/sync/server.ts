import type { SyncContext, SyncError, SyncMessage, SyncOp } from './types.js'
import { SYNC_PROTOCOL_VERSION } from './types.js'

export interface SyncServerOptions {
  protocolVersion?: number
  maxOps?: number
  authorize?: (operation: SyncOp, context: SyncContext) => Promise<boolean> | boolean
  clientMetadata?: (client: SyncServerClient) => Record<string, unknown> | undefined
}

export type SyncServerClient = {
  send: (msg: SyncMessage) => void
  clientId?: string
}

export interface SyncSubscriptionOptions {
  filter?: (operation: SyncOp, context: SyncContext) => boolean
}

/**
 * Simple relay server. Receives ops from clients, stores them in an
 * in-memory op log, and broadcasts them to all other connected clients.
 */
export class SyncServer {
  private opLog: SyncOp[] = []
  private seenOps = new Set<string>()
  private clients: SyncServerClient[] = []
  private clientFilters = new Map<SyncServerClient, SyncSubscriptionOptions['filter']>()
  private baseCursor = 0
  private readonly options: SyncServerOptions & { protocolVersion: number; maxOps: number }
  onOp?: (op: SyncOp) => void

  constructor(options: SyncServerOptions = {}) {
    this.options = { ...options, protocolVersion: options.protocolVersion ?? SYNC_PROTOCOL_VERSION, maxOps: options.maxOps ?? Number.POSITIVE_INFINITY }
    if ((this.options.maxOps !== Number.POSITIVE_INFINITY && !Number.isInteger(this.options.maxOps)) || this.options.maxOps < 1) throw new RangeError('maxOps must be a positive integer or Infinity')
  }

  /** Register a client transport. Returns a function to handle incoming messages. */
  addClient(handle: SyncServerClient, options: SyncSubscriptionOptions = {}): (msg: SyncMessage) => void {
    this.clients.push(handle)
    this.clientFilters.set(handle, options.filter)
    return (msg: SyncMessage) => this.handleMessage(msg, handle)
  }

  /** Stop broadcasting to a previously registered client. */
  removeClient(handle: SyncServerClient): boolean {
    const index = this.clients.indexOf(handle)
    if (index === -1) return false
    this.clients.splice(index, 1)
    this.clientFilters.delete(handle)
    return true
  }

  private handleMessage(msg: SyncMessage, sender: SyncServerClient): void {
    if (msg.protocolVersion !== undefined && msg.protocolVersion !== this.options.protocolVersion) {
      sender.send({ type: 'ack', clientId: 'server', fromSeq: msg.fromSeq, ops: [], protocolVersion: this.options.protocolVersion, errors: [{ code: 'protocol_version', message: `Unsupported sync protocol version ${msg.protocolVersion}` }] })
      return
    }
    if (msg.type === 'request-snapshot') {
      const cursorIsValid = msg.fromSeq >= this.baseCursor && msg.fromSeq <= this.cursor
      const requestedCursor = cursorIsValid ? msg.fromSeq : 0
      const offset = requestedCursor === 0 ? 0 : requestedCursor - this.baseCursor
      sender.send({
        type: requestedCursor === 0 ? 'snapshot' : 'delta',
        clientId: 'server',
        fromSeq: requestedCursor,
        ops: this.opLog.slice(offset),
        protocolVersion: this.options.protocolVersion,
        errors: cursorIsValid ? undefined : [{ code: 'cursor_expired', message: 'Requested cursor is no longer available' }],
      })
      return
    }
    if (msg.type === 'delta') {
      if (this.options.authorize) {
        void this.authorizeAndProcess(msg, sender)
        return
      }
      this.processDelta(msg, sender)
    }
  }

  private async authorizeAndProcess(msg: SyncMessage, sender: SyncServerClient): Promise<void> {
    const accepted: SyncOp[] = []
    const errors: SyncError[] = []
    const context: SyncContext = { clientId: msg.clientId, protocolVersion: msg.protocolVersion ?? this.options.protocolVersion, metadata: this.options.clientMetadata?.(sender) }
    for (const op of msg.ops) {
      if (await this.options.authorize!(op, context)) accepted.push(op)
      else errors.push({ code: 'unauthorized', message: `Operation ${op.operationId ?? `${op.clientId}:${op.seq}`} was not authorized`, operationId: op.operationId })
    }
    this.processDelta({ ...msg, ops: accepted }, sender, errors)
  }

  private processDelta(msg: SyncMessage, sender: SyncServerClient, errors: SyncError[] = []): void {
      const broadcastCursor = this.cursor
      const accepted: SyncOp[] = []
      for (const op of msg.ops) {
        const key = `${op.clientId}:${op.seq}`
        if (this.seenOps.has(key)) continue
        this.seenOps.add(key)
        this.opLog.push(op)
        accepted.push(op)
        this.onOp?.(op)
        while (this.opLog.length > this.options.maxOps) {
          this.opLog.shift()
          this.baseCursor++
        }
      }
      const acknowledgedSeq = msg.ops.reduce((max, op) => Math.max(max, op.seq), msg.fromSeq)
      sender.send({
        type: 'ack',
        clientId: msg.clientId,
        fromSeq: acknowledgedSeq,
        ops: [],
        protocolVersion: this.options.protocolVersion,
        errors: errors.length ? errors : undefined,
      })
      // Broadcast to all OTHER clients
      for (const client of accepted.length === 0 ? [] : this.clients) {
        if (client === sender) continue
        const filter = this.clientFilters.get(client)
        const context: SyncContext = { clientId: client.clientId ?? 'unknown', protocolVersion: this.options.protocolVersion }
        const visible = filter ? accepted.filter(op => filter(op, context)) : accepted
        if (visible.length === 0) continue
        client.send({
          type: 'delta',
          clientId: 'server',
          fromSeq: broadcastCursor,
          ops: visible,
          protocolVersion: this.options.protocolVersion,
        })
      }
  }

  /** Every operation the server has accepted, in order. */
  get ops(): readonly SyncOp[] {
    return this.opLog
  }

  /** Cursor immediately after the latest accepted server operation. */
  get cursor(): number {
    return this.baseCursor + this.opLog.length
  }
}
