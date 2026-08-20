import type { SyncConflictResult, SyncContext, SyncError, SyncMessage, SyncOp } from './types.js'
import { SYNC_PROTOCOL_VERSION } from './types.js'
import type { SyncOperationLog } from './log.js'
import { syncChecksum } from './checksum.js'

export interface SyncServerOptions {
  protocolVersion?: number
  maxOps?: number
  authorize?: (operation: SyncOp, context: SyncContext) => Promise<boolean> | boolean
  conflict?: (operation: SyncOp, context: SyncContext) => Promise<SyncConflictResult> | SyncConflictResult
  clientMetadata?: (client: SyncServerClient) => Record<string, unknown> | undefined
  operationLog?: SyncOperationLog
  maxBatchOps?: number
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
  private seenOperationIds = new Set<string>()
  private clients: SyncServerClient[] = []
  private clientFilters = new Map<SyncServerClient, SyncSubscriptionOptions['filter']>()
  private baseCursor = 0
  private readonly options: SyncServerOptions & { protocolVersion: number; maxOps: number; maxBatchOps: number }
  private readyPromise: Promise<void> | null = null
  private durableQueue: Promise<void> = Promise.resolve()
  onOp?: (op: SyncOp) => void

  constructor(options: SyncServerOptions = {}) {
    this.options = { ...options, protocolVersion: options.protocolVersion ?? SYNC_PROTOCOL_VERSION, maxOps: options.maxOps ?? Number.POSITIVE_INFINITY, maxBatchOps: options.maxBatchOps ?? Number.POSITIVE_INFINITY }
    if ((this.options.maxOps !== Number.POSITIVE_INFINITY && !Number.isInteger(this.options.maxOps)) || this.options.maxOps < 1) throw new RangeError('maxOps must be a positive integer or Infinity')
    if ((this.options.maxBatchOps !== Number.POSITIVE_INFINITY && !Number.isInteger(this.options.maxBatchOps)) || this.options.maxBatchOps < 1) throw new RangeError('maxBatchOps must be a positive integer or Infinity')
  }

  /** Load durable server history before accepting requests. */
  async ready(): Promise<void> {
    if (!this.options.operationLog) return
    if (!this.readyPromise) {
      this.readyPromise = this.options.operationLog.load().then(state => {
        this.baseCursor = state.baseCursor
        this.opLog = state.ops.slice(-this.options.maxOps)
        this.baseCursor += state.ops.length - this.opLog.length
        this.seenOps = new Set(this.opLog.map(op => `${op.clientId}:${op.seq}`))
        this.seenOperationIds = new Set(this.opLog.flatMap(op => op.operationId ? [`${op.clientId}:${op.operationId}`] : []))
      })
    }
    await this.readyPromise
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
      if (this.options.operationLog) {
        void this.ready().then(() => this.sendSnapshot(msg, sender))
        return
      }
      this.sendSnapshot(msg, sender)
      return
    }
    if (msg.type === 'delta') {
      if (msg.ops.length > this.options.maxBatchOps) {
        sender.send({ type: 'ack', clientId: msg.clientId, fromSeq: msg.fromSeq, ops: [], protocolVersion: this.options.protocolVersion, errors: [{ code: 'batch_too_large', message: `Batch contains ${msg.ops.length} operations; maximum is ${this.options.maxBatchOps}` }] })
        return
      }
      if (this.options.authorize || this.options.conflict) {
        void this.authorizeAndProcess(msg, sender)
        return
      }
      this.processDelta(msg, sender)
    }
  }

  private sendSnapshot(msg: SyncMessage, sender: SyncServerClient): void {
    const cursorIsValid = msg.fromSeq >= this.baseCursor && msg.fromSeq <= this.cursor
    const requestedCursor = cursorIsValid ? msg.fromSeq : 0
    const offset = requestedCursor === 0 ? 0 : requestedCursor - this.baseCursor
    const available = this.opLog.slice(offset)
    const ops = available.slice(0, this.options.maxBatchOps)
    sender.send({
      type: requestedCursor === 0 ? 'snapshot' : 'delta',
      clientId: 'server',
      fromSeq: requestedCursor,
      cursor: requestedCursor + ops.length,
      more: ops.length < available.length,
      ops,
      checksum: syncChecksum(ops),
      protocolVersion: this.options.protocolVersion,
      errors: cursorIsValid ? undefined : [{ code: 'cursor_expired', message: 'Requested cursor is no longer available' }],
    })
  }

  private async authorizeAndProcess(msg: SyncMessage, sender: SyncServerClient): Promise<void> {
    const accepted: SyncOp[] = []
    const errors: SyncError[] = []
    const context: SyncContext = { clientId: msg.clientId, protocolVersion: msg.protocolVersion ?? this.options.protocolVersion, metadata: this.options.clientMetadata?.(sender) }
    for (const op of msg.ops) {
      if (this.options.authorize && !(await this.options.authorize(op, context))) {
        errors.push({ code: 'unauthorized', message: `Operation ${op.operationId ?? `${op.clientId}:${op.seq}`} was not authorized`, operationId: op.operationId })
        continue
      }
      if (this.options.conflict) {
        const result = await this.options.conflict(op, context)
        const rejected = result === false || (typeof result === 'object' && !result.ok)
        if (rejected) {
          const message = typeof result === 'object' && result.message ? result.message : `Operation ${op.operationId ?? `${op.clientId}:${op.seq}`} conflicts with current state`
          errors.push({ code: 'conflict', message, operationId: op.operationId })
          continue
        }
      }
      accepted.push(op)
    }
    this.processDelta({ ...msg, ops: accepted }, sender, errors)
  }

  private processDelta(msg: SyncMessage, sender: SyncServerClient, errors: SyncError[] = []): void {
    if (this.options.operationLog) {
      const run = this.durableQueue.then(() => this.processDurableDelta(msg, sender, errors))
      this.durableQueue = run.then(() => undefined, () => undefined)
      return
    }
    this.processInMemoryDelta(msg, sender, errors)
  }

  private processInMemoryDelta(msg: SyncMessage, sender: SyncServerClient, errors: SyncError[] = []): void {
      const broadcastCursor = this.cursor
      const accepted: SyncOp[] = []
      for (const op of msg.ops) {
        const key = `${op.clientId}:${op.seq}`
        const operationKey = op.operationId ? `${op.clientId}:${op.operationId}` : undefined
        if (this.seenOps.has(key) || (operationKey && this.seenOperationIds.has(operationKey))) continue
        this.seenOps.add(key)
        if (operationKey) this.seenOperationIds.add(operationKey)
        this.opLog.push(op)
        accepted.push(op)
        this.onOp?.(op)
        while (this.opLog.length > this.options.maxOps) {
          this.forgetOperation(this.opLog.shift()!)
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
          cursor: this.cursor,
          ops: visible,
          checksum: syncChecksum(visible),
          protocolVersion: this.options.protocolVersion,
        })
      }
  }

  private async processDurableDelta(msg: SyncMessage, sender: SyncServerClient, errors: SyncError[] = []): Promise<void> {
    await this.ready()
    const broadcastCursor = this.cursor
    const accepted: SyncOp[] = []
    for (const op of msg.ops) {
      const key = `${op.clientId}:${op.seq}`
      const operationKey = op.operationId ? `${op.clientId}:${op.operationId}` : undefined
      if (this.seenOps.has(key) || (operationKey && this.seenOperationIds.has(operationKey))) continue
      await this.options.operationLog!.append(op)
      this.seenOps.add(key)
      if (operationKey) this.seenOperationIds.add(operationKey)
      this.opLog.push(op)
      accepted.push(op)
      this.onOp?.(op)
      while (this.opLog.length > this.options.maxOps) {
        this.forgetOperation(this.opLog.shift()!)
        this.baseCursor++
      }
    }
    const log = this.options.operationLog!
    if (log.compact && this.baseCursor > 0) await log.compact(this.baseCursor)
    const acknowledgedSeq = msg.ops.reduce((max, op) => Math.max(max, op.seq), msg.fromSeq)
    sender.send({ type: 'ack', clientId: msg.clientId, fromSeq: acknowledgedSeq, ops: [], protocolVersion: this.options.protocolVersion, errors: errors.length ? errors : undefined })
    for (const client of accepted.length === 0 ? [] : this.clients) {
      if (client === sender) continue
      const filter = this.clientFilters.get(client)
      const context: SyncContext = { clientId: client.clientId ?? 'unknown', protocolVersion: this.options.protocolVersion }
      const visible = filter ? accepted.filter(op => filter(op, context)) : accepted
      if (visible.length === 0) continue
      client.send({ type: 'delta', clientId: 'server', fromSeq: broadcastCursor, cursor: this.cursor, ops: visible, checksum: syncChecksum(visible), protocolVersion: this.options.protocolVersion })
    }
  }

  private forgetOperation(op: SyncOp): void {
    this.seenOps.delete(`${op.clientId}:${op.seq}`)
    if (op.operationId) this.seenOperationIds.delete(`${op.clientId}:${op.operationId}`)
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
