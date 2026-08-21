import { Subscription } from 'rxjs'
import type { PolyGraph } from '../graph.js'
import type { GraphChangeEvent, NodeActivation } from '../types.js'
import { mergeActivation } from '../activation.js'
import type { SyncTransport } from './transport.js'
import type { SyncError, SyncMessage, SyncOp } from './types.js'
import { SYNC_PROTOCOL_VERSION } from './types.js'
import { edgeId } from '../utils.js'
import { OpLog } from './oplog.js'
import { syncChecksum } from './checksum.js'
import type { SyncClientState, SyncClientStateStore } from './log.js'

/** Configuration for a graph synchronization client. */
export interface SyncClientOptions {
  graph: PolyGraph
  transport: SyncTransport
  clientId?: string
  /** Automatically flush on every mutation (default: true). Set false for manual batching. */
  autoFlush?: boolean
  /** Retry interval for unacknowledged operations in milliseconds. Set 0 to disable. */
  retryMs?: number
  /**
   * Activation deltas below this magnitude are dropped instead of synced, so
   * transient noise never floods the op log. Default 0.05.
   */
  activationSyncThreshold?: number
  protocolVersion?: number
  onError?: (error: SyncError) => void
  /** Maximum operations sent in one transport message. Defaults to Infinity. */
  maxOpsPerMessage?: number
  /** Maximum locally retained operations, including acknowledged history. Defaults to Infinity. */
  maxRetainedOps?: number
  /** Optional durable store for pending operations and sync cursors. */
  stateStore?: SyncClientStateStore
  /** State loaded by {@link SyncClient.restore}; normally not set by callers. */
  initialState?: SyncClientState
}

const NODE_OPS: Record<string, SyncOp['kind']> = {
  node_added: 'addNode',
  node_updated: 'updateNode',
  node_removed: 'removeNode',
}

const EDGE_OPS: Record<string, SyncOp['kind']> = {
  edge_added: 'addEdge',
  edge_updated: 'updateEdge',
  edge_removed: 'removeEdges',
}

const DEFAULT_ACTIVATION_SYNC_THRESHOLD = 0.05

/**
 * Syncs a local PolyGraph with a remote server: subscribes to graph change
 * events, forwards mutations through a transport, and applies remote
 * operations back onto the graph without echoing them.
 */
export class SyncClient {
  private graph: PolyGraph
  private transport!: SyncTransport
  readonly oplog: OpLog
  private lastAckedSeq = 0
  private subscription: Subscription
  private autoFlush: boolean
  private retryMs: number
  private activationSyncThreshold: number
  /** Coalesced, not-yet-materialised activation deltas per node. */
  private pendingActivation = new Map<string, { delta: number; reason?: string }>()
  private retryTimer: ReturnType<typeof setTimeout> | null = null
  private serverCursor = 0
  private applyingRemote = false
  private protocolVersion: number
  private onError?: (error: SyncError) => void
  private maxOpsPerMessage: number
  private maxRetainedOps: number
  private scheduledTransactionFlushes = new Set<string>()
  private disconnected = false
  private stateStore?: SyncClientStateStore
  private stateQueue: Promise<void> = Promise.resolve()

  /** Open a client from its durable local state before subscribing to graph changes. */
  static async restore(options: SyncClientOptions & { stateStore: SyncClientStateStore }): Promise<SyncClient> {
    const state = await options.stateStore.load()
    if (state && options.clientId && state.clientId !== options.clientId) {
      throw new Error(`Persisted sync client id ${state.clientId} does not match ${options.clientId}`)
    }
    const client = new SyncClient({ ...options, clientId: options.clientId ?? state?.clientId, initialState: state ?? undefined })
    client.flush()
    client.requestSync()
    return client
  }

  constructor(options: SyncClientOptions) {
    this.graph = options.graph
    this.transport = options.transport
    const initialState = options.initialState
    this.oplog = new OpLog(options.clientId ?? initialState?.clientId ?? `client-${Date.now()}`, initialState?.ops)
    this.lastAckedSeq = initialState?.lastAckedSeq ?? 0
    this.serverCursor = initialState?.serverCursor ?? 0
    this.stateStore = options.stateStore
    this.oplog.onChange = () => this.persistState()
    this.autoFlush = options.autoFlush ?? true
    this.retryMs = options.retryMs ?? 1000
    this.activationSyncThreshold = options.activationSyncThreshold ?? DEFAULT_ACTIVATION_SYNC_THRESHOLD
    this.protocolVersion = options.protocolVersion ?? SYNC_PROTOCOL_VERSION
    this.onError = options.onError
    this.maxOpsPerMessage = options.maxOpsPerMessage ?? Number.POSITIVE_INFINITY
    if (this.maxOpsPerMessage !== Number.POSITIVE_INFINITY && (!Number.isInteger(this.maxOpsPerMessage) || this.maxOpsPerMessage < 1)) throw new RangeError('maxOpsPerMessage must be a positive integer or Infinity')
    this.maxRetainedOps = options.maxRetainedOps ?? Number.POSITIVE_INFINITY
    if (this.maxRetainedOps !== Number.POSITIVE_INFINITY && (!Number.isInteger(this.maxRetainedOps) || this.maxRetainedOps < 1)) throw new RangeError('maxRetainedOps must be a positive integer or Infinity')
    this.bindTransport(options.transport)

    this.subscription = this.graph.changes.subscribe((event) => {
      if (this.applyingRemote) return
      this.handleLocalChange(event)
    })
  }

  private bindTransport(transport: SyncTransport): void {
    this.transport = transport
    this.transport.onMessage = (msg) => this.handleMessage(msg)
  }

  private persistState(): void {
    if (!this.stateStore) return
    const state: SyncClientState = {
      clientId: this.oplog.clientId,
      lastAckedSeq: this.lastAckedSeq,
      serverCursor: this.serverCursor,
      ops: [...this.oplog.all],
    }
    this.stateQueue = this.stateQueue.then(() => this.stateStore!.save(state), () => this.stateStore!.save(state))
  }

  /** Wait until the latest local pending state has been durably saved. */
  async persist(): Promise<void> {
    this.persistState()
    await this.stateQueue
  }

  private handleLocalChange(event: GraphChangeEvent): void {
    if (event.type === 'activation_updated') {
      // Activation is a derived, statistical signal: coalesce deltas per node
      // and only materialise them into ops at flush time, gated by threshold.
      if (event.nodeId && typeof event.delta === 'number') {
        const pending = this.pendingActivation.get(event.nodeId)
        this.pendingActivation.set(event.nodeId, {
          delta: (pending?.delta ?? 0) + event.delta,
          reason: event.reason ?? pending?.reason,
        })
        if (this.autoFlush) this.flush()
      }
      return
    }

    let kind: SyncOp['kind'] | undefined
    let payload: Record<string, unknown> | undefined

    if (event.type.startsWith('node_')) {
      kind = NODE_OPS[event.type]
      if (!kind) return
      const node = event.nodeId ? this.graph.getNode(event.nodeId) : undefined
      if (event.type === 'node_removed') {
        payload = { id: event.nodeId }
      } else if (node) {
        payload = {
          id: node.id,
          type: node.type,
          data: node.data,
          vector: node.vector ? [...node.vector] : null,
          insertedAt: node.insertedAt,
          updatedAt: node.updatedAt,
          revision: node.revision,
          activation: node.activation ? { ...node.activation } : undefined,
        }
      } else {
        return
      }
    } else if (event.type.startsWith('edge_')) {
      kind = EDGE_OPS[event.type]
      if (!kind) return
      if (event.type === 'edge_added' || event.type === 'edge_updated') {
        const edges = event.source ? this.graph.getEdges(event.source) : []
        const edge = event.edgeId ? edges.find(e => e.id === event.edgeId) : edges.find(e => e.target === event.target && e.type === event.edgeType)
        payload = {
          source: event.source,
          id: edge?.id,
          type: event.edgeType,
          target: event.target,
          data: edge?.data ?? null,
          createdAt: edge?.createdAt ?? Date.now(),
          revision: edge?.revision,
        }
      } else {
        payload = {
          source: event.source,
          id: event.edgeId,
          type: event.edgeType,
          target: event.target,
        }
      }
    }

    if (!kind || !payload) return

    this.oplog.append(kind, payload, event.transactionId ? {
      transactionId: event.transactionId,
      operationId: `${event.transactionId}:${this.oplog.latestSeq + 1}`,
    } : undefined)
    if (this.autoFlush) {
      if (event.transactionId) {
        if (this.scheduledTransactionFlushes.has(event.transactionId)) return
        this.scheduledTransactionFlushes.add(event.transactionId)
        queueMicrotask(() => {
          this.scheduledTransactionFlushes.delete(event.transactionId!)
          if (!this.disconnected) this.flush()
        })
      } else {
        this.flush()
      }
    }
  }

  /** Apply remote ops to the local graph without triggering re-sync. */
  applyRemote(ops: SyncOp[]): void {
    if (ops.length === 0) return
    // A client has already applied its own operations locally. A cursor-gap
    // catch-up can re-deliver them (the relay never echoes them to their
    // sender), so skip them to avoid duplicate change events and redundant work.
    const remote = ops.filter(op => !(op.clientId === this.oplog.clientId && op.seq <= this.oplog.latestSeq))
    if (remote.length === 0) return
    this.applyingRemote = true
    this.graph.startBatch()
    try {
      for (const op of remote) {
        this.applyOp(op)
      }
    } finally {
      this.graph.endBatch()
      this.applyingRemote = false
    }
  }

  private applyOp(op: SyncOp): void {
    const p = op.payload
    switch (op.kind) {
      case 'addNode':
        this.graph.addNode({
          id: p.id as string,
          type: p.type as string,
          data: p.data as Record<string, unknown>,
          vector: p.vector ? new Float64Array(p.vector as number[]) : undefined,
          insertedAt: p.insertedAt as number,
          updatedAt: p.updatedAt as number,
          activation: this.mergedActivation(p.id as string, p.activation as NodeActivation | undefined),
        })
        break
      case 'updateNode':
        this.graph.updateNode(
          p.id as string,
          p.data as Record<string, unknown>,
          p.vector ? new Float64Array(p.vector as number[]) : undefined,
          this.mergedActivation(p.id as string, p.activation as NodeActivation | undefined),
        )
        break
      case 'removeNode':
        this.graph.removeNode(p.id as string)
        break
      case 'addEdge':
        this.graph.addEdge({
          id: (p.id as string | undefined) ?? edgeId(p.source as string, p.type as string, p.target as string),
          source: p.source as string,
          type: p.type as string,
          target: p.target as string,
          data: p.data as Record<string, unknown> | undefined,
          createdAt: (p.createdAt as number | undefined) ?? Date.now(),
          revision: p.revision as number | undefined,
        })
        break
      case 'updateEdge':
        this.graph.updateEdge(
          p.id as string,
          p.data as Record<string, unknown> ?? {},
          typeof p.revision === 'number' ? { expectedRevision: p.revision - 1 } : undefined,
        )
        break
      case 'removeEdges':
        if (typeof p.id === 'string') this.graph.removeEdge(p.id)
        else this.graph.removeEdges(p.source as string, p.type as string, p.target as string | undefined)
        break
      case 'activationUpdate':
        // Activation deltas accumulate — not last-write-wins. Apply synchronously
        // when the node is loaded so the resulting change event stays inside the
        // `applyingRemote` window (no echo); fall back to an async restore for
        // evicted nodes.
        if (this.graph.hasLoadedNode(p.node as string)) {
          this.graph.reinforceNode(p.node as string, p.delta as number, p.reason as string | undefined)
        } else {
          this.graph.reinforceNodeSafe(p.node as string, p.delta as number, p.reason as string | undefined)
        }
        break
    }
  }

  /**
   * Merge a full-node activation payload into the local node's durable state.
   * Full states merge by max (idempotent across re-delivered snapshots) and an
   * absent incoming activation never wipes locally learned state; concurrent
   * deltas use the additive `activationUpdate` op instead.
   */
  private mergedActivation(id: string, incoming: NodeActivation | undefined): NodeActivation | undefined {
    const existing = this.graph.getNode(id)?.activation
    if (existing && !incoming) return existing
    if (!existing && incoming) return incoming
    if (existing && incoming) return mergeActivation(existing, incoming)
    return undefined
  }

  /** Materialise coalesced activation deltas into ops, dropping sub-threshold noise. */
  private materializeActivationOps(): void {
    if (this.pendingActivation.size === 0) return
    for (const [nodeId, { delta, reason }] of this.pendingActivation) {
      if (Math.abs(delta) >= this.activationSyncThreshold) {
        this.oplog.append('activationUpdate', { node: nodeId, delta, reason })
      }
    }
    this.pendingActivation.clear()
  }

  /** Send pending local ops through the transport. */
  flush(): void {
    this.materializeActivationOps()
    const ops = this.oplog.since(this.lastAckedSeq)
    if (ops.length === 0) return
    this.transport.send({
      type: 'delta',
      clientId: this.oplog.clientId,
      fromSeq: this.lastAckedSeq,
      ops: ops.slice(0, this.maxOpsPerMessage),
      protocolVersion: this.protocolVersion,
    })
    this.scheduleRetry()
  }

  private scheduleRetry(): void {
    if (this.retryTimer) clearTimeout(this.retryTimer)
    this.retryTimer = null
    if (this.retryMs <= 0 || this.oplog.since(this.lastAckedSeq).length === 0) return
    this.retryTimer = setTimeout(() => {
      this.retryTimer = null
      this.flush()
    }, this.retryMs)
  }

  /** Handle an incoming message from the transport. */
  handleMessage(msg: SyncMessage): void {
    if (msg.type === 'ack' && msg.clientId === this.oplog.clientId) {
      for (const error of msg.errors ?? []) this.onError?.(error)
      this.lastAckedSeq = Math.max(this.lastAckedSeq, Math.min(msg.fromSeq, this.oplog.latestSeq))
      while (this.oplog.size > this.maxRetainedOps && this.oplog.all[0]?.seq <= this.lastAckedSeq) {
        this.oplog.dropThrough(this.oplog.all[0].seq)
      }
      this.persistState()
      this.scheduleRetry()
      return
    }
    if (msg.type === 'snapshot' && msg.clientId === 'server') {
      for (const error of msg.errors ?? []) this.onError?.(error)
      if (msg.checksum && msg.checksum !== syncChecksum(msg.ops)) {
        this.onError?.({ code: 'checksum_mismatch', message: 'Snapshot checksum mismatch' })
        this.requestSync(true)
        return
      }
      this.applyRemote(msg.ops)
      this.serverCursor = msg.cursor ?? msg.fromSeq + msg.ops.length
      this.persistState()
      if (msg.more) this.requestSync()
      return
    }
    if (msg.type === 'delta' && msg.clientId === 'server') {
      for (const error of msg.errors ?? []) this.onError?.(error)
      if (msg.checksum && msg.checksum !== syncChecksum(msg.ops)) {
        this.onError?.({ code: 'checksum_mismatch', message: 'Delta checksum mismatch' })
        this.requestSync(true)
        return
      }
      if (msg.fromSeq > this.serverCursor) {
        this.requestSync()
        return
      }
      const alreadyApplied = Math.max(0, this.serverCursor - msg.fromSeq)
      const unseen = msg.ops.slice(alreadyApplied)
      this.applyRemote(unseen)
      this.serverCursor = Math.max(this.serverCursor, msg.cursor ?? msg.fromSeq + msg.ops.length)
      this.persistState()
      if (msg.more) this.requestSync()
      return
    }
    if (msg.type === 'delta' || msg.type === 'snapshot') {
      this.applyRemote(msg.ops)
    }
  }

  /** Request all server operations after the last applied server cursor. */
  requestSync(fromStart = false): void {
    this.transport.send({
      type: 'request-snapshot',
      clientId: this.oplog.clientId,
      fromSeq: fromStart ? 0 : this.serverCursor,
      ops: [],
      protocolVersion: this.protocolVersion,
    })
  }

  /** Last server op sequence this client has caught up to. */
  get syncCursor(): number {
    return this.serverCursor
  }

  /** Operations retained until the server acknowledges their sequence. */
  get pendingOps(): readonly SyncOp[] {
    return this.oplog.since(this.lastAckedSeq)
  }

  /** Replace a disconnected transport and immediately resend pending operations. */
  reconnect(transport: SyncTransport): void {
    this.disconnected = false
    if (this.retryTimer) clearTimeout(this.retryTimer)
    this.retryTimer = null
    this.transport.close()
    this.bindTransport(transport)
    this.flush()
    this.requestSync()
  }

  /**
   * Flush pending operations, stop retrying, unsubscribe from graph changes,
   * and close the transport. Flushing first means buffered ops (including
   * coalesced sub-threshold `autoFlush: false` activation deltas) are sent
   * before teardown instead of silently discarded.
   */
  disconnect(): void {
    this.flush()
    this.disconnected = true
    if (this.retryTimer) clearTimeout(this.retryTimer)
    this.retryTimer = null
    this.subscription.unsubscribe()
    this.transport.close()
  }
}
