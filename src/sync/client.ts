import { Subscription } from 'rxjs'
import type { PolyGraph } from '../graph.js'
import type { GraphChangeEvent } from '../types.js'
import type { SyncTransport } from './transport.js'
import type { SyncMessage, SyncOp } from './types.js'
import { OpLog } from './oplog.js'

/** Configuration for a graph synchronization client. */
export interface SyncClientOptions {
  graph: PolyGraph
  transport: SyncTransport
  clientId?: string
  /** Automatically flush on every mutation (default: true). Set false for manual batching. */
  autoFlush?: boolean
  /** Retry interval for unacknowledged operations in milliseconds. Set 0 to disable. */
  retryMs?: number
}

const NODE_OPS: Record<string, SyncOp['kind']> = {
  node_added: 'addNode',
  node_updated: 'updateNode',
  node_removed: 'removeNode',
}

const EDGE_OPS: Record<string, SyncOp['kind']> = {
  edge_added: 'addEdge',
  edge_removed: 'removeEdges',
}

/**
 * Syncs a local PolyGraph with a remote server by subscribing to graph
 * change events and forwarding mutations through a transport.
 */
/** Captures local graph changes and applies remote operations without echoing them. */
export class SyncClient {
  private graph: PolyGraph
  private transport!: SyncTransport
  readonly oplog: OpLog
  private lastAckedSeq = 0
  private subscription: Subscription
  private autoFlush: boolean
  private retryMs: number
  private retryTimer: ReturnType<typeof setTimeout> | null = null
  private applyingRemote = false

  constructor(options: SyncClientOptions) {
    this.graph = options.graph
    this.transport = options.transport
    this.oplog = new OpLog(options.clientId ?? `client-${Date.now()}`)
    this.autoFlush = options.autoFlush ?? true
    this.retryMs = options.retryMs ?? 1000
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

  private handleLocalChange(event: GraphChangeEvent): void {
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
        }
      } else {
        return
      }
    } else if (event.type.startsWith('edge_')) {
      kind = EDGE_OPS[event.type]
      if (!kind) return
      if (event.type === 'edge_added') {
        const edges = event.source ? this.graph.getEdges(event.source) : []
        const edge = edges.find(e => e.target === event.target && e.type === event.edgeType)
        payload = {
          source: event.source,
          type: event.edgeType,
          target: event.target,
          data: edge?.data ?? null,
        }
      } else {
        payload = {
          source: event.source,
          type: event.edgeType,
          target: event.target,
        }
      }
    }

    if (!kind || !payload) return

    this.oplog.append(kind, payload)
    if (this.autoFlush) this.flush()
  }

  /** Apply remote ops to the local graph without triggering re-sync. */
  applyRemote(ops: SyncOp[]): void {
    if (ops.length === 0) return
    this.applyingRemote = true
    this.graph.startBatch()
    try {
      for (const op of ops) {
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
        })
        break
      case 'updateNode':
        this.graph.updateNode(
          p.id as string,
          p.data as Record<string, unknown>,
          p.vector ? new Float64Array(p.vector as number[]) : undefined,
        )
        break
      case 'removeNode':
        this.graph.removeNode(p.id as string)
        break
      case 'addEdge':
        this.graph.addEdge(
          p.source as string,
          p.type as string,
          p.target as string,
          p.data as Record<string, unknown> | undefined,
        )
        break
      case 'removeEdges':
        this.graph.removeEdges(
          p.source as string,
          p.type as string,
          p.target as string | undefined,
        )
        break
    }
  }

  /** Send pending local ops through the transport. */
  flush(): void {
    const ops = this.oplog.since(this.lastAckedSeq)
    if (ops.length === 0) return
    this.transport.send({
      type: 'delta',
      clientId: this.oplog.clientId,
      fromSeq: this.lastAckedSeq,
      ops,
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
      this.lastAckedSeq = Math.max(this.lastAckedSeq, Math.min(msg.fromSeq, this.oplog.latestSeq))
      this.scheduleRetry()
      return
    }
    if (msg.type === 'delta' || msg.type === 'snapshot') {
      this.applyRemote(msg.ops)
    }
  }

  /** Operations retained until the server acknowledges their sequence. */
  get pendingOps(): readonly SyncOp[] {
    return this.oplog.since(this.lastAckedSeq)
  }

  /** Replace a disconnected transport and immediately resend pending operations. */
  reconnect(transport: SyncTransport): void {
    if (this.retryTimer) clearTimeout(this.retryTimer)
    this.retryTimer = null
    this.transport.close()
    this.bindTransport(transport)
    this.flush()
  }

  disconnect(): void {
    if (this.retryTimer) clearTimeout(this.retryTimer)
    this.retryTimer = null
    this.subscription.unsubscribe()
    this.transport.close()
  }
}
