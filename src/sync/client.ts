import { Subscription } from 'rxjs'
import type { PolyGraph } from '../graph'
import type { GraphChangeEvent } from '../types'
import type { SyncTransport } from './transport'
import type { SyncMessage, SyncOp } from './types'
import { OpLog } from './oplog'

export interface SyncClientOptions {
  graph: PolyGraph
  transport: SyncTransport
  clientId?: string
  /** Automatically flush on every mutation (default: true). Set false for manual batching. */
  autoFlush?: boolean
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
export class SyncClient {
  private graph: PolyGraph
  private transport: SyncTransport
  readonly oplog: OpLog
  private lastSentSeq = 0
  private subscription: Subscription
  private autoFlush: boolean
  private applyingRemote = false

  constructor(options: SyncClientOptions) {
    this.graph = options.graph
    this.transport = options.transport
    this.oplog = new OpLog(options.clientId ?? `client-${Date.now()}`)
    this.autoFlush = options.autoFlush ?? true

    this.subscription = this.graph.changes.subscribe((event) => {
      if (this.applyingRemote) return
      this.handleLocalChange(event)
    })
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
    const ops = this.oplog.since(this.lastSentSeq)
    if (ops.length === 0) return
    this.lastSentSeq = this.oplog.latestSeq
    this.transport.send({
      type: 'delta',
      clientId: this.oplog.clientId,
      fromSeq: 0,
      ops,
    })
  }

  /** Handle an incoming message from the transport. */
  handleMessage(msg: SyncMessage): void {
    if (msg.type === 'delta' || msg.type === 'snapshot') {
      this.applyRemote(msg.ops)
    }
  }

  disconnect(): void {
    this.subscription.unsubscribe()
    this.transport.close()
  }
}
