import type { PersistenceAdapter } from '../persistence/adapter'
import type { SerializedNode, SerializedEdge } from '../types'
import { OpLog } from './oplog'

export type OpCallback = (op: import('./types').SyncOp) => void

/**
 * Wraps any PersistenceAdapter and records all mutations to an OpLog.
 * When `onOp` is set, every new op is forwarded (typically to a SyncClient).
 */
export class SyncAdapter implements PersistenceAdapter {
  readonly oplog: OpLog
  private inner: PersistenceAdapter
  onOp: OpCallback | null = null

  constructor(inner: PersistenceAdapter, clientId: string) {
    this.inner = inner
    this.oplog = new OpLog(clientId)
  }

  private record(kind: import('./types').SyncOp['kind'], payload: Record<string, unknown>) {
    const op = this.oplog.append(kind, payload)
    this.onOp?.(op)
  }

  async putNode(node: SerializedNode): Promise<void> {
    await this.inner.putNode(node)
    this.record('addNode', node as unknown as Record<string, unknown>)
  }

  async bulkPutNodes(nodes: SerializedNode[]): Promise<void> {
    await this.inner.bulkPutNodes(nodes)
    for (const node of nodes) {
      this.record('addNode', node as unknown as Record<string, unknown>)
    }
  }

  async getNode(id: string): Promise<SerializedNode | undefined> {
    return this.inner.getNode(id)
  }

  async getNodes(ids: string[]): Promise<SerializedNode[]> {
    return this.inner.getNodes(ids)
  }

  async deleteNode(id: string): Promise<void> {
    await this.inner.deleteNode(id)
    this.record('removeNode', { id })
  }

  async bulkDeleteNodes(ids: string[]): Promise<void> {
    await this.inner.bulkDeleteNodes(ids)
    for (const id of ids) {
      this.record('removeNode', { id })
    }
  }

  async allNodeIds(): Promise<string[]> {
    return this.inner.allNodeIds()
  }

  async putEdge(edge: SerializedEdge): Promise<void> {
    await this.inner.putEdge(edge)
    this.record('addEdge', edge as unknown as Record<string, unknown>)
  }

  async bulkPutEdges(edges: SerializedEdge[]): Promise<void> {
    await this.inner.bulkPutEdges(edges)
    for (const edge of edges) {
      this.record('addEdge', edge as unknown as Record<string, unknown>)
    }
  }

  async getAllEdges(): Promise<SerializedEdge[]> {
    return this.inner.getAllEdges()
  }

  async deleteEdge(id: string): Promise<void> {
    await this.inner.deleteEdge(id)
    const [source, type, ...rest] = id.split('::')
    this.record('removeEdges', { source, type: type ?? '', target: rest.join('::') })
  }

  async bulkDeleteEdges(ids: string[]): Promise<void> {
    await this.inner.bulkDeleteEdges(ids)
    for (const id of ids) {
      const [source, type, ...rest] = id.split('::')
      this.record('removeEdges', { source, type: type ?? '', target: rest.join('::') })
    }
  }

  async putVector(id: string, vector: number[]): Promise<void> {
    await this.inner.putVector(id, vector)
  }

  async bulkPutVectors(entries: Array<{ id: string; vector: number[] }>): Promise<void> {
    await this.inner.bulkPutVectors(entries)
  }

  async deleteVector(id: string): Promise<void> {
    await this.inner.deleteVector(id)
  }

  async getAllVectors(): Promise<Array<{ id: string; vector: number[] }>> {
    return this.inner.getAllVectors()
  }

  async clearAll(): Promise<void> {
    await this.inner.clearAll()
  }

  async close(): Promise<void> {
    await this.inner.close()
  }
}
