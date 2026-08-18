import type { PersistenceAdapter, PersistenceChanges, PersistedNodeQuery } from '../persistence/adapter.js'
import { applyPersistedNodeQuery } from '../persistence/query.js'
import type { SerializedNode, SerializedEdge, MutationRecord } from '../types.js'
import { OpLog } from './oplog.js'

export type OpCallback = (op: import('./types').SyncOp) => void

/**
 * Wraps any PersistenceAdapter and records all mutations to an OpLog.
 * When `onOp` is set, every new op is forwarded (typically to a SyncClient).
 */
export class SyncAdapter implements PersistenceAdapter {
  readonly oplog: OpLog
  private inner: PersistenceAdapter
  onOp: OpCallback | null = null

  get capabilities() {
    return this.inner.capabilities
  }

  constructor(inner: PersistenceAdapter, clientId: string) {
    this.inner = inner
    this.oplog = new OpLog(clientId)
  }

  private record(kind: import('./types').SyncOp['kind'], payload: Record<string, unknown>) {
    const op = this.oplog.append(kind, payload)
    this.onOp?.(op)
  }

  private validateEdgeRecord(edge: SerializedEdge): void {
    if (!edge.id || !edge.source || !edge.type || !edge.target) {
      throw new TypeError('Edge id, source, type, and target must not be empty')
    }
  }

  async applyChanges(changes: PersistenceChanges): Promise<void> {
    for (const edge of changes.putEdges) this.validateEdgeRecord(edge)
    if (this.inner.applyChanges) {
      await this.inner.applyChanges(changes)
    } else {
      await this.inner.bulkDeleteNodes(changes.deleteNodeIds)
      await this.inner.bulkDeleteEdges(changes.deleteEdgeIds)
      await Promise.all(changes.deleteVectorIds.map(id => this.inner.deleteVector(id)))
      await this.inner.bulkPutNodes(changes.putNodes)
      await this.inner.bulkPutEdges(changes.putEdges)
      await this.inner.bulkPutVectors(changes.putVectors)
    }
    for (const id of changes.deleteNodeIds) this.record('removeNode', { id })
    for (const id of changes.deleteEdgeIds) {
      this.record('removeEdges', this.edgeRemovalPayload(id))
    }
    for (const node of changes.putNodes) {
      this.record('addNode', node as unknown as Record<string, unknown>)
    }
    for (const edge of changes.putEdges) {
      this.record('addEdge', edge as unknown as Record<string, unknown>)
    }
  }

  async getIndexDefinitions() {
    return this.inner.getIndexDefinitions ? this.inner.getIndexDefinitions() : []
  }

  async getMutationsSince(sequence: bigint): Promise<MutationRecord[]> {
    return this.inner.getMutationsSince ? this.inner.getMutationsSince(sequence) : []
  }

  async latestMutationSequence(): Promise<bigint> {
    return this.inner.latestMutationSequence ? this.inner.latestMutationSequence() : 0n
  }

  async stats() {
    return this.inner.stats ? this.inner.stats() : {}
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

  async queryNodes(query: PersistedNodeQuery): Promise<SerializedNode[]> {
    if (this.inner.queryNodes) return this.inner.queryNodes(query)
    const nodes = await this.inner.getNodes(await this.inner.allNodeIds())
    return applyPersistedNodeQuery(nodes, query)
  }

  async countNodes(query: PersistedNodeQuery): Promise<number> {
    if (this.inner.countNodes) return this.inner.countNodes(query)
    return (await this.queryNodes(query)).length
  }

  async putEdge(edge: SerializedEdge): Promise<void> {
    this.validateEdgeRecord(edge)
    await this.inner.putEdge(edge)
    this.record('addEdge', edge as unknown as Record<string, unknown>)
  }

  async bulkPutEdges(edges: SerializedEdge[]): Promise<void> {
    for (const edge of edges) this.validateEdgeRecord(edge)
    await this.inner.bulkPutEdges(edges)
    for (const edge of edges) {
      this.record('addEdge', edge as unknown as Record<string, unknown>)
    }
  }

  async getAllEdges(): Promise<SerializedEdge[]> {
    return this.inner.getAllEdges()
  }

  async getEdgesBySources(sources: string[], type?: string): Promise<SerializedEdge[]> {
    if (this.inner.getEdgesBySources) return this.inner.getEdgesBySources(sources, type)
    const sourceSet = new Set(sources)
    return (await this.inner.getAllEdges()).filter(edge =>
      sourceSet.has(edge.source) && (type === undefined || edge.type === type)
    )
  }

  async getEdgesByTargets(targets: string[], type?: string): Promise<SerializedEdge[]> {
    if (this.inner.getEdgesByTargets) return this.inner.getEdgesByTargets(targets, type)
    const targetSet = new Set(targets)
    return (await this.inner.getAllEdges()).filter(edge =>
      targetSet.has(edge.target) && (type === undefined || edge.type === type)
    )
  }

  async deleteEdge(id: string): Promise<void> {
    await this.inner.deleteEdge(id)
    this.record('removeEdges', this.edgeRemovalPayload(id))
  }

  async bulkDeleteEdges(ids: string[]): Promise<void> {
    await this.inner.bulkDeleteEdges(ids)
    for (const id of ids) {
      this.record('removeEdges', this.edgeRemovalPayload(id))
    }
  }

  private edgeRemovalPayload(id: string): Record<string, unknown> {
    const parts = id.split('::')
    if (parts.length >= 3) return { id, source: parts[0], type: parts[1], target: parts.slice(2).join('::') }
    return { id }
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
