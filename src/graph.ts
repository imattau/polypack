import { Subject } from 'rxjs'
import type { PolyNode, EdgeOwnership, GraphChangeEvent, SerializedNode, SerializedEdge } from './types'
import { VectorIndex } from './vector-index'
import { GraphQuery } from './query'
import { edgeId, yieldToUI } from './utils'
import type { PersistenceAdapter } from './persistence/adapter'
import { MemoryAdapter } from './persistence/memory'

type EdgeIndex = Map<string, Array<{ target: string; type: string; data?: Record<string, unknown> }>>

const DEFAULT_HOT_CACHE_MAX = 10000

const OWNERSHIP_KEY = '__ownership'

function getOwnership(data?: Record<string, unknown>): EdgeOwnership {
  return (data?.[OWNERSHIP_KEY] as EdgeOwnership) ?? 'reference'
}

export class PolyGraph {
  protected nodes = new Map<string, PolyNode>()
  protected edges: EdgeIndex = new Map()
  protected nodeToEdgeMap = new Map<string, Set<string>>()

  readonly vectors: VectorIndex
  readonly changes = new Subject<GraphChangeEvent>()
  readonly persistence: PersistenceAdapter
  readonly hotCacheMax: number

  protected hotCacheOrder = new Map<string, true>()
  protected _byType = new Map<string, Set<string>>()

  protected dirtyNodes = new Set<string>()
  protected dirtyEdges = new Set<string>()
  protected dirtyVectors = new Set<string>()
  protected removedEdgeIds = new Set<string>()
  protected removedNodeIds = new Set<string>()
  protected persistTimer: ReturnType<typeof setTimeout> | null = null
  protected evictionSkipCounter = 0

  protected batchDepth = 0
  protected pendingBatchEvents: GraphChangeEvent[] = []

  startBatch(): void {
    this.batchDepth++
  }

  endBatch(): void {
    if (this.batchDepth === 0) throw new Error('endBatch without startBatch')
    this.batchDepth--
    if (this.batchDepth > 0) return
    const events = this.pendingBatchEvents
    this.pendingBatchEvents = []
    for (const ev of events) {
      this.changes.next(ev)
    }
  }

  protected emitChange(event: GraphChangeEvent): void {
    if (this.batchDepth > 0) {
      this.pendingBatchEvents.push(event)
    } else {
      this.changes.next(event)
    }
  }

  constructor(adapter?: PersistenceAdapter, hotCacheMax?: number) {
    this.persistence = adapter ?? new MemoryAdapter()
    this.hotCacheMax = hotCacheMax ?? DEFAULT_HOT_CACHE_MAX
    this.vectors = new VectorIndex((id) => {
      this.dirtyVectors.add(id)
      this.schedulePersist()
    })
  }

  protected markDirty(id: string): void {
    if (!this.removedNodeIds.has(id)) this.dirtyNodes.add(id)
    this.schedulePersist()
  }

  protected schedulePersist(): void {
    if (this.persistTimer) clearTimeout(this.persistTimer)
    this.persistTimer = setTimeout(() => {
      this.flush().catch((err) => console.warn('[PolyGraph] Flush error:', err))
    }, 2000)
  }

  async flush(): Promise<void> {
    const nodesToSave: SerializedNode[] = []
    for (const id of this.dirtyNodes) {
      const node = this.nodes.get(id)
      if (node) {
        nodesToSave.push({
          id: node.id,
          type: node.type,
          data: node.data,
          vector: node.vector ? [...node.vector] : null,
          insertedAt: node.insertedAt,
          updatedAt: node.updatedAt,
        })
      }
    }

    if (this.removedNodeIds.size > 0) {
      const removed = [...this.removedNodeIds]
      await this.persistence.bulkDeleteNodes(removed)
      for (const id of removed) {
        await this.persistence.deleteVector(id)
      }
      this.removedNodeIds.clear()
    }

    if (nodesToSave.length > 0) {
      await this.persistence.bulkPutNodes(nodesToSave)
    }

    const dirtyEdgeList: SerializedEdge[] = []
    for (const edgeIdStr of this.dirtyEdges) {
      const parts = edgeIdStr.split('::')
      if (parts.length < 3) continue
      const [source, type, ...rest] = parts
      const target = rest.join('::')
      const edges = this.edges.get(source)
      const edge = edges?.find(e => e.type === type && e.target === target)
      if (edge) {
        dirtyEdgeList.push({
          id: edgeIdStr,
          source,
          target: edge.target,
          type: edge.type,
          data: edge.data ?? null,
          createdAt: Date.now(),
        })
      }
    }
    await this.persistence.bulkPutEdges(dirtyEdgeList)
    this.dirtyEdges.clear()

    if (this.removedEdgeIds.size > 0) {
      const removedEdges = [...this.removedEdgeIds]
      await this.persistence.bulkDeleteEdges(removedEdges)
      this.removedEdgeIds.clear()
    }

    const dirtyVectorEntries: Array<{ id: string; vector: number[] }> = []
    for (const id of this.dirtyVectors) {
      const vector = this.vectors.get(id)
      if (vector) dirtyVectorEntries.push({ id, vector })
    }
    await this.persistence.bulkPutVectors(dirtyVectorEntries)

    this.dirtyNodes.clear()
    this.dirtyVectors.clear()
  }

  // ── Node CRUD ──

  addNode(node: PolyNode): void {
    this.nodes.set(node.id, node)
    this.touchHotCache(node.id)
    this.markDirty(node.id)
    this.indexNode(node)
    if (node.vector) {
      this.vectors.add(node.id, [...node.vector])
    }
    this.emitChange({ type: 'node_added', nodeId: node.id, nodeType: node.type })
  }

  protected indexNode(node: PolyNode): void {
    const id = node.id
    if (!this._byType.has(node.type)) this._byType.set(node.type, new Set())
    this._byType.get(node.type)!.add(id)
    this.onNodeIndex?.(node)
  }

  protected unindexNode(id: string): void {
    const node = this.nodes.get(id)
    if (!node) return
    const typeSet = this._byType.get(node.type)
    if (typeSet) {
      typeSet.delete(id)
      if (typeSet.size === 0) this._byType.delete(node.type)
    }
    this.onNodeUnindex?.(id, node)
  }

  protected onNodeIndex?(node: PolyNode): void
  protected onNodeUnindex?(id: string, node: PolyNode): void

  /** Extension hook: called when a target node loses its last incoming edge of
   *  any type (becomes disconnected). The node still exists in the graph — the
   *  subclass decides what to do (clean up, log, ignore). */
  protected onOrphan?(id: string): void

  getNode(id: string): PolyNode | undefined {
    const node = this.nodes.get(id)
    if (node) {
      this.touchHotCache(id)
      return node
    }
    return undefined
  }

  async getNodeSafe(id: string): Promise<PolyNode | undefined> {
    const node = this.nodes.get(id)
    if (node) {
      this.touchHotCache(id)
      return node
    }
    const serialized = await this.persistence.getNode(id)
    if (!serialized) return undefined
    const restored: PolyNode = {
      id: serialized.id,
      type: serialized.type,
      data: serialized.data,
      vector: serialized.vector ? new Float64Array(serialized.vector) : undefined,
      insertedAt: serialized.insertedAt,
      updatedAt: serialized.updatedAt,
    }
    this.nodes.set(id, restored)
    this.touchHotCache(id)
    if (restored.vector) {
      this.vectors.add(restored.id, [...restored.vector])
    }
    return restored
  }

  updateNode(id: string, data: Partial<Record<string, unknown>>, vector?: Float64Array): PolyNode | undefined {
    const node = this.nodes.get(id)
    if (!node) return undefined
    Object.assign(node.data, data)
    if (vector !== undefined) {
      node.vector = vector
      this.vectors.add(id, [...vector])
      this.dirtyVectors.add(id)
    }
    node.updatedAt = Date.now()
    this.touchHotCache(id)
    this.markDirty(id)
    this.emitChange({ type: 'node_updated', nodeId: id, nodeType: node.type })
    return node
  }

  /** Returns true if `target` has at least one incoming 'owned' edge from a
   *  source other than `excludeSource`. */
  protected hasOtherOwnedSource(target: string, excludeSource: string): boolean {
    const sources = this.nodeToEdgeMap.get(target)
    if (!sources) return false
    for (const src of sources) {
      if (src === excludeSource) continue
      const srcEdges = this.edges.get(src)
      if (srcEdges?.some(e => e.target === target && getOwnership(e.data) === 'owned')) {
        return true
      }
    }
    return false
  }

  /** Returns true if `target` has at least one incoming edge of any type from
   *  any source other than `excludeSource`. */
  protected hasOtherIncoming(target: string, excludeSource: string): boolean {
    const sources = this.nodeToEdgeMap.get(target)
    if (!sources) return false
    for (const src of sources) {
      if (src === excludeSource) continue
      const srcEdges = this.edges.get(src)
      if (srcEdges?.some(e => e.target === target)) {
        return true
      }
    }
    return false
  }

  /**
   * Remove `id` and cascade through 'owned' edges. Targets of 'owned' edges
   * are also removed unless they have another 'owned' source keeping them
   * alive. Targets of 'shared' edges are notified via `onOrphan` if they
   * become disconnected. Cyclic owned edges (A → B → A) are detected and
   * each node is only removed once.
   */
  removeNode(id: string, cascadeVisited?: Set<string>): void {
    const visited = cascadeVisited ?? new Set<string>()
    if (visited.has(id)) return
    visited.add(id)

    const node = this.nodes.get(id)
    if (!node) return

    // Process outgoing edges before cleanup — a snapshot avoids concurrent
    // modification issues when cascading recursively mutates this.edges.
    const outgoing = [...(this.edges.get(id) ?? [])]
    for (const edge of outgoing) {
      const ownership = getOwnership(edge.data)
      if (ownership === 'owned') {
        if (!this.hasOtherOwnedSource(edge.target, id)) {
          this.removeNode(edge.target, visited)
        }
      }
    }

    this.cleanupNodeEdges(id)
    this.unindexNode(id)
    this.nodes.delete(id)
    this.vectors.remove(id)
    this.dirtyVectors.delete(id)

    this.dirtyNodes.delete(id)
    this.removedNodeIds.add(id)
    this.schedulePersist()
    this.hotCacheOrder.delete(id)
    this.emitChange({ type: 'node_removed', nodeId: id, nodeType: node.type })
  }

  protected cleanupNodeEdges(id: string): void {
    const sourceEdges = this.edges.get(id)
    if (sourceEdges) {
      for (const e of sourceEdges) {
        this.nodeToEdgeMap.get(e.target)?.delete(id)
      }
      this.edges.delete(id)
    }
    this.nodeToEdgeMap.delete(id)
  }

  // ── Edge CRUD ──

  addEdge(source: string, type: string, target: string, data?: Record<string, unknown>, ownership?: EdgeOwnership): void {
    const id = edgeId(source, type, target)
    if (!this.edges.has(source)) this.edges.set(source, [])
    const edges = this.edges.get(source)!
    const existing = edges.find(e => e.type === type && e.target === target)
    if (existing) return

    const fullData = ownership !== undefined ? { ...data, [OWNERSHIP_KEY]: ownership } : (data ?? undefined)

    edges.push({ target, type, data: fullData })
    if (!this.nodeToEdgeMap.has(target)) this.nodeToEdgeMap.set(target, new Set())
    this.nodeToEdgeMap.get(target)!.add(source)
    this.dirtyEdges.add(id)
    this.schedulePersist()
    this.emitChange({ type: 'edge_added', edgeId: id, edgeType: type, source, target })
  }

  markVectorDirty(id: string): void {
    if (this.vectors.has(id)) this.dirtyVectors.add(id)
    this.schedulePersist()
  }

  getEdges(source: string, type?: string): Array<{ target: string; type: string; data?: Record<string, unknown> }> {
    const edges = this.edges.get(source)
    if (!edges) return []
    if (type) return edges.filter(e => e.type === type)
    return [...edges]
  }

  getEdgeTargets(source: string, type: string): string[] {
    const edges = this.edges.get(source)
    if (!edges) return []
    return edges.filter(e => e.type === type).map(e => e.target)
  }

  getEdgeSources(target: string, type: string): string[] {
    const sources = this.nodeToEdgeMap.get(target)
    if (!sources) return []
    return [...sources].filter(source => {
      const edges = this.edges.get(source)
      return edges?.some(e => e.type === type && e.target === target)
    })
  }

  /**
   * Remove edges from `source`. 'owned' edges cascade-delete their target
   * (unless another source also owns it). 'shared' edges fire `onOrphan` if
   * the target becomes disconnected (no incoming edges remain).
   */
  removeEdges(source: string, type?: string, target?: string): void {
    const edges = this.edges.get(source)
    if (!edges) return
    const removed = edges.filter(e => (!type || e.type === type) && (!target || e.target === target))
    if (removed.length === 0) return

    // 1. Cascade: delete owned targets that lose their last owner
    for (const edge of removed) {
      const ownership = getOwnership(edge.data)
      if (ownership === 'owned' && !this.hasOtherOwnedSource(edge.target, source)) {
        this.removeNode(edge.target)
      }
    }

    // 2. Clean up all removed edges from this source (including those whose
    //    targets may have been cascaded — removeNode does NOT clean incoming
    //    edges, so we must do that here).
    for (const edge of removed) {
      this.nodeToEdgeMap.get(edge.target)?.delete(source)
      this.dirtyEdges.delete(edgeId(source, edge.type, edge.target))
      this.removedEdgeIds.add(edgeId(source, edge.type, edge.target))
      this.emitChange({ type: 'edge_removed', edgeType: edge.type, source, target: edge.target })

      if (getOwnership(edge.data) === 'shared') {
        const stillConnected = this.hasOtherIncoming(edge.target, source)
        if (!stillConnected) {
          this.onOrphan?.(edge.target)
        }
      }
    }

    this.edges.set(source, edges.filter(e => !removed.includes(e)))
    if (this.edges.get(source)?.length === 0) this.edges.delete(source)
    this.schedulePersist()
  }

  // ── Query ──

  query(): GraphQuery {
    return new GraphQuery(this.nodes, this.edges, this.nodeToEdgeMap)
  }

  // ── Hot Cache ──

  protected touchHotCache(id: string): void {
    this.hotCacheOrder.delete(id)
    this.hotCacheOrder.set(id, true)
    if (++this.evictionSkipCounter % 10 === 0) {
      this.evictOldestIfOverCap()
    }
  }

  protected evictOldestIfOverCap(): void {
    while (this.hotCacheOrder.size > this.hotCacheMax) {
      const evict = this.hotCacheOrder.keys().next().value
      if (evict === undefined) break
      this.hotCacheOrder.delete(evict)
      this.cleanupNodeEdges(evict)
      this.unindexNode(evict)
      this.nodes.delete(evict)
      this.vectors.remove(evict)
      this.dirtyVectors.delete(evict)
    }
  }

  // ── Persistence ──

  async save(): Promise<void> {
    const nodes: SerializedNode[] = []
    const edges: SerializedEdge[] = []
    const vectors: Array<{ id: string; vector: number[] }> = []

    for (const node of this.nodes.values()) {
      nodes.push({
        id: node.id,
        type: node.type,
        data: node.data,
        vector: node.vector ? [...node.vector] : null,
        insertedAt: node.insertedAt,
        updatedAt: node.updatedAt,
      })
    }

    for (const [source, edgeList] of this.edges) {
      for (const e of edgeList) {
        edges.push({
          id: edgeId(source, e.type, e.target),
          source,
          target: e.target,
          type: e.type,
          data: e.data ?? null,
          createdAt: Date.now(),
        })
      }
    }

    for (const [id, vector] of this.vectors.entries()) {
      vectors.push({ id, vector })
    }

    await Promise.all([
      this.persistence.bulkPutNodes(nodes),
      this.persistence.bulkPutEdges(edges),
      this.persistence.bulkPutVectors(vectors),
    ])
  }

  async load(): Promise<void> {
    await this.warm()
  }

  protected async rebuildEdgeIndex(): Promise<void> {
    this.edges.clear()
    this.nodeToEdgeMap.clear()
    const allEdges = await this.persistence.getAllEdges()
    for (const e of allEdges) {
      if (!this.edges.has(e.source)) this.edges.set(e.source, [])
      this.edges.get(e.source)!.push({ target: e.target, type: e.type, data: e.data ?? undefined })
      if (!this.nodeToEdgeMap.has(e.target)) this.nodeToEdgeMap.set(e.target, new Set())
      this.nodeToEdgeMap.get(e.target)!.add(e.source)
    }
  }

  async warm(): Promise<void> {
    const allNodeIds = await this.persistence.allNodeIds()
    if (allNodeIds.length === 0) return

    const serialized = await this.persistence.getNodes(allNodeIds)
    for (const sn of serialized) {
      if (!this.nodes.has(sn.id)) {
        this.nodes.set(sn.id, {
          id: sn.id,
          type: sn.type,
          data: sn.data,
          vector: sn.vector ? new Float64Array(sn.vector) : undefined,
          insertedAt: sn.insertedAt,
          updatedAt: sn.updatedAt,
        })
        this.indexNode(this.nodes.get(sn.id)!)
        this.hotCacheOrder.delete(sn.id)
        this.hotCacheOrder.set(sn.id, true)
      }
    }

    await yieldToUI()

    const allVectors = await this.persistence.getAllVectors()
    for (const { id, vector } of allVectors) {
      this.vectors.add(id, vector)
    }

    await yieldToUI()

    this.evictOldestIfOverCap()

    await this.rebuildEdgeIndex()

    for (const nodeType of this._byType.keys()) {
      this.emitChange({ type: 'node_added', nodeType })
    }
  }

  async prune(maxNodes: number): Promise<void> {
    if (this.nodes.size <= maxNodes) return
    const excess = [...this.nodes.entries()]
      .sort(([, a], [, b]) => a.insertedAt - b.insertedAt)
      .slice(0, this.nodes.size - maxNodes)

    for (const [id] of excess) {
      this.removeNode(id)
    }
  }

  // ── Clear / Dispose ──

  clear(): void {
    this.nodes.clear()
    this.edges.clear()
    this.vectors.clear()
    this.hotCacheOrder.clear()
    this.nodeToEdgeMap.clear()
    this._byType.clear()
    this.dirtyEdges.clear()
    this.dirtyVectors.clear()
    this.dirtyNodes.clear()
    this.removedNodeIds.clear()
    this.removedEdgeIds.clear()
  }

  async dispose(): Promise<void> {
    if (this.persistTimer) {
      clearTimeout(this.persistTimer)
      this.persistTimer = null
    }
    this.clear()
    await this.persistence.close()
  }

  // ── Public Query API ──

  whereType(type: string): PolyNode[] {
    const ids = this._byType.get(type)
    if (!ids) return []
    const results: PolyNode[] = []
    for (const id of ids) {
      const node = this.nodes.get(id)
      if (node) results.push(node)
    }
    return results
  }

  get size(): number {
    return this.nodes.size
  }
}
