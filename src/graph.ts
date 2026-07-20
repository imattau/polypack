import { Subject } from 'rxjs'
import type { PolyNode, EdgeOwnership, GraphChangeEvent, SerializedNode, SerializedEdge, DataTransform } from './types.js'
import { VectorIndex } from './vector-index.js'
import { GraphQuery } from './query.js'
import { PersistedGraphQuery } from './persisted-query.js'
import { assertFiniteVector, cloneData, clonePolyNode, edgeId, yieldToUI } from './utils.js'
import type { PersistenceAdapter, PersistenceChanges } from './persistence/adapter.js'
import { MemoryAdapter } from './persistence/memory.js'
import { createEmbedding, defaultEmbedding } from './embedding.js'
import type { EmbeddingProvider } from './embedding.js'

type EdgeEntry = { target: string; type: string; data?: Record<string, unknown> }
type EdgeIndex = Map<string, Map<string, EdgeEntry>>

const DEFAULT_HOT_CACHE_MAX = 50000

const OWNERSHIP_KEY = '__ownership'

function getOwnership(data?: Record<string, unknown>): EdgeOwnership {
  return (data?.[OWNERSHIP_KEY] as EdgeOwnership) ?? 'reference'
}

/**
 * In-memory property graph with vector search, reactive change events, a
 * bounded node cache, and pluggable persistence.
 *
 * Call {@link warm} before querying existing persisted data and {@link dispose}
 * during shutdown. Mutations are persisted after a two-second debounce or
 * immediately with {@link flush}.
 */
export class PolyGraph {
  protected nodes = new Map<string, PolyNode>()
  protected edges: EdgeIndex = new Map()
  protected nodeToEdgeMap = new Map<string, Set<string>>()

  readonly vectors: VectorIndex
  readonly changes = new Subject<GraphChangeEvent>()
  readonly persistence: PersistenceAdapter
  readonly hotCacheMax: number
  readonly embedding: EmbeddingProvider
  readonly transform?: DataTransform

  protected sidecarData = new Map<string, unknown>()
  protected hotCacheOrder = new Map<string, true>()
  protected _byType = new Map<string, Set<string>>()
  protected evictedDirtyNodes = new Map<string, SerializedNode>()

  protected dirtyNodes = new Set<string>()
  protected dirtyEdges = new Set<string>()
  protected dirtyVectors = new Set<string>()
  protected removedVectorIds = new Set<string>()
  protected removedEdgeIds = new Set<string>()
  protected removedNodeIds = new Set<string>()
  protected _warmed = false
  protected persistTimer: ReturnType<typeof setTimeout> | null = null
  protected flushInFlight: Promise<void> | null = null
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

  constructor(
    adapter?: PersistenceAdapter,
    hotCacheMax?: number,
    embedding?: EmbeddingProvider,
    transform?: DataTransform,
    createVectorIndex?: (onChange: (id: string) => void) => VectorIndex,
  ) {
    this.persistence = adapter ?? new MemoryAdapter()
    this.hotCacheMax = hotCacheMax ?? DEFAULT_HOT_CACHE_MAX
    this.embedding = embedding ?? defaultEmbedding
    this.transform = transform
    this.vectors = (createVectorIndex ?? ((onChange) => new VectorIndex(onChange)))((id) => {
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
      this.persistTimer = null
      this.flush().catch((err) => console.warn('[PolyGraph] Flush error:', err))
    }, 2000)
  }

  async flush(): Promise<void> {
    if (this.flushInFlight) {
      await this.flushInFlight
      if (this.hasPendingPersistence()) await this.flush()
      return
    }
    this.flushInFlight = this.flushPending()
    try {
      await this.flushInFlight
    } finally {
      this.flushInFlight = null
    }
    if (this.hasPendingPersistence()) await this.flush()
    this.evictOldestIfOverCap()
  }

  protected hasPendingPersistence(): boolean {
    return this.dirtyNodes.size > 0 || this.dirtyEdges.size > 0 ||
      this.dirtyVectors.size > 0 || this.removedNodeIds.size > 0 ||
      this.removedEdgeIds.size > 0 || this.removedVectorIds.size > 0
  }

  private async flushPending(): Promise<void> {
    const dirtyNodeIds = [...this.dirtyNodes]
    const dirtyEdgeIds = [...this.dirtyEdges]
    const dirtyVectorIds = [...this.dirtyVectors]
    const removedNodeIds = [...this.removedNodeIds]
    const removedEdgeIds = [...this.removedEdgeIds]
    const removedVectorIds = [...this.removedVectorIds]
    const evictedSnapshots = new Map<string, SerializedNode>()
    for (const id of dirtyNodeIds) this.dirtyNodes.delete(id)
    for (const id of dirtyEdgeIds) this.dirtyEdges.delete(id)
    for (const id of dirtyVectorIds) this.dirtyVectors.delete(id)
    for (const id of removedNodeIds) this.removedNodeIds.delete(id)
    for (const id of removedEdgeIds) this.removedEdgeIds.delete(id)
    for (const id of removedVectorIds) this.removedVectorIds.delete(id)

    const nodesToSave: SerializedNode[] = []
    for (const id of dirtyNodeIds) {
      const node = this.nodes.get(id)
      if (node) {
        nodesToSave.push({
          id: node.id,
          type: node.type,
          data: cloneData(node.data),
          vector: node.vector ? [...node.vector] : null,
          insertedAt: node.insertedAt,
          updatedAt: node.updatedAt,
        })
      } else {
        const snapshot = this.evictedDirtyNodes.get(id)
        if (snapshot) {
          nodesToSave.push(snapshot)
          evictedSnapshots.set(id, snapshot)
          this.evictedDirtyNodes.delete(id)
        }
      }
    }

    const dirtyEdgeList: SerializedEdge[] = []
    for (const edgeIdStr of dirtyEdgeIds) {
      const parts = edgeIdStr.split('::')
      if (parts.length < 3) continue
      const source = parts[0]
      const innerKey = parts.slice(1).join('::')
      const sourceEdges = this.edges.get(source)
      const edge = sourceEdges?.get(innerKey)
      if (edge) {
        dirtyEdgeList.push({
          id: edgeIdStr,
          source,
          target: edge.target,
          type: edge.type,
          data: edge.data ? cloneData(edge.data) : null,
          createdAt: Date.now(),
        })
      }
    }
    const dirtyVectorEntries: Array<{ id: string; vector: number[] }> = []
    for (const id of dirtyVectorIds) {
      const vec = this.vectors.get(id) ?? evictedSnapshots.get(id)?.vector ?? undefined
      if (vec) dirtyVectorEntries.push({ id, vector: Array.isArray(vec) ? vec : [...vec] })
    }
    try {
      const changes: PersistenceChanges = {
        putNodes: nodesToSave,
        deleteNodeIds: removedNodeIds,
        putEdges: dirtyEdgeList,
        deleteEdgeIds: removedEdgeIds,
        putVectors: dirtyVectorEntries,
        deleteVectorIds: [...new Set([...removedNodeIds, ...removedVectorIds])],
      }
      if (this.persistence.applyChanges) {
        await this.persistence.applyChanges(changes)
      } else {
        if (removedNodeIds.length > 0) {
          await this.persistence.bulkDeleteNodes(removedNodeIds)
          await Promise.all(removedNodeIds.map(id => this.persistence.deleteVector(id)))
        }
        await this.persistence.bulkPutNodes(nodesToSave)
        await this.persistence.bulkPutEdges(dirtyEdgeList)
        await this.persistence.bulkDeleteEdges(removedEdgeIds)
        await Promise.all(removedVectorIds.map(id => this.persistence.deleteVector(id)))
        await this.persistence.bulkPutVectors(dirtyVectorEntries)
      }
    } catch (error) {
      for (const id of dirtyNodeIds) if (!this.removedNodeIds.has(id)) this.dirtyNodes.add(id)
      for (const id of dirtyEdgeIds) if (!this.removedEdgeIds.has(id)) this.dirtyEdges.add(id)
      for (const id of dirtyVectorIds) if (!this.removedNodeIds.has(id)) this.dirtyVectors.add(id)
      for (const id of removedNodeIds) this.removedNodeIds.add(id)
      for (const id of removedEdgeIds) this.removedEdgeIds.add(id)
      for (const id of removedVectorIds) if (!this.vectors.has(id)) this.removedVectorIds.add(id)
      for (const [id, snapshot] of evictedSnapshots) this.evictedDirtyNodes.set(id, snapshot)
      throw error
    }
  }

  // ── Node CRUD ──

  addNode(node: PolyNode): void {
    if (!node.id) throw new TypeError('Node id must not be empty')
    if (!node.type) throw new TypeError('Node type must not be empty')
    if (!Number.isFinite(node.insertedAt) || node.insertedAt < 0 ||
        !Number.isFinite(node.updatedAt) || node.updatedAt < 0) {
      throw new RangeError('Node timestamps must be finite non-negative numbers')
    }
    if (node.vector) assertFiniteVector(node.vector)
    const serializedData = this.applySerialize(node.id, node.data as Record<string, unknown>)
    const stored = clonePolyNode({ ...node, data: serializedData })
    const previous = this.nodes.get(stored.id)
    if (previous) {
      this.unindexNode(stored.id)
    }
    if (!stored.vector) {
      this.vectors.remove(stored.id)
      this.dirtyVectors.delete(stored.id)
      this.removedVectorIds.add(stored.id)
    }
    this.removedNodeIds.delete(stored.id)
    this.nodes.set(stored.id, stored)
    this.touchHotCache(stored.id)
    this.markDirty(stored.id)
    this.indexNode(stored)
    if (stored.vector) {
      this.removedVectorIds.delete(stored.id)
      this.vectors.add(stored.id, stored.vector)
    }
    this.emitChange({ type: 'node_added', nodeId: stored.id, nodeType: stored.type })
  }

  /** Embed `text` with the configured provider and add the resulting node. */
  async addNodeWithEmbedding(node: Omit<PolyNode, 'vector'>, text: string): Promise<void> {
    const vector = await this.embed(text)
    this.addNode({ ...node, vector })
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

  protected applySerialize(id: string, data: Record<string, unknown>): Record<string, unknown> {
    if (!this.transform?.serialize) return data
    const result = this.transform.serialize(data)
    if (result.sidecar !== undefined) {
      this.sidecarData.set(id, result.sidecar)
    }
    return result.data
  }

  protected applyDeserialize(node: PolyNode): PolyNode {
    if (!this.transform?.deserialize) return node
    const sidecar = this.sidecarData.get(node.id)
    return {
      ...node,
      data: this.transform.deserialize(node.data, sidecar) as Record<string, unknown>,
    }
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
      return this.applyDeserialize(clonePolyNode(node))
    }
    return undefined
  }

  async getNodeSafe(id: string): Promise<PolyNode | undefined> {
    if (this.removedNodeIds.has(id)) return undefined
    const node = this.nodes.get(id)
    if (node) {
      this.touchHotCache(id)
      return clonePolyNode(node)
    }
    const serialized = this.evictedDirtyNodes.get(id) ?? await this.persistence.getNode(id)
    if (!serialized) return undefined
    const restored: PolyNode = {
      id: serialized.id,
      type: serialized.type,
      data: cloneData(serialized.data),
      vector: serialized.vector ? new Float64Array(serialized.vector) : undefined,
      insertedAt: serialized.insertedAt,
      updatedAt: serialized.updatedAt,
    }
    this.nodes.set(id, restored)
    this.evictedDirtyNodes.delete(id)
    this.indexNode(restored)
    this.touchHotCache(id)
    if (restored.vector) {
      this.vectors.hydrate(restored.id, restored.vector)
    }
    return this.applyDeserialize(clonePolyNode(restored))
  }

  updateNode(id: string, data: Partial<Record<string, unknown>>, vector?: Float64Array): PolyNode | undefined {
    const node = this.nodes.get(id)
    if (!node) return undefined
    const serialized = this.applySerialize(id, data as Record<string, unknown>)
    Object.assign(node.data, cloneData(serialized))
    if (vector !== undefined) {
      assertFiniteVector(vector)
      node.vector = new Float64Array(vector)
      this.removedVectorIds.delete(id)
      this.vectors.add(id, [...vector])
      this.dirtyVectors.add(id)
    }
    node.updatedAt = Date.now()
    this.touchHotCache(id)
    this.markDirty(id)
    this.emitChange({ type: 'node_updated', nodeId: id, nodeType: node.type })
    return this.applyDeserialize(clonePolyNode(node))
  }

  /** Embed `text` and update a loaded node's data and vector together. */
  async updateNodeWithEmbedding(
    id: string,
    data: Partial<Record<string, unknown>>,
    text: string,
  ): Promise<PolyNode | undefined> {
    const vector = await this.embed(text)
    return this.updateNode(id, data, vector)
  }

  /** Embed `text`, restoring an evicted node before updating when necessary. */
  async updateNodeSafeWithEmbedding(
    id: string,
    data: Partial<Record<string, unknown>>,
    text: string,
  ): Promise<PolyNode | undefined> {
    const vector = await this.embed(text)
    return this.updateNodeSafe(id, data, vector)
  }

  /** Remove a node's vector while keeping the node and its data. */
  removeNodeVector(id: string): PolyNode | undefined {
    const node = this.nodes.get(id)
    if (!node) return undefined
    node.vector = undefined
    this.vectors.remove(id)
    this.dirtyVectors.delete(id)
    this.removedVectorIds.add(id)
    node.updatedAt = Date.now()
    this.touchHotCache(id)
    this.markDirty(id)
    this.emitChange({ type: 'node_updated', nodeId: id, nodeType: node.type })
    return this.applyDeserialize(clonePolyNode(node))
  }

  /** Restore an evicted node when necessary, then remove its vector. */
  async removeNodeVectorSafe(id: string): Promise<PolyNode | undefined> {
    if (!await this.getNodeSafe(id)) return undefined
    return this.removeNodeVector(id)
  }

  /** Restore an evicted node when necessary, then update it. */
  async updateNodeSafe(
    id: string,
    data: Partial<Record<string, unknown>>,
    vector?: Float64Array,
  ): Promise<PolyNode | undefined> {
    const node = await this.getNodeSafe(id)
    if (!node) return undefined
    return this.updateNode(id, data, vector)
  }

  /** Returns true if `target` has at least one incoming 'owned' edge from a
   *  source other than `excludeSource`. */
  protected hasOtherOwnedSource(target: string, excludeSource: string): boolean {
    const sources = this.nodeToEdgeMap.get(target)
    if (!sources) return false
    for (const src of sources) {
      if (src === excludeSource) continue
      const srcEdges = this.edges.get(src)
      if (srcEdges) {
        for (const e of srcEdges.values()) {
          if (e.target === target && getOwnership(e.data) === 'owned') return true
        }
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
      if (srcEdges) {
        for (const e of srcEdges.values()) {
          if (e.target === target) return true
        }
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
    const outgoing = [...(this.edges.get(id)?.values() ?? [])]
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
    this.removedVectorIds.delete(id)

    this.dirtyNodes.delete(id)
    this.removedNodeIds.add(id)
    this.sidecarData.delete(id)
    this.schedulePersist()
    this.hotCacheOrder.delete(id)
    this.emitChange({ type: 'node_removed', nodeId: id, nodeType: node.type })
  }

  /**
   * Restore and remove a node even when it has been evicted. Owned descendants
   * are restored and removed recursively; another owning source keeps a target
   * alive. Existing persisted graphs must be warmed first so edge ownership is
   * available in the in-memory edge index.
   */
  async removeNodeSafe(id: string): Promise<boolean> {
    return this.removeNodeSafeRecursive(id, new Set<string>())
  }

  private async removeNodeSafeRecursive(id: string, visited: Set<string>): Promise<boolean> {
    if (visited.has(id)) return false
    visited.add(id)

    const node = await this.getNodeSafe(id)
    if (!node) return false

    const outgoing = [...(this.edges.get(id)?.values() ?? [])]
    for (const edge of outgoing) {
      if (getOwnership(edge.data) === 'owned' && !this.hasOtherOwnedSource(edge.target, id)) {
        await this.removeNodeSafeRecursive(edge.target, visited)
      }
    }

    if (this.removedNodeIds.has(id)) return true

    // Recursive restoration can evict this node again when the working set is
    // very small, so restore it once more before using the synchronous remover.
    if (!this.nodes.has(id)) await this.getNodeSafe(id)
    if (!this.nodes.has(id)) return false
    this.removeNode(id)
    return true
  }

  protected cleanupNodeEdges(id: string): void {
    const incomingSources = [...(this.nodeToEdgeMap.get(id) ?? [])]
    for (const source of incomingSources) {
      const sourceEdges = this.edges.get(source)
      if (!sourceEdges) continue
      for (const [key, edge] of sourceEdges) {
        if (edge.target === id) {
          sourceEdges.delete(key)
          this.recordRemovedEdge(source, edge)
        }
      }
      if (sourceEdges.size === 0) this.edges.delete(source)
    }
    const sourceEdges = this.edges.get(id)
    if (sourceEdges) {
      for (const edge of sourceEdges.values()) {
        this.nodeToEdgeMap.get(edge.target)?.delete(id)
        this.recordRemovedEdge(id, edge)
      }
      this.edges.delete(id)
    }
    this.nodeToEdgeMap.delete(id)
  }

  private recordRemovedEdge(source: string, edge: EdgeEntry): void {
    const id = edgeId(source, edge.type, edge.target)
    this.dirtyEdges.delete(id)
    this.removedEdgeIds.add(id)
    this.emitChange({ type: 'edge_removed', edgeType: edge.type, source, target: edge.target })
  }

  // ── Edge CRUD ──

  addEdge(source: string, type: string, target: string, data?: Record<string, unknown>, ownership?: EdgeOwnership): void {
    if (!source || !type || !target) throw new TypeError('Edge source, type, and target must not be empty')
    const id = edgeId(source, type, target)
    this.removedEdgeIds.delete(id)
    const inner = id.slice(source.length + 2)
    if (!this.edges.has(source)) this.edges.set(source, new Map())
    const sourceEdges = this.edges.get(source)!
    if (sourceEdges.has(inner)) return

    const inputData = data === undefined ? undefined : cloneData(data)
    const fullData = ownership !== undefined ? { ...inputData, [OWNERSHIP_KEY]: ownership } : inputData

    sourceEdges.set(inner, { target, type, data: fullData })
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

  getEdges(source: string, type?: string): EdgeEntry[] {
    const edges = this.edges.get(source)
    if (!edges) return []
    const all = [...edges.values()]
    const selected = type ? all.filter(e => e.type === type) : all
    return selected.map(edge => ({ ...edge, data: edge.data ? cloneData(edge.data) : undefined }))
  }

  getEdgeTargets(source: string, type: string): string[] {
    const edges = this.edges.get(source)
    if (!edges) return []
    return [...edges.values()].filter(e => e.type === type).map(e => e.target)
  }

  getEdgeSources(target: string, type: string): string[] {
    const sources = this.nodeToEdgeMap.get(target)
    if (!sources) return []
    return [...sources].filter(source => {
      const edges = this.edges.get(source)
      if (!edges) return false
      for (const e of edges.values()) {
        if (e.type === type && e.target === target) return true
      }
      return false
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
    const removed: EdgeEntry[] = []
    for (const edge of edges.values()) {
      if ((!type || edge.type === type) && (!target || edge.target === target)) {
        removed.push(edge)
      }
    }
    if (removed.length === 0) return

    for (const edge of removed) {
      const ownership = getOwnership(edge.data)
      if (ownership === 'owned' && !this.hasOtherOwnedSource(edge.target, source)) {
        this.removeNode(edge.target)
      }
    }

    const removeAll = !type && !target
    for (const edge of removed) {
      if (!removeAll) edges.delete(`${edge.type}::${edge.target}`)
      this.nodeToEdgeMap.get(edge.target)?.delete(source)
      const id = edgeId(source, edge.type, edge.target)
      this.dirtyEdges.delete(id)
      this.removedEdgeIds.add(id)
      this.emitChange({ type: 'edge_removed', edgeType: edge.type, source, target: edge.target })

      if (getOwnership(edge.data) === 'shared') {
        const stillConnected = this.hasOtherIncoming(edge.target, source)
        if (!stillConnected) {
          this.onOrphan?.(edge.target)
        }
      }
    }

    if (removeAll) this.edges.delete(source)
    else if (edges.size === 0) this.edges.delete(source)
    this.schedulePersist()
  }

  // ── Query ──

  query(): GraphQuery {
    return new GraphQuery(this.nodes, this.edges, this.nodeToEdgeMap, this.transform, this.sidecarData)
  }

  /** Generate a detached embedding using the graph's configured provider. */
  async embed(text: string): Promise<Float64Array> {
    return createEmbedding(this.embedding, text)
  }

  /** Start an in-memory similarity query from text using the configured provider. */
  async queryText(text: string, threshold = 0, topK?: number): Promise<GraphQuery> {
    return this.query().similarTo([...(await this.embed(text))], threshold, topK)
  }

  /** Start a complete persisted similarity query from embedded text. */
  async queryPersistedText(text: string, threshold = 0, topK?: number): Promise<PersistedGraphQuery> {
    return this.queryPersisted().similarTo([...(await this.embed(text))], threshold, topK)
  }

  /** Query all persisted nodes without loading them into the hot working set. */
  queryPersisted(): PersistedGraphQuery {
    return new PersistedGraphQuery(this.persistence, this.transform, this.sidecarData)
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
      const node = this.nodes.get(evict)
      if (node && this.dirtyNodes.has(evict)) {
        this.evictedDirtyNodes.set(evict, {
          id: node.id,
          type: node.type,
          data: cloneData(node.data),
          vector: node.vector ? [...node.vector] : null,
          insertedAt: node.insertedAt,
          updatedAt: node.updatedAt,
        })
      }
      this.hotCacheOrder.delete(evict)
      this.unindexNode(evict)
      this.nodes.delete(evict)
      this.vectors.remove(evict)
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
        data: cloneData(node.data),
        vector: node.vector ? [...node.vector] : null,
        insertedAt: node.insertedAt,
        updatedAt: node.updatedAt,
      })
    }

    for (const [source, edgeList] of this.edges) {
      for (const e of edgeList.values()) {
        edges.push({
          id: edgeId(source, e.type, e.target),
          source,
          target: e.target,
          type: e.type,
          data: e.data ? cloneData(e.data) : null,
          createdAt: Date.now(),
        })
      }
    }

    for (const [id, vector] of this.vectors.entries()) {
      vectors.push({ id, vector: [...vector] })
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
      if (e.id !== edgeId(e.source, e.type, e.target)) {
        throw new Error(`Invalid persisted edge ID: ${e.id}`)
      }
      const inner = e.id.slice(e.source.length + 2)
      if (!this.edges.has(e.source)) this.edges.set(e.source, new Map())
      this.edges.get(e.source)!.set(inner, {
        target: e.target,
        type: e.type,
        data: e.data ? cloneData(e.data) : undefined,
      })
      if (!this.nodeToEdgeMap.has(e.target)) this.nodeToEdgeMap.set(e.target, new Set())
      this.nodeToEdgeMap.get(e.target)!.add(e.source)
    }
  }

  async warm(): Promise<void> {
    if (this._warmed) return
    this._warmed = true
    const allNodeIds = await this.persistence.allNodeIds()
    if (allNodeIds.length === 0) return

    const loadCount = Math.min(allNodeIds.length, this.hotCacheMax)
    const hotIds = allNodeIds.slice(allNodeIds.length - loadCount)
    const serialized = await this.persistence.getNodes(hotIds)
    for (const sn of serialized) {
      if (!this.nodes.has(sn.id)) {
        this.nodes.set(sn.id, {
          id: sn.id,
          type: sn.type,
          data: cloneData(sn.data),
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

    const hotNodeIds = new Set(hotIds)
    if (this.persistence.getVectors) {
      const vectors = await this.persistence.getVectors(hotIds)
      for (const { id, vector } of vectors) {
        this.vectors.hydrate(id, vector)
      }
    } else {
      const allVectors = await this.persistence.getAllVectors()
      for (const { id, vector } of allVectors) {
        if (hotNodeIds.has(id)) {
          this.vectors.hydrate(id, vector)
        }
      }
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
    if (this.persistTimer) {
      clearTimeout(this.persistTimer)
      this.persistTimer = null
    }
    this.nodes.clear()
    this.edges.clear()
    this.vectors.clear()
    this.hotCacheOrder.clear()
    this.nodeToEdgeMap.clear()
    this._byType.clear()
    this.evictedDirtyNodes.clear()
    this.dirtyEdges.clear()
    this.dirtyVectors.clear()
    this.dirtyNodes.clear()
    this.removedNodeIds.clear()
    this.removedEdgeIds.clear()
    this.removedVectorIds.clear()
    this._warmed = false
  }

  async dispose(): Promise<void> {
    if (this.persistTimer) {
      clearTimeout(this.persistTimer)
      this.persistTimer = null
    }
    await this.flush()
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
      if (node) results.push(this.applyDeserialize(clonePolyNode(node)))
    }
    return results
  }

  get size(): number {
    return this.nodes.size
  }

  /** Number of nodes in the loaded working set. */
  get loadedSize(): number {
    return this.nodes.size
  }

  /** Whether a node is currently present in the loaded working set. */
  hasLoadedNode(id: string): boolean {
    return this.nodes.has(id)
  }

  /** Number of nodes currently stored by the persistence adapter. */
  async persistedSize(): Promise<number> {
    return (await this.persistence.allNodeIds()).length
  }

  // ── Graph traversal ──

  /**
   * Walk ancestor chain backwards through incoming edges of `edgeType`.
   * Returns nodes from root ancestor to start node (inclusive). Detects cycles.
   */
  walkAncestors(id: string, edgeType: string): PolyNode[] {
    const path: PolyNode[] = []
    const seen = new Set<string>()
    let current: string | undefined = id
    while (current && !seen.has(current)) {
      seen.add(current)
      const node = this.nodes.get(current)
      if (!node) break
      path.unshift(this.applyDeserialize(clonePolyNode(node)))
      const sources = this.getEdgeSources(current, edgeType)
      current = sources.length > 0 ? sources[0] : undefined
    }
    return path
  }

  /**
   * Walk descendant chain forwards through outgoing edges of `edgeType`.
   * Returns nodes from start node to deepest child (inclusive). Detects cycles.
   */
  walkDescendants(id: string, edgeType: string): PolyNode[] {
    const path: PolyNode[] = []
    const seen = new Set<string>()
    let current: string | undefined = id
    while (current && !seen.has(current)) {
      seen.add(current)
      const node = this.nodes.get(current)
      if (!node) break
      path.push(this.applyDeserialize(clonePolyNode(node)))
      const targets = this.getEdgeTargets(current, edgeType)
      current = targets.length > 0 ? targets[0] : undefined
    }
    return path
  }

  // ── Full-text search ──

  /**
   * Quick full-text search across persisted nodes of a single type.
   * Shorthand for `queryPersistedText(text, threshold, topK).whereNodeType(type).toArray()`.
   */
  async searchNodes<T = PolyNode>(text: string, type: string, threshold = 0, topK?: number): Promise<T[]> {
    if (!text.trim()) return []
    return (await this.queryPersistedText(text, threshold, topK).then(q => q.whereNodeType(type).toArray())) as unknown as T[]
  }

  // ── Convenience queries ──

  /** Load all persisted nodes of a type into the hot cache. */
  async getNodesByType<T = PolyNode>(type: string): Promise<T[]> {
    return (await this.queryPersisted().whereNodeType(type).toArray()) as unknown as T[]
  }

  /** Load all persisted nodes of a type ordered by a data field. */
  async getNodesByTypeOrdered<T = PolyNode>(type: string, field: string, direction: 'asc' | 'desc' = 'desc'): Promise<T[]> {
    return (await this.queryPersisted().whereNodeType(type).orderBy(field, direction).toArray()) as unknown as T[]
  }

  /** Count persisted nodes of a type. */
  async countNodesByType(type: string): Promise<number> {
    return this.queryPersisted().whereNodeType(type).count()
  }

  /** Remove all persisted nodes of a type. */
  async deleteNodesByType(type: string): Promise<void> {
    const ids = await this.queryPersisted().whereNodeType(type).ids()
    for (const id of ids) {
      await this.removeNodeSafe(id)
    }
  }
}
