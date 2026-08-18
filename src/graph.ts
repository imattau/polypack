import { Subject } from 'rxjs'
import type { PolyNode, PolyEdge, EdgeOwnership, GraphChangeEvent, SerializedNode, SerializedEdge, DataTransform, NodeActivation, WriteOptions, NodePatch, GraphTransaction, IndexDefinition, GraphStats, QueryResourceLimits, NodeTypeDefinition, EdgeTypeDefinition, EdgeCardinality, QueryMetrics, GraphResourceLimits, TransactionOptions, VerificationReport } from './types.js'
import { ConflictError } from './errors.js'
import { UniqueConstraintError } from './index-errors.js'
import { SchemaValidationError } from './schema-errors.js'
import { ResourceLimitError } from './resource-errors.js'
import { AdapterCapabilityError } from './capability-errors.js'
import { VectorIndex } from './vector-index.js'
import type { VectorIndexLike } from './vector-index.js'
import { GraphQuery } from './query.js'
import { PersistedGraphQuery } from './persisted-query.js'
import { MigrationRegistry } from './migrations.js'
import {
  ACTIVATION_DEFAULTS,
  activationScoreOf,
  assertFiniteVector,
  cloneData,
  clonePolyNode,
  decayActivationState,
  edgeId,
  reinforceActivation,
  yieldToUI,
} from './utils.js'
import type { PersistenceAdapter, PersistenceChanges } from './persistence/adapter.js'
import type { FileIO } from './persistence/file-io.js'
import { MemoryAdapter } from './persistence/memory.js'
import { createEmbedding, defaultEmbedding } from './embedding.js'
import type { EmbeddingProvider } from './embedding.js'

type EdgeEntry = { id: string; target: string; type: string; data?: Record<string, unknown>; revision: number; createdAt: number }
type EdgeIndex = Map<string, Map<string, EdgeEntry>>

const DEFAULT_HOT_CACHE_MAX = 50000

const OWNERSHIP_KEY = '__ownership'

function getOwnership(data?: Record<string, unknown>): EdgeOwnership {
  return (data?.[OWNERSHIP_KEY] as EdgeOwnership) ?? 'reference'
}

/** A stable, read-only view of graph state captured at one point in time. */
export class GraphSnapshot {
  constructor(
    private readonly nodes: Map<string, PolyNode>,
    private readonly edges: EdgeIndex,
    private readonly nodeToEdgeMap: Map<string, Set<string>>,
    private readonly transform?: DataTransform,
    private readonly sidecarData: Map<string, unknown> = new Map(),
    private readonly indexes: IndexDefinition[] = [],
  ) {}

  query(): GraphQuery {
    return new GraphQuery(this.nodes, this.edges, this.nodeToEdgeMap, this.transform, this.sidecarData, this.indexes)
  }

  getNode(id: string): PolyNode | undefined {
    const node = this.nodes.get(id)
    if (!node) return undefined
    const copy = clonePolyNode(node)
    if (this.transform?.deserialize) {
      copy.data = this.transform.deserialize(copy.data, this.sidecarData.get(id)) as Record<string, unknown>
    }
    return copy
  }

  getNodes(): PolyNode[] {
    return [...this.nodes.keys()].map(id => this.getNode(id)).filter((node): node is PolyNode => node !== undefined)
  }

  getEdges(source?: string, type?: string): PolyEdge[] {
    const sources = source === undefined ? [...this.edges.keys()] : [source]
    const result: PolyEdge[] = []
    for (const currentSource of sources) {
      for (const edge of this.edges.get(currentSource)?.values() ?? []) {
        if (type !== undefined && edge.type !== type) continue
        result.push({
          id: edge.id,
          source: currentSource,
          target: edge.target,
          type: edge.type,
          data: edge.data ? cloneData(edge.data) : undefined,
          createdAt: edge.createdAt,
          revision: edge.revision,
        })
      }
    }
    return result
  }
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

  /** The vector engine (default `VectorIndex`, or the `createVectorIndex` override). Mutating it directly bypasses persistence tracking — call {@link markVectorDirty} afterwards. */
  readonly vectors: VectorIndexLike
  /** Emits a {@link GraphChangeEvent} per mutation, coalesced by {@link startBatch}/{@link endBatch}. */
  readonly changes = new Subject<GraphChangeEvent>()
  readonly persistence: PersistenceAdapter
  readonly hotCacheMax: number
  readonly embedding: EmbeddingProvider
  readonly transform?: DataTransform
  private resourceLimits: GraphResourceLimits = {}
  readonly migrations = new MigrationRegistry()

  protected sidecarData = new Map<string, unknown>()
  protected hotCacheOrder = new Map<string, true>()
  protected _byType = new Map<string, Set<string>>()
  private secondaryIndexes = new Map<string, IndexDefinition>()
  private secondaryIndexData = new Map<string, Map<string, Set<string>>>()
  private nodeTypeDefinitions = new Map<string, NodeTypeDefinition>()
  private edgeTypeDefinitions = new Map<string, EdgeTypeDefinition>()
  private queryCount = 0
  private queryDurationMs = 0
  private queryScannedRecords = 0
  private queryIndexUsage = new Map<string, number>()
  private indexDefinitionsDirty = false
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
  private transactionTail: Promise<void> = Promise.resolve()
  private transactionActive = false
  private transactionMutationDepth = 0
  private transactionMutationCount = 0
  private currentTransactionId: string | undefined
  private currentTransactionOptions: TransactionOptions | undefined

  protected batchDepth = 0
  protected pendingBatchEvents: GraphChangeEvent[] = []

  /** Restore a binary backup into a destination FileIO before opening a graph. */
  static async restore(source: FileIO, destination: FileIO): Promise<void> {
    const { BinaryStoreAdapter } = await import('./persistence/binary-store.js')
    await BinaryStoreAdapter.restore(source, destination)
  }

  /** Run a group of graph mutations as one isolated, atomic commit. */
  async transaction<T>(callback: (tx: GraphTransaction) => Promise<T> | T, options?: TransactionOptions): Promise<T> {
    if (this.transactionActive) throw new Error('Nested transactions are not supported')
    if (this.persistence.capabilities && (!this.persistence.capabilities.atomicBatches || !this.persistence.capabilities.transactions)) {
      throw new AdapterCapabilityError(!this.persistence.capabilities.atomicBatches ? 'atomicBatches' : 'transactions')
    }
    let release!: () => void
    const wait = this.transactionTail
    this.transactionTail = new Promise<void>(resolve => { release = resolve })
    await wait
    this.transactionActive = true
    this.transactionMutationCount = 0
    const snapshot = this.captureTransactionState()
    this.startBatch()
    try {
      const id = `tx-${Date.now()}-${Math.random().toString(36).slice(2)}`
      this.currentTransactionId = id
      this.currentTransactionOptions = options
      const inTransaction = <R>(fn: () => R): R => {
        if (this.resourceLimits.maxBatchSize !== undefined && this.transactionMutationCount >= this.resourceLimits.maxBatchSize) {
          throw new ResourceLimitError('maxBatchSize', this.resourceLimits.maxBatchSize)
        }
        this.transactionMutationCount++
        this.transactionMutationDepth++
        try { return fn() } finally { this.transactionMutationDepth-- }
      }
      const tx: GraphTransaction = {
        id,
        addNode: (node, options) => inTransaction(() => this.addNode(node, options)),
        addEdge: (edge, options) => inTransaction(() => this.addEdge(edge, options)),
        updateEdge: (edgeId, data, options) => inTransaction(() => this.updateEdge(edgeId, data, options)),
        updateNode: (nodeId, data, options) => inTransaction(() => this.updateNode(nodeId, data, undefined, undefined, options)),
        patchNode: (nodeId, patch, options) => inTransaction(() => this.patchNode(nodeId, patch, options)),
        removeNode: (nodeId, options) => inTransaction(() => this.removeNode(nodeId, options)),
        removeEdge: (edgeId, options) => inTransaction(() => this.removeEdge(edgeId, options)),
        getNode: id => this.getNode(id),
        query: () => this.query(),
      }
      const result = await callback(tx)
      if (this.persistTimer) { clearTimeout(this.persistTimer); this.persistTimer = null }
      await this.flush()
      this.endBatch()
      return result
    } catch (error) {
      this.restoreTransactionState(snapshot)
      this.pendingBatchEvents = []
      this.batchDepth = 0
      throw error
    } finally {
      this.transactionActive = false
      this.transactionMutationCount = 0
      this.currentTransactionId = undefined
      this.currentTransactionOptions = undefined
      release()
    }
  }

  private assertMutationAllowed(): void {
    if (this.transactionActive && this.transactionMutationDepth === 0) {
      throw new Error('A transaction is active; mutate through its transaction context')
    }
  }

  /** Return the persistence guarantees declared by the active adapter. */
  get adapterCapabilities() {
    return this.persistence.capabilities ? { ...this.persistence.capabilities } : undefined
  }

  /** Reject a configuration unless the active adapter declares each capability. */
  requireAdapterCapabilities(required: Partial<import('./types.js').AdapterCapabilities>): void {
    const capabilities = this.persistence.capabilities
    if (!capabilities) throw new AdapterCapabilityError('declared capabilities')
    for (const [name, expected] of Object.entries(required)) {
      if (capabilities[name as keyof typeof capabilities] !== expected) throw new AdapterCapabilityError(name)
    }
  }

  /** Configure write-side resource limits. Omitted values remain unlimited. */
  setResourceLimits(limits: GraphResourceLimits): void {
    for (const [name, value] of Object.entries(limits)) {
      if (value !== undefined && (!Number.isInteger(value) || value < 1)) throw new RangeError(`${name} must be a positive integer`)
    }
    this.resourceLimits = { ...limits }
  }

  get resourceLimitConfig(): GraphResourceLimits {
    return { ...this.resourceLimits }
  }

  private checkNodeResources(node: PolyNode, data: Record<string, unknown>): void {
    const maxDimensions = this.resourceLimits.maxVectorDimensions
    if (maxDimensions !== undefined && node.vector && node.vector.length > maxDimensions) {
      throw new ResourceLimitError('maxVectorDimensions', maxDimensions)
    }
    const maxPayload = this.resourceLimits.maxNodePayloadBytes
    if (maxPayload !== undefined) {
      let bytes = 0
      try { bytes = new TextEncoder().encode(JSON.stringify(data)).length } catch { bytes = maxPayload + 1 }
      if (bytes > maxPayload) throw new ResourceLimitError('maxNodePayloadBytes', maxPayload)
    }
  }

  private toPolyEdge(source: string, edge: EdgeEntry): PolyEdge {
    return {
      id: edge.id,
      source,
      target: edge.target,
      type: edge.type,
      data: edge.data ? cloneData(edge.data) : undefined,
      createdAt: edge.createdAt,
      revision: edge.revision,
    }
  }

  /** Register optional validation and constraints for a node type. */
  registerNodeType(type: string, definition: NodeTypeDefinition = {}): void {
    if (!type) throw new TypeError('Node type must not be empty')
    const previous = this.nodeTypeDefinitions.get(type)
    this.nodeTypeDefinitions.set(type, {
      ...definition,
      indexes: definition.indexes ? [...definition.indexes] : undefined,
      requiredFields: definition.requiredFields ? [...definition.requiredFields] : undefined,
      dataTypes: definition.dataTypes ? { ...definition.dataTypes } : undefined,
    })
    try {
      for (const node of this.nodes.values()) if (node.type === type) this.validateNodeSchema(node)
    } catch (error) {
      if (previous) this.nodeTypeDefinitions.set(type, previous)
      else this.nodeTypeDefinitions.delete(type)
      throw error
    }
    for (const field of definition.indexes ?? []) {
      this.defineIndex({ name: `${type}.${field}`, nodeType: type, fields: [field] })
    }
  }

  /** Register optional endpoint, cardinality, and validation constraints for an edge type. */
  registerEdgeType(type: string, definition: EdgeTypeDefinition = {}): void {
    if (!type) throw new TypeError('Edge type must not be empty')
    const previous = this.edgeTypeDefinitions.get(type)
    const normalized = {
      ...definition,
      sourceTypes: definition.sourceTypes ? [...definition.sourceTypes] : undefined,
      targetTypes: definition.targetTypes ? [...definition.targetTypes] : undefined,
      requiredFields: definition.requiredFields ? [...definition.requiredFields] : undefined,
      dataTypes: definition.dataTypes ? { ...definition.dataTypes } : undefined,
    }
    this.edgeTypeDefinitions.set(type, normalized)
    try {
      for (const [source, entries] of this.edges) {
        for (const edge of entries.values()) {
          if (edge.type === type) this.validateEdgeSchema(this.toPolyEdge(source, edge))
        }
      }
    } catch (error) {
      if (previous) this.edgeTypeDefinitions.set(type, previous)
      else this.edgeTypeDefinitions.delete(type)
      throw error
    }
  }

  get nodeTypes(): ReadonlyMap<string, NodeTypeDefinition> {
    return new Map([...this.nodeTypeDefinitions].map(([type, definition]) => [type, {
      ...definition,
      indexes: definition.indexes ? [...definition.indexes] : undefined,
      requiredFields: definition.requiredFields ? [...definition.requiredFields] : undefined,
      dataTypes: definition.dataTypes ? { ...definition.dataTypes } : undefined,
    }]))
  }

  get edgeTypes(): ReadonlyMap<string, EdgeTypeDefinition> {
    return new Map([...this.edgeTypeDefinitions].map(([type, definition]) => [type, {
      ...definition,
      sourceTypes: definition.sourceTypes ? [...definition.sourceTypes] : undefined,
      targetTypes: definition.targetTypes ? [...definition.targetTypes] : undefined,
      requiredFields: definition.requiredFields ? [...definition.requiredFields] : undefined,
      dataTypes: definition.dataTypes ? { ...definition.dataTypes } : undefined,
    }]))
  }

  private validateNodeSchema(node: PolyNode): void {
    const definition = this.nodeTypeDefinitions.get(node.type)
    if (!definition) return
    for (const field of definition.requiredFields ?? []) {
      if (this.indexValue(node, field) === undefined) throw new SchemaValidationError('node', node.type, `required field ${field} is missing`)
    }
    for (const [field, expected] of Object.entries(definition.dataTypes ?? {})) {
      const value = this.indexValue(node, field)
      if (value === undefined) continue
      const actual = Array.isArray(value) ? 'array' : value === null ? 'null' : typeof value
      const valid = expected === 'integer'
        ? typeof value === 'number' && Number.isInteger(value)
        : actual === expected
      if (!valid) throw new SchemaValidationError('node', node.type, `field ${field} must be ${expected}`)
    }
    if (!definition.validate) return
    try {
      if (definition.validate(clonePolyNode(node)) === false) throw new Error('custom validator returned false')
    } catch (error) {
      if (error instanceof SchemaValidationError) throw error
      throw new SchemaValidationError('node', node.type, error instanceof Error ? error.message : String(error))
    }
  }

  private validateEdgeSchema(edge: PolyEdge): void {
    const definition = this.edgeTypeDefinitions.get(edge.type)
    if (!definition) return
    const source = this.nodes.get(edge.source)
    const target = this.nodes.get(edge.target)
    if (!source || !target) throw new SchemaValidationError('edge', edge.type, 'source and target nodes must exist')
    if (definition.sourceTypes && !definition.sourceTypes.includes(source.type)) {
      throw new SchemaValidationError('edge', edge.type, `source type ${source.type} is not permitted`)
    }
    if (definition.targetTypes && !definition.targetTypes.includes(target.type)) {
      throw new SchemaValidationError('edge', edge.type, `target type ${target.type} is not permitted`)
    }
    for (const field of definition.requiredFields ?? []) {
      if (this.dataValue(edge.data, field) === undefined) throw new SchemaValidationError('edge', edge.type, `required field ${field} is missing`)
    }
    for (const [field, expected] of Object.entries(definition.dataTypes ?? {})) {
      const value = this.dataValue(edge.data, field)
      if (value === undefined) continue
      const actual = Array.isArray(value) ? 'array' : value === null ? 'null' : typeof value
      const valid = expected === 'integer' ? typeof value === 'number' && Number.isInteger(value) : actual === expected
      if (!valid) throw new SchemaValidationError('edge', edge.type, `field ${field} must be ${expected}`)
    }
    this.validateCardinality(edge, definition.cardinality)
    if (definition.validate) {
      try {
        if (definition.validate({ ...edge, data: edge.data ? cloneData(edge.data) : undefined }) === false) {
          throw new Error('custom validator returned false')
        }
      } catch (error) {
        if (error instanceof SchemaValidationError) throw error
        throw new SchemaValidationError('edge', edge.type, error instanceof Error ? error.message : String(error))
      }
    }
  }

  private validateCardinality(edge: PolyEdge, cardinality?: EdgeCardinality): void {
    if (!cardinality || cardinality === 'many-to-many') return
    let outgoing = 0
    let incoming = 0
    for (const [candidateSource, entries] of this.edges) {
      for (const candidate of entries.values()) {
        if (candidate.id === edge.id || candidate.type !== edge.type) continue
        if (candidateSource === edge.source) outgoing++
        if (candidate.target === edge.target) incoming++
      }
    }
    const violates = cardinality === 'one-to-one'
      ? outgoing > 0 || incoming > 0
      : cardinality === 'one-to-many'
        ? incoming > 0
        : outgoing > 0
    if (violates) throw new SchemaValidationError('edge', edge.type, `cardinality ${cardinality} would be exceeded`)
  }

  /** Define or replace a node-data index and validate all currently loaded records. */
  defineIndex(definition: IndexDefinition): void {
    if (!definition.name || !definition.fields.length) throw new TypeError('Index name and fields must not be empty')
    if (new Set(definition.fields).size !== definition.fields.length) throw new TypeError('Index fields must be unique')
    const normalized = { ...definition, fields: [...definition.fields] }
    const previous = this.secondaryIndexes.get(definition.name)
    this.secondaryIndexes.set(definition.name, normalized)
    this.secondaryIndexData.set(definition.name, new Map())
    try {
      for (const node of this.nodes.values()) this.indexSecondaryNode(node)
    } catch (error) {
      if (previous) {
        this.secondaryIndexes.set(definition.name, previous)
        this.secondaryIndexData.set(definition.name, new Map())
        for (const node of this.nodes.values()) this.indexSecondaryNode(node)
      } else {
        this.secondaryIndexes.delete(definition.name)
        this.secondaryIndexData.delete(definition.name)
      }
      throw error
    }
    this.indexDefinitionsDirty = true
    this.schedulePersist()
  }

  /** Remove a configured secondary index. */
  dropIndex(name: string): boolean {
    const removed = this.secondaryIndexes.delete(name)
    this.secondaryIndexData.delete(name)
    if (removed) { this.indexDefinitionsDirty = true; this.schedulePersist() }
    return removed
  }

  /** Rebuild all in-memory secondary-index buckets from the current node state. */
  rebuildIndexes(): void {
    const previous = this.captureSecondaryIndexes()
    try {
      for (const buckets of this.secondaryIndexData.values()) buckets.clear()
      for (const node of this.nodes.values()) this.indexSecondaryNode(node)
    } catch (error) {
      this.restoreSecondaryIndexes(previous)
      throw error
    }
  }

  /** Rebuild secondary-index buckets from all persisted records, including evicted nodes. */
  async rebuildIndexesFromPersistence(): Promise<void> {
    const ids = await this.persistence.allNodeIds()
    const nodes = await this.persistence.getNodes(ids)
    const previous = this.captureSecondaryIndexes()
    try {
      for (const buckets of this.secondaryIndexData.values()) buckets.clear()
      for (const node of nodes) this.indexSecondaryNode(node as unknown as PolyNode)
    } catch (error) {
      this.restoreSecondaryIndexes(previous)
      throw error
    }
  }

  private captureSecondaryIndexes(): Map<string, Map<string, Set<string>>> {
    return new Map([...this.secondaryIndexData].map(([name, buckets]) => [
      name,
      new Map([...buckets].map(([key, ids]) => [key, new Set(ids)])),
    ]))
  }

  private restoreSecondaryIndexes(snapshot: Map<string, Map<string, Set<string>>>): void {
    for (const [name, buckets] of this.secondaryIndexData) {
      buckets.clear()
      for (const [key, ids] of snapshot.get(name) ?? []) buckets.set(key, new Set(ids))
    }
  }

  get indexes(): IndexDefinition[] {
    return [...this.secondaryIndexes.values()].map(index => ({ ...index, fields: [...index.fields] }))
  }

  private indexValue(node: PolyNode, field: string): unknown {
    if (field === 'type') return node.type
    return this.dataValue(node.data, field)
  }

  private dataValue(data: Record<string, unknown> | undefined, field: string): unknown {
    const path = field.startsWith('data.') ? field.split('.').slice(1) : [field]
    let value: any = data
    for (const part of path) value = value?.[part]
    return value
  }

  private secondaryKey(node: PolyNode, index: IndexDefinition): string | undefined {
    if (index.nodeType && index.nodeType !== node.type) return undefined
    const values = index.fields.map(field => this.indexValue(node, field))
    if (index.sparse && values.some(value => value === undefined || value === null)) return undefined
    return JSON.stringify(values)
  }

  private indexSecondaryNode(node: PolyNode): void {
    for (const index of this.secondaryIndexes.values()) {
      const key = this.secondaryKey(node, index)
      if (key === undefined) continue
      const buckets = this.secondaryIndexData.get(index.name)!
      const ids = buckets.get(key) ?? new Set<string>()
      if (index.unique && ids.size > 0 && !ids.has(node.id)) {
        throw new UniqueConstraintError(index.name, key, [...ids][0])
      }
      ids.add(node.id)
      buckets.set(key, ids)
    }
  }

  private validateSecondaryNode(node: PolyNode, replacingId?: string): void {
    for (const index of this.secondaryIndexes.values()) {
      const key = this.secondaryKey(node, index)
      if (key === undefined || !index.unique) continue
      const ids = this.secondaryIndexData.get(index.name)?.get(key)
      if (ids && [...ids].some(id => id !== (replacingId ?? node.id))) {
        throw new UniqueConstraintError(index.name, key, [...ids][0])
      }
    }
  }

  private unindexSecondaryNode(node: PolyNode): void {
    for (const index of this.secondaryIndexes.values()) {
      const key = this.secondaryKey(node, index)
      if (key === undefined) continue
      const buckets = this.secondaryIndexData.get(index.name)
      const ids = buckets?.get(key)
      if (!ids) continue
      ids.delete(node.id)
      if (ids.size === 0) buckets!.delete(key)
    }
  }

  private indexCandidates(
    nodeTypes: string[] | undefined,
    attributes: Record<string, unknown> | undefined,
    ranges?: Record<string, { above?: number; below?: number }>,
  ): string[] | undefined {
    if (!attributes && !ranges) return undefined
    const type = nodeTypes?.length === 1 ? nodeTypes[0] : undefined
    for (const index of this.secondaryIndexes.values()) {
      if (index.nodeType && index.nodeType !== type) continue
      if (attributes && index.fields.every(field => Object.prototype.hasOwnProperty.call(attributes, field) ||
        Object.prototype.hasOwnProperty.call(attributes, field.replace(/^data\./, '')))) {
        const values = index.fields.map(field => Object.prototype.hasOwnProperty.call(attributes, field)
          ? attributes[field]
          : attributes[field.replace(/^data\./, '')])
        const ids = this.secondaryIndexData.get(index.name)?.get(JSON.stringify(values))
        return ids ? [...ids] : []
      }
      if (index.fields.length !== 1 || !ranges) continue
      const field = index.fields[0]
      const range = ranges[field] ?? ranges[field.replace(/^data\./, '')]
      if (!range) continue
      const ids: string[] = []
      for (const [encoded, bucket] of this.secondaryIndexData.get(index.name) ?? []) {
        const value = JSON.parse(encoded)[0]
        if (typeof value !== 'number' || !Number.isFinite(value)) continue
        if (range.above !== undefined && value <= range.above) continue
        if (range.below !== undefined && value >= range.below) continue
        ids.push(...bucket)
      }
      return ids
    }
    return undefined
  }

  private captureTransactionState(): any {
    return {
      nodes: new Map([...this.nodes].map(([id, node]) => [id, clonePolyNode(node)])),
      edges: new Map([...this.edges].map(([source, entries]) => [source, new Map([...entries].map(([key, edge]) => [key, { ...edge, data: edge.data ? cloneData(edge.data) : undefined }]))])),
      nodeToEdgeMap: new Map([...this.nodeToEdgeMap].map(([id, sources]) => [id, new Set(sources)])),
      dirtyNodes: new Set(this.dirtyNodes), dirtyEdges: new Set(this.dirtyEdges), dirtyVectors: new Set(this.dirtyVectors),
      removedNodeIds: new Set(this.removedNodeIds), removedEdgeIds: new Set(this.removedEdgeIds), removedVectorIds: new Set(this.removedVectorIds),
      hotCacheOrder: new Map(this.hotCacheOrder), evictedDirtyNodes: new Map(this.evictedDirtyNodes), sidecarData: new Map(this.sidecarData),
      vectors: [...this.vectors.entries()].map(([id, vector]) => [id, [...vector]] as const),
    }
  }

  private restoreTransactionState(snapshot: any): void {
    this.nodes = snapshot.nodes; this.edges = snapshot.edges; this.nodeToEdgeMap = snapshot.nodeToEdgeMap
    this.dirtyNodes = snapshot.dirtyNodes; this.dirtyEdges = snapshot.dirtyEdges; this.dirtyVectors = snapshot.dirtyVectors
    this.removedNodeIds = snapshot.removedNodeIds; this.removedEdgeIds = snapshot.removedEdgeIds; this.removedVectorIds = snapshot.removedVectorIds
    this.hotCacheOrder = snapshot.hotCacheOrder; this.evictedDirtyNodes = snapshot.evictedDirtyNodes; this.sidecarData = snapshot.sidecarData
    this.vectors.clear()
    for (const [id, vector] of snapshot.vectors) this.vectors.hydrate(id, vector)
    this._byType.clear()
    for (const node of this.nodes.values()) this.indexNode(node)
    for (const buckets of this.secondaryIndexData.values()) buckets.clear()
    for (const node of this.nodes.values()) this.indexSecondaryNode(node)
  }

  /** Queue change-event notifications until the matching {@link endBatch}. Nestable. */
  startBatch(): void {
    this.batchDepth++
  }

  /** Flush queued notifications from {@link startBatch}. Throws if no batch is open. */
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
    if (this.currentTransactionId && !event.transactionId) event = { ...event, transactionId: this.currentTransactionId }
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
    createVectorIndex?: (onChange: (id: string) => void) => VectorIndexLike,
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

  /** Validate a {@link NodeActivation} record: finite, in [0,1], non-negative count. */
  protected assertActivation(activation: NodeActivation): void {
    for (const key of ['score', 'importance'] as const) {
      if (!Number.isFinite(activation[key]) || activation[key] < 0 || activation[key] > 1) {
        throw new RangeError(`activation.${key} must be a finite number in [0, 1]`)
      }
    }
    if (!Number.isInteger(activation.reinforcementCount) || activation.reinforcementCount < 0) {
      throw new RangeError('activation.reinforcementCount must be a non-negative integer')
    }
    if (!Number.isFinite(activation.lastMeaningfulActivation) || activation.lastMeaningfulActivation < 0) {
      throw new RangeError('activation.lastMeaningfulActivation must be a finite non-negative number')
    }
  }

  protected schedulePersist(): void {
    if (this.persistTimer) clearTimeout(this.persistTimer)
    this.persistTimer = setTimeout(() => {
      this.persistTimer = null
      this.flush().catch((err) => console.warn('[PolyGraph] Flush error:', err))
    }, 2000)
  }

  /** Immediately persist queued mutations. Concurrent calls are serialized. */
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
      this.removedEdgeIds.size > 0 || this.removedVectorIds.size > 0 || this.indexDefinitionsDirty
  }

  private async flushPending(): Promise<void> {
    const dirtyNodeIds = [...this.dirtyNodes]
    const dirtyEdgeIds = [...this.dirtyEdges]
    const dirtyVectorIds = [...this.dirtyVectors]
    const removedNodeIds = [...this.removedNodeIds]
    const removedEdgeIds = [...this.removedEdgeIds]
    const removedVectorIds = [...this.removedVectorIds]
    const indexDefinitionsDirty = this.indexDefinitionsDirty
    const evictedSnapshots = new Map<string, SerializedNode>()
    for (const id of dirtyNodeIds) this.dirtyNodes.delete(id)
    for (const id of dirtyEdgeIds) this.dirtyEdges.delete(id)
    for (const id of dirtyVectorIds) this.dirtyVectors.delete(id)
    for (const id of removedNodeIds) this.removedNodeIds.delete(id)
    for (const id of removedEdgeIds) this.removedEdgeIds.delete(id)
    for (const id of removedVectorIds) this.removedVectorIds.delete(id)
    this.indexDefinitionsDirty = false

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
          revision: node.revision ?? 0,
          activation: node.activation ? { ...node.activation } : undefined,
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
      let source = parts[0]
      const innerKey = parts.slice(1).join('::')
      let sourceEdges = this.edges.get(source)
      let edge = parts.length >= 3
        ? sourceEdges?.get(innerKey) ?? [...(sourceEdges?.values() ?? [])].find(candidate => candidate.id === edgeIdStr)
        : undefined
      if (!edge) {
        for (const [candidateSource, entries] of this.edges) {
          const candidate = [...entries.values()].find(value => value.id === edgeIdStr)
          if (candidate) { source = candidateSource; sourceEdges = entries; edge = candidate; break }
        }
      }
      if (edge) {
        dirtyEdgeList.push({
          id: edge.id,
          source,
          target: edge.target,
          type: edge.type,
          data: edge.data ? cloneData(edge.data) : null,
          createdAt: edge.createdAt,
          revision: edge.revision,
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
        transactionId: this.currentTransactionId,
        operationId: this.currentTransactionId,
        actor: this.currentTransactionOptions?.actor,
        baseRevision: this.currentTransactionOptions?.baseRevision,
        metadata: this.currentTransactionOptions?.metadata,
        indexDefinitions: indexDefinitionsDirty ? this.indexes : undefined,
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
        if (indexDefinitionsDirty) throw new Error('Persistence adapter cannot persist index definitions')
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
      if (indexDefinitionsDirty) this.indexDefinitionsDirty = true
      throw error
    }
  }

  // ── Node CRUD ──

  /** Insert or replace a node. Replacement updates type/vector indexes; data and vector are structured-cloned on entry. */
  addNode(node: PolyNode, options?: WriteOptions): void {
    this.assertMutationAllowed()
    const stored = this.prepareNode(node)
    this.insertNode(stored, options?.expectedRevision)
    this.schedulePersist()
  }

  /**
   * Add several nodes in a single call. All nodes are validated before any are
   * inserted (an invalid entry inserts nothing), change events are coalesced
   * into one flush, and the persistence debounce is scheduled once for the
   * whole batch. Prefer this over a loop of `addNode` for large inserts.
   */
  addNodes(nodes: PolyNode[]): void {
    this.assertMutationAllowed()
    if (nodes.length === 0) return
    if (this.resourceLimits.maxBatchSize !== undefined && nodes.length > this.resourceLimits.maxBatchSize) {
      throw new ResourceLimitError('maxBatchSize', this.resourceLimits.maxBatchSize)
    }
    const prepared = new Array<PolyNode>(nodes.length)
    for (let i = 0; i < nodes.length; i++) prepared[i] = this.prepareNode(nodes[i])
    this.startBatch()
    try {
      for (const stored of prepared) this.insertNode(stored)
    } finally {
      this.endBatch()
    }
    this.schedulePersist()
  }

  protected prepareNode(node: PolyNode): PolyNode {
    if (!node.id) throw new TypeError('Node id must not be empty')
    if (!node.type) throw new TypeError('Node type must not be empty')
    if (!Number.isFinite(node.insertedAt) || node.insertedAt < 0 ||
        !Number.isFinite(node.updatedAt) || node.updatedAt < 0) {
      throw new RangeError('Node timestamps must be finite non-negative numbers')
    }
    if (node.vector) assertFiniteVector(node.vector)
    if (node.activation) this.assertActivation(node.activation)
    const serializedData = this.applySerialize(node.id, node.data as Record<string, unknown>)
    if (node.revision !== undefined && (!Number.isInteger(node.revision) || node.revision < 0)) {
      throw new RangeError('Node revision must be a non-negative integer')
    }
    const prepared = clonePolyNode({ ...node, data: serializedData, revision: node.revision ?? 0 })
    this.checkNodeResources(prepared, serializedData)
    this.validateNodeSchema(prepared)
    return prepared
  }

  protected insertNode(stored: PolyNode, expectedRevision?: number): void {
    const previous = this.nodes.get(stored.id)
    if (expectedRevision !== undefined && (previous?.revision ?? 0) !== expectedRevision) {
      throw new ConflictError(stored.id, expectedRevision, previous?.revision)
    }
    this.validateSecondaryNode(stored, previous?.id)
    if (previous) {
      this.unindexNode(stored.id)
      this.unindexSecondaryNode(previous)
    }
    if (!stored.vector) {
      this.vectors.remove(stored.id)
      this.dirtyVectors.delete(stored.id)
      this.removedVectorIds.add(stored.id)
    }
    this.removedNodeIds.delete(stored.id)
    if (previous) stored.revision = (previous.revision ?? 0) + 1
    this.nodes.set(stored.id, stored)
    this.touchHotCache(stored.id)
    this.dirtyNodes.add(stored.id)
    this.indexNode(stored)
    this.indexSecondaryNode(stored)
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

  /** Return a detached snapshot of a currently loaded node. Does not restore evicted nodes; see {@link getNodeSafe}. */
  getNode(id: string): PolyNode | undefined {
    const node = this.nodes.get(id)
    if (node) {
      this.touchHotCache(id)
      return this.applyDeserialize(clonePolyNode(node))
    }
    return undefined
  }

  /** Like {@link getNode}, but restores the node from persistence first if it was evicted. */
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
      revision: serialized.revision ?? 0,
      activation: serialized.activation ? { ...serialized.activation } : undefined,
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

  /** Shallow-merge `data` into a loaded node and optionally replace its vector or activation. No-op (returns `undefined`) if the node isn't loaded; see {@link updateNodeSafe}. */
  updateNode(
    id: string,
    data: Partial<Record<string, unknown>>,
    vector?: Float64Array | WriteOptions,
    activation?: NodeActivation,
    options?: WriteOptions,
  ): PolyNode | undefined {
    this.assertMutationAllowed()
    const node = this.nodes.get(id)
    if (!node) return undefined
    const writeOptions = (vector && !(vector instanceof Float64Array) ? vector : options)
    if (writeOptions?.expectedRevision !== undefined && (node.revision ?? 0) !== writeOptions.expectedRevision) {
      throw new ConflictError(id, writeOptions.expectedRevision, node.revision ?? 0)
    }
    if (vector && vector instanceof Float64Array) {
      assertFiniteVector(vector)
    }
    const serialized = this.applySerialize(id, data as Record<string, unknown>)
    const previous = clonePolyNode(node)
    const candidate: PolyNode = {
      ...clonePolyNode(node),
      data: { ...cloneData(node.data), ...cloneData(serialized) },
      vector: vector instanceof Float64Array ? new Float64Array(vector) : node.vector,
      activation: activation === undefined ? node.activation : { ...activation },
      updatedAt: Date.now(),
      revision: (node.revision ?? 0) + 1,
    }
    this.checkNodeResources(candidate, candidate.data)
    this.validateNodeSchema(candidate)
    Object.assign(node.data, cloneData(serialized))
    if (vector instanceof Float64Array) {
      node.vector = new Float64Array(vector)
      this.removedVectorIds.delete(id)
      this.vectors.add(id, [...vector])
      this.dirtyVectors.add(id)
    }
    if (activation !== undefined) {
      this.assertActivation(activation)
      node.activation = { ...activation }
    }
    node.updatedAt = Date.now()
    node.revision = (node.revision ?? 0) + 1
    this.unindexSecondaryNode(previous)
    try {
      this.validateSecondaryNode(node, id)
      this.indexSecondaryNode(node)
    } catch (error) {
      this.unindexSecondaryNode(node)
      Object.assign(node, previous)
      this.indexSecondaryNode(previous)
      throw error
    }
    this.touchHotCache(id)
    this.markDirty(id)
    this.emitChange({ type: 'node_updated', nodeId: id, nodeType: node.type })
    return this.applyDeserialize(clonePolyNode(node))
  }

  /** Apply a small, deterministic partial-update language to a node. */
  patchNode(id: string, patch: NodePatch, options?: WriteOptions): PolyNode | undefined {
    this.assertMutationAllowed()
    const node = this.nodes.get(id)
    if (!node) return undefined
    if (options?.expectedRevision !== undefined && (node.revision ?? 0) !== options.expectedRevision) {
      throw new ConflictError(id, options.expectedRevision, node.revision ?? 0)
    }
    const data = cloneData(node.data)
    const get = (path: string): unknown => path === 'data' ? data : path.split('.').slice(1).reduce((v: any, key) => v?.[key], data)
    const set = (path: string, value: unknown): void => {
      if (path === 'updatedAt') { if (typeof value !== 'number' || !Number.isFinite(value)) throw new RangeError('updatedAt must be finite'); node.updatedAt = value; return }
      if (!path.startsWith('data.')) throw new TypeError(`Unsupported patch path: ${path}`)
      const parts = path.split('.').slice(1); let target: any = data
      for (const part of parts.slice(0, -1)) { if (!target[part] || typeof target[part] !== 'object') target[part] = {}; target = target[part] }
      target[parts[parts.length - 1]] = cloneData(value)
    }
    for (const [path, value] of Object.entries(patch.compareAndSet ?? {})) {
      if (JSON.stringify(get(path)) !== JSON.stringify(value.expected)) throw new ConflictError(id, options?.expectedRevision ?? (node.revision ?? 0), node.revision ?? 0)
      set(path, value.value)
    }
    for (const [path, value] of Object.entries(patch.set ?? {})) set(path, value)
    for (const path of patch.unset ?? []) {
      if (!path.startsWith('data.')) throw new TypeError(`Unsupported patch path: ${path}`)
      const parts = path.split('.').slice(1); let target: any = data
      for (const part of parts.slice(0, -1)) target = target?.[part]
      if (target && typeof target === 'object') delete target[parts[parts.length - 1]]
    }
    for (const [path, amount] of Object.entries(patch.increment ?? {})) {
      if (!Number.isFinite(amount)) throw new RangeError(`Increment for ${path} must be finite`)
      const current = get(path); if (current !== undefined && typeof current !== 'number') throw new TypeError(`Increment target ${path} is not numeric`)
      set(path, (current as number | undefined ?? 0) + amount)
    }
    return this.updateNode(id, data, undefined, undefined, { expectedRevision: node.revision ?? 0 })
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
  removeNode(id: string, cascadeVisitedOrOptions?: Set<string> | WriteOptions): void {
    this.assertMutationAllowed()
    const options = cascadeVisitedOrOptions instanceof Set ? undefined : cascadeVisitedOrOptions
    const visited = cascadeVisitedOrOptions instanceof Set ? cascadeVisitedOrOptions : new Set<string>()
    if (visited.has(id)) return
    visited.add(id)

    const node = this.nodes.get(id)
    if (!node) return
    if (options?.expectedRevision !== undefined && (node.revision ?? 0) !== options.expectedRevision) {
      throw new ConflictError(id, options.expectedRevision, node.revision ?? 0)
    }

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
    this.unindexSecondaryNode(node)
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
    const id = edge.id
    this.dirtyEdges.delete(id)
    this.removedEdgeIds.add(id)
    this.emitChange({ type: 'edge_removed', edgeId: edge.id, edgeType: edge.type, source, target: edge.target })
  }

  // ── Edge CRUD ──

  /** Add one directed edge. A no-op if an edge with the same source/type/target already exists (edges are unique per triple). */
  addEdge(edge: PolyEdge, options?: WriteOptions): void
  addEdge(source: string, type: string, target: string, data?: Record<string, unknown>, ownership?: EdgeOwnership): void
  addEdge(sourceOrEdge: string | PolyEdge, type?: string | WriteOptions, target?: string, data?: Record<string, unknown>, ownership?: EdgeOwnership, options?: WriteOptions): void {
    this.assertMutationAllowed()
    const objectInput = typeof sourceOrEdge !== 'string'
    const source = objectInput ? sourceOrEdge.source : sourceOrEdge
    const edgeType = objectInput ? sourceOrEdge.type : type as string
    const edgeTarget = objectInput ? sourceOrEdge.target : target!
    const edgeData = objectInput ? sourceOrEdge.data : data
    const edgeRevision = objectInput ? sourceOrEdge.revision ?? 0 : 0
    const expectedRevision = objectInput
      ? (typeof type === 'object' ? type.expectedRevision : options?.expectedRevision)
      : undefined
    if (!Number.isInteger(edgeRevision) || edgeRevision < 0) throw new RangeError('Edge revision must be a non-negative integer')
    if (!source || !edgeType || !edgeTarget) throw new TypeError('Edge source, type, and target must not be empty')
    const id = objectInput ? sourceOrEdge.id : edgeId(source, edgeType, edgeTarget)
    if (!id) throw new TypeError('Edge id must not be empty')
    const sourceEdges = this.edges.get(source)
    const existing = sourceEdges?.get(id)
    if (expectedRevision !== undefined && (existing?.revision ?? 0) !== expectedRevision) {
      throw new ConflictError(id, expectedRevision, existing?.revision)
    }
    if (existing) return

    const inputData = edgeData === undefined ? undefined : cloneData(edgeData)
    const fullData = ownership !== undefined ? { ...inputData, [OWNERSHIP_KEY]: ownership } : inputData

    const candidate: PolyEdge = {
      id,
      source,
      target: edgeTarget,
      type: edgeType,
      data: fullData,
      createdAt: objectInput ? sourceOrEdge.createdAt : Date.now(),
      revision: edgeRevision,
    }
    this.validateEdgeSchema(candidate)

    this.removedEdgeIds.delete(id)
    if (!this.edges.has(source)) this.edges.set(source, new Map())
    const targetEdges = this.edges.get(source)!
    targetEdges.set(id, { id, target: edgeTarget, type: edgeType, data: fullData, revision: edgeRevision, createdAt: candidate.createdAt })
    if (!this.nodeToEdgeMap.has(edgeTarget)) this.nodeToEdgeMap.set(edgeTarget, new Set())
    this.nodeToEdgeMap.get(edgeTarget)!.add(source)
    this.dirtyEdges.add(id)
    this.schedulePersist()
    this.emitChange({ type: 'edge_added', edgeId: id, edgeType, source, target: edgeTarget })
  }

  /** Remove one edge by its canonical ID, optionally checking its revision. */
  updateEdge(id: string, data: Partial<Record<string, unknown>>, options?: WriteOptions): PolyEdge | undefined {
    this.assertMutationAllowed()
    let source: string | undefined
    let edge: EdgeEntry | undefined
    for (const [candidateSource, entries] of this.edges) {
      const candidate = entries.get(id)
      if (candidate) {
        source = candidateSource
        edge = candidate
        break
      }
    }
    if (!edge || source === undefined) return undefined
    if (options?.expectedRevision !== undefined && edge.revision !== options.expectedRevision) {
      throw new ConflictError(id, options.expectedRevision, edge.revision)
    }
    const candidate: PolyEdge = {
      id: edge.id,
      source,
      target: edge.target,
      type: edge.type,
      data: { ...(edge.data ?? {}), ...cloneData(data) },
      createdAt: edge.createdAt,
      revision: edge.revision + 1,
    }
    this.validateEdgeSchema(candidate)
    edge.data = candidate.data
    edge.revision = candidate.revision!
    this.dirtyEdges.add(id)
    this.schedulePersist()
    this.emitChange({ type: 'edge_updated', edgeId: id, edgeType: edge.type, source, target: edge.target })
    return { ...candidate, data: candidate.data ? cloneData(candidate.data) : undefined }
  }

  /** Remove one edge by its canonical ID, optionally checking its revision. */
  removeEdge(id: string, options?: WriteOptions): boolean {
    this.assertMutationAllowed()
    let source = id.split('::')[0]
    let edge = [...(this.edges.get(source)?.values() ?? [])].find(candidate => candidate.id === id)
    if (!edge) {
      for (const [candidateSource, entries] of this.edges) {
        const candidate = [...entries.values()].find(value => value.id === id)
        if (candidate) { source = candidateSource; edge = candidate; break }
      }
    }
    if (!edge) return false
    if (options?.expectedRevision !== undefined && edge.revision !== options.expectedRevision) {
      throw new ConflictError(id, options.expectedRevision, edge.revision)
    }
    this.removeEdges(source, edge.type, edge.target)
    return true
  }

  /**
   * Mark `id`'s vector for persistence after it was mutated directly through
   * `vectors` (e.g. `graph.vectors.add(...)`) rather than via `addNode`/
   * `updateNode`, which schedule persistence themselves.
   */
  markVectorDirty(id: string): void {
    if (this.vectors.has(id)) this.dirtyVectors.add(id)
    this.schedulePersist()
  }

  /** Detached snapshots of `source`'s outgoing edges, optionally filtered by type. */
  getEdges(source: string, type?: string): EdgeEntry[] {
    const edges = this.edges.get(source)
    if (!edges) return []
    const all = [...edges.values()]
    const selected = type ? all.filter(e => e.type === type) : all
    return selected.map(edge => ({ ...edge, data: edge.data ? cloneData(edge.data) : undefined }))
  }

  /** IDs of nodes reachable from `source` via outgoing edges of `type`. */
  getEdgeTargets(source: string, type: string): string[] {
    const edges = this.edges.get(source)
    if (!edges) return []
    return [...edges.values()].filter(e => e.type === type).map(e => e.target)
  }

  /** IDs of nodes with an outgoing edge of `type` into `target`. */
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
    this.assertMutationAllowed()
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
      if (!removeAll) edges.delete(edge.id)
      this.nodeToEdgeMap.get(edge.target)?.delete(source)
      const id = edge.id
      this.dirtyEdges.delete(id)
      this.removedEdgeIds.add(id)
      this.emitChange({ type: 'edge_removed', edgeId: edge.id, edgeType: edge.type, source, target: edge.target })

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

  /** Create a mutable {@link GraphQuery} over the currently loaded (hot) nodes. */
  query(): GraphQuery {
    return new GraphQuery(this.nodes, this.edges, this.nodeToEdgeMap, this.transform, this.sidecarData, this.indexes, this.indexCandidates.bind(this), metrics => this.recordQueryMetrics(metrics))
  }

  /** Return operational counters without binding the graph to a metrics package. */
  async stats(): Promise<GraphStats> {
    const adapterStats = this.persistence.stats ? await this.persistence.stats() : {}
    const persistedNodeCount = adapterStats.persistedNodeCount ?? (await this.persistence.allNodeIds()).length
    const edgeCount = adapterStats.edgeCount ?? (await this.persistence.getAllEdges()).length
    const vectorCount = adapterStats.vectorCount ?? (await this.persistence.getAllVectors()).length
    let memoryEstimateBytes = 0
    try { memoryEstimateBytes = JSON.stringify([...this.nodes.values()]).length * 2 } catch { /* non-JSON sidecars are excluded */ }
    return {
      loadedNodeCount: this.nodes.size,
      persistedNodeCount,
      edgeCount,
      vectorCount,
      dirtyRecordCount: this.dirtyNodes.size + this.dirtyEdges.size + this.dirtyVectors.size + this.removedNodeIds.size + this.removedEdgeIds.size + this.removedVectorIds.size,
      pendingPersistence: this.hasPendingPersistence(),
      indexCount: this.secondaryIndexes.size,
      memoryEstimateBytes,
      queryCount: this.queryCount,
      queryDurationMs: this.queryDurationMs,
      queryScannedRecords: this.queryScannedRecords,
      queryIndexUsage: Object.fromEntries(this.queryIndexUsage),
      ...adapterStats,
    }
  }

  /** Force a durable checkpoint when the active adapter supports snapshots. */
  async checkpoint(): Promise<void> {
    if (!this.persistence.checkpoint) throw new AdapterCapabilityError('snapshots')
    await this.flush()
    await this.persistence.checkpoint()
  }

  /** Copy a consistent store backup when the active adapter supports it. */
  async backup(destination: FileIO): Promise<void> {
    if (!this.persistence.backup) throw new AdapterCapabilityError('snapshots')
    await this.flush()
    await this.persistence.backup(destination)
  }

  /** Verify durable storage and logical graph invariants when supported. */
  async verify(): Promise<VerificationReport> {
    await this.flush()
    if (this.persistence.verify) return this.persistence.verify()
    const errors: string[] = []
    for (const node of this.nodes.values()) {
      if (!node.id || !node.type) errors.push(`invalid node identity: ${node.id}`)
      if (!Number.isFinite(node.insertedAt) || !Number.isFinite(node.updatedAt)) errors.push(`invalid node timestamps: ${node.id}`)
      if (node.vector && ![...node.vector].every(Number.isFinite)) errors.push(`invalid node vector: ${node.id}`)
      if (node.vector && this.vectors.get(node.id) && this.vectors.get(node.id)!.length !== node.vector.length) errors.push(`vector dimensionality mismatch: ${node.id}`)
    }
    for (const [source, entries] of this.edges) {
      for (const edge of entries.values()) {
        if (!edge.id || !source || !edge.target || !edge.type) errors.push(`invalid edge identity: ${edge.id}`)
        if (!this.nodes.has(source)) errors.push(`edge ${edge.id} has missing source ${source}`)
        if (!this.nodes.has(edge.target)) errors.push(`edge ${edge.id} has missing target ${edge.target}`)
      }
    }
    for (const [type, ids] of this._byType) {
      for (const id of ids) if (this.nodes.get(id)?.type !== type) errors.push(`stale type index entry: ${type}/${id}`)
    }
    return { ok: errors.length === 0, errors, nodeCount: this.nodes.size, edgeCount: [...this.edges.values()].reduce((count, entries) => count + entries.size, 0), vectorCount: this.vectors.size, mutationCount: 0 }
  }

  /** Capture a detached, queryable view of the currently warmed graph state. */
  async snapshot(): Promise<GraphSnapshot> {
    await this.warm()
    const nodes = new Map([...this.nodes].map(([id, node]) => [id, clonePolyNode(node)] as const))
    const edges = new Map([...this.edges].map(([source, entries]) => [source, new Map([...entries].map(([id, edge]) => [id, {
      ...edge,
      data: edge.data ? cloneData(edge.data) : undefined,
    }]))] as const))
    const nodeToEdgeMap = new Map([...this.nodeToEdgeMap].map(([id, sources]) => [id, new Set(sources)] as const))
    return new GraphSnapshot(nodes, edges, nodeToEdgeMap, this.transform, new Map(this.sidecarData), this.indexes)
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
  queryPersisted(limits?: QueryResourceLimits): PersistedGraphQuery {
    return new PersistedGraphQuery(this.persistence, this.transform, this.sidecarData, limits, metrics => this.recordQueryMetrics(metrics))
  }

  private recordQueryMetrics(metrics: QueryMetrics): void {
    this.queryCount++
    this.queryDurationMs += metrics.durationMs
    this.queryScannedRecords += metrics.scannedRecords
    if (metrics.index) this.queryIndexUsage.set(metrics.index, (this.queryIndexUsage.get(metrics.index) ?? 0) + 1)
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
          revision: node.revision ?? 0,
          activation: node.activation ? { ...node.activation } : undefined,
        })
      }
      this.hotCacheOrder.delete(evict)
      this.unindexNode(evict)
      this.nodes.delete(evict)
      this.vectors.remove(evict)
    }
  }

  // ── Persistence ──

  /** Write the complete currently loaded graph to the adapter without clearing dirty state. */
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
        revision: node.revision ?? 0,
        activation: node.activation ? { ...node.activation } : undefined,
      })
    }

    for (const [source, edgeList] of this.edges) {
      for (const e of edgeList.values()) {
        edges.push({
          id: e.id,
          source,
          target: e.target,
          type: e.type,
          data: e.data ? cloneData(e.data) : null,
          createdAt: e.createdAt,
          revision: e.revision,
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

  /** Alias for {@link warm}. */
  async load(): Promise<void> {
    await this.warm()
  }

  protected async rebuildEdgeIndex(): Promise<void> {
    this.edges.clear()
    this.nodeToEdgeMap.clear()
    const allEdges = await this.persistence.getAllEdges()
    for (const e of allEdges) {
      if (!e.id) throw new Error('Invalid persisted edge ID: empty')
      if (!this.edges.has(e.source)) this.edges.set(e.source, new Map())
      this.edges.get(e.source)!.set(e.id, {
        id: e.id,
        target: e.target,
        type: e.type,
        data: e.data ? cloneData(e.data) : undefined,
        revision: e.revision ?? 0,
        createdAt: e.createdAt,
      })
      if (!this.nodeToEdgeMap.has(e.target)) this.nodeToEdgeMap.set(e.target, new Set())
      this.nodeToEdgeMap.get(e.target)!.add(e.source)
    }
  }

  /** Load persisted nodes, vectors, and edges. Idempotent — a no-op after the first call until {@link clear} or {@link dispose}. */
  async warm(): Promise<void> {
    if (this._warmed) return
    this._warmed = true
    if (this.persistence.getIndexDefinitions) {
      const persistedIndexes = await this.persistence.getIndexDefinitions()
      for (const index of persistedIndexes) this.defineIndex(index)
      this.indexDefinitionsDirty = false
    }
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
          revision: sn.revision ?? 0,
          activation: sn.activation ? { ...sn.activation } : undefined,
        })
        this.indexNode(this.nodes.get(sn.id)!)
        this.indexSecondaryNode(this.nodes.get(sn.id)!)
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

  /** Clear in-memory state only — does not flush pending mutations or touch the adapter's persisted contents. */
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
    for (const buckets of this.secondaryIndexData.values()) buckets.clear()
    this.evictedDirtyNodes.clear()
    this.dirtyEdges.clear()
    this.dirtyVectors.clear()
    this.dirtyNodes.clear()
    this.removedNodeIds.clear()
    this.removedEdgeIds.clear()
    this.removedVectorIds.clear()
    this._warmed = false
  }

  /** Flush queued mutations, clear in-memory state, then close the adapter. */
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

  /** Detached snapshots of currently loaded nodes of `type`. */
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
    if (this.persistence.countNodes) return this.persistence.countNodes({})
    return (await this.persistence.allNodeIds()).length
  }

  // ── Activation ──

  /**
   * Current decayed activation score of a loaded node, or 0 when it has none.
   * Decay is a pure function of elapsed time from `lastMeaningfulActivation`,
   * so this is deterministic across replicas with the same stored state.
   * `halfLifeMs` defaults to the standard 24h score curve.
   */
  getActivation(id: string, halfLifeMs: number = ACTIVATION_DEFAULTS.scoreHalfLifeMs): number {
    const node = this.nodes.get(id)
    return node ? activationScoreOf(node, Date.now(), halfLifeMs) : 0
  }

  /** Decay-corrected view of a loaded node's durable activation, or `undefined`. */
  getActivationState(id: string): NodeActivation | undefined {
    const node = this.nodes.get(id)
    if (!node?.activation) return undefined
    return decayActivationState(node.activation, Date.now())
  }

  /**
   * Apply a durable reinforcement delta to a loaded node's activation. The
   * prior state is decay-corrected to now, `amount` is added to `score`, a
   * fraction is folded into `importance`, the reinforcement counter increments,
   * and the decay anchor re-sets to now. Persists and emits an
   * `activation_updated` change event. Returns the updated node or `undefined`
   * if the node isn't loaded (see {@link reinforceNodeSafe}).
   */
  reinforceNode(id: string, amount: number, reason?: string): PolyNode | undefined {
    if (!Number.isFinite(amount)) throw new RangeError('Reinforcement amount must be finite')
    const node = this.nodes.get(id)
    if (!node) return undefined
    const now = Date.now()
    node.activation = reinforceActivation(node.activation, amount, now, ACTIVATION_DEFAULTS)
    node.updatedAt = now
    this.touchHotCache(id)
    this.markDirty(id)
    this.emitChange({ type: 'activation_updated', nodeId: id, nodeType: node.type, delta: amount, reason })
    return this.applyDeserialize(clonePolyNode(node))
  }

  /** Restore an evicted node when necessary, then reinforce it. */
  async reinforceNodeSafe(id: string, amount: number, reason?: string): Promise<PolyNode | undefined> {
    if (!Number.isFinite(amount)) throw new RangeError('Reinforcement amount must be finite')
    if (!await this.getNodeSafe(id)) return undefined
    return this.reinforceNode(id, amount, reason)
  }

  /** Loaded nodes with the highest current activation, descending. The working-memory primitive. */
  topActivated(limit: number, minScore = 0): PolyNode[] {
    if (!Number.isInteger(limit) || limit < 0) throw new RangeError('limit must be a non-negative integer')
    if (limit === 0) return []
    const now = Date.now()
    const scored: Array<{ node: PolyNode; score: number }> = []
    for (const node of this.nodes.values()) {
      const score = activationScoreOf(node, now)
      if (score > minScore) scored.push({ node, score })
    }
    scored.sort((a, b) => b.score - a.score)
    return scored.slice(0, limit).map(({ node }) => this.applyDeserialize(clonePolyNode(node)))
  }

  /**
   * Materialize decay for every loaded node with activation: rewrite each
   * node's stored values to their current decayed state and re-anchor to
   * `now`. Reads already decay lazily, so this only matters for persisting
   * fresh values (e.g. before eviction-driven lifecycle events).
   */
  decay(now: number = Date.now()): void {
    for (const node of this.nodes.values()) {
      if (!node.activation) continue
      const corrected = decayActivationState(node.activation, now)
      node.activation = {
        score: corrected.score,
        importance: corrected.importance,
        reinforcementCount: node.activation.reinforcementCount,
        lastMeaningfulActivation: now,
      }
      node.updatedAt = now
      this.markDirty(node.id)
    }
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
