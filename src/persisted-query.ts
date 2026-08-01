import type { PolyNode, SerializedEdge, SerializedNode, DataTransform } from './types.js'
import type { PersistenceAdapter, PersistedNodeQuery } from './persistence/adapter.js'
import { applyPersistedNodeQuery } from './persistence/query.js'
import { cosineSimilarity } from './vector-index.js'
import { activationScoreOf, assertFiniteVector, assertNonNegativeInteger, cloneData } from './utils.js'

function readNode(node: PolyNode, transform?: DataTransform, sidecarData?: Map<string, unknown>): PolyNode {
  if (!transform?.deserialize) return node
  const sidecar = sidecarData?.get(node.id)
  return { ...node, data: transform.deserialize(node.data, sidecar) as Record<string, unknown> }
}

function restoreNode(node: SerializedNode, transform?: DataTransform, sidecarData?: Map<string, unknown>): PolyNode {
  return readNode({
    id: node.id,
    type: node.type,
    data: cloneData(node.data),
    vector: node.vector ? new Float64Array(node.vector) : undefined,
    insertedAt: node.insertedAt,
    updatedAt: node.updatedAt,
    activation: node.activation ? { ...node.activation } : undefined,
  }, transform, sidecarData)
}

/** Chainable asynchronous query over all persisted nodes. Results are detached. */
export class PersistedGraphQuery {
  private readonly adapter: PersistenceAdapter
  private readonly transform?: DataTransform
  private readonly sidecarData: Map<string, unknown>
  private query: PersistedNodeQuery = {}
  private resultOffset?: number
  private resultLimit?: number
  private similarVector?: { vector: number[]; threshold: number; topK?: number }
  private edgeType?: string
  private edgeTarget?: string
  private edgeSource?: string
  private activationAbove?: number
  private activationOrder?: 'asc' | 'desc'
  private joins: Array<{
    edgeType: string
    direction: 'out' | 'in'
    predicate?: (node: PolyNode) => boolean
  }> = []
  private traversals: Array<{
    edgeType: string
    depth: number
    direction: 'out' | 'in'
  }> = []

  constructor(adapter: PersistenceAdapter, transform?: DataTransform, sidecarData?: Map<string, unknown>) {
    this.adapter = adapter
    this.transform = transform
    this.sidecarData = sidecarData ?? new Map()
  }

  where(field: string, value: unknown): this {
    this.query.attributes = { ...this.query.attributes, [field]: value }
    return this
  }

  whereAttribute(name: string, value: unknown): this {
    return this.where(name, value)
  }

  whereAttributeRange(name: string, range: { above?: number; below?: number }): this {
    if (range.above !== undefined && !Number.isFinite(range.above)) throw new RangeError('Range above must be finite')
    if (range.below !== undefined && !Number.isFinite(range.below)) throw new RangeError('Range below must be finite')
    this.query.attributeRanges = { ...this.query.attributeRanges, [name]: range }
    return this
  }

  whereNodeType(...types: string[]): this {
    this.query.nodeTypes = types
    return this
  }

  whereEdge(type: string, target?: string): this {
    if (!type) throw new TypeError('Edge type must not be empty')
    this.edgeType = type
    this.edgeTarget = target
    return this
  }

  whereEdgeSource(source: string): this {
    if (!source) throw new TypeError('Edge source must not be empty')
    this.edgeSource = source
    return this
  }

  join(
    edgeType: string,
    direction: 'out' | 'in' = 'out',
    predicate?: (connectedNode: PolyNode) => boolean,
  ): this {
    if (!edgeType) throw new TypeError('Edge type must not be empty')
    this.joins.push({ edgeType, direction, predicate })
    return this
  }

  traverse(edgeType: string, depth: number, direction: 'out' | 'in' = 'out'): this {
    if (!edgeType) throw new TypeError('Edge type must not be empty')
    assertNonNegativeInteger(depth, 'depth')
    this.traversals.push({ edgeType, depth, direction })
    return this
  }

  orderBy(field: string, direction: 'asc' | 'desc' = 'asc'): this {
    this.query.orderBy = { field, direction }
    return this
  }

  /** Keep only nodes whose current (decay-corrected) activation exceeds `above`. */
  whereActivated(above: number): this {
    if (!Number.isFinite(above)) throw new RangeError('above must be finite')
    this.activationAbove = above
    return this
  }

  /** Order results by current (decay-corrected) activation instead of a data field. */
  orderByActivation(direction: 'asc' | 'desc' = 'desc'): this {
    this.activationOrder = direction
    return this
  }

  offset(n: number): this {
    assertNonNegativeInteger(n, 'offset')
    this.resultOffset = n
    return this
  }

  limit(n: number): this {
    assertNonNegativeInteger(n, 'limit')
    this.resultLimit = n
    return this
  }

  similarTo(vector: number[], threshold = 0, topK?: number): this {
    assertFiniteVector(vector, 'query vector')
    if (!Number.isFinite(threshold)) throw new RangeError('threshold must be finite')
    if (topK !== undefined) assertNonNegativeInteger(topK, 'topK')
    this.similarVector = { vector, threshold, topK }
    return this
  }

  private async serialized(includeOrder = true, includePagination = false): Promise<SerializedNode[]> {
    const query: PersistedNodeQuery = {
      ...this.query,
      orderBy: includeOrder ? this.query.orderBy : undefined,
      offset: includePagination ? this.resultOffset : undefined,
      limit: includePagination ? this.resultLimit : undefined,
    }
    if (this.adapter.queryNodes) return this.adapter.queryNodes(query)
    const ids = await this.adapter.allNodeIds()
    return applyPersistedNodeQuery(await this.adapter.getNodes(ids), query)
  }

  private async edgesBySources(sources: string[], type?: string): Promise<SerializedEdge[]> {
    if (this.adapter.getEdgesBySources) return this.adapter.getEdgesBySources(sources, type)
    const sourceSet = new Set(sources)
    return (await this.adapter.getAllEdges()).filter(edge =>
      sourceSet.has(edge.source) && (type === undefined || edge.type === type)
    )
  }

  private async edgesByTargets(targets: string[], type?: string): Promise<SerializedEdge[]> {
    if (this.adapter.getEdgesByTargets) return this.adapter.getEdgesByTargets(targets, type)
    const targetSet = new Set(targets)
    return (await this.adapter.getAllEdges()).filter(edge =>
      targetSet.has(edge.target) && (type === undefined || edge.type === type)
    )
  }

  private async applyEdgeFilters(nodes: SerializedNode[]): Promise<SerializedNode[]> {
    let results = nodes
    if (this.edgeType) {
      const edges = await this.edgesBySources(results.map(node => node.id), this.edgeType)
      const sources = new Set(edges
        .filter(edge => this.edgeTarget === undefined || edge.target === this.edgeTarget)
        .map(edge => edge.source))
      results = results.filter(node => sources.has(node.id))
    }
    if (this.edgeSource) {
      const edges = await this.edgesBySources([this.edgeSource], this.edgeType)
      const targets = new Set(edges.map(edge => edge.target))
      results = results.filter(node => targets.has(node.id))
    }
    return results
  }

  private async applyJoins(nodes: SerializedNode[]): Promise<SerializedNode[]> {
    let results = nodes
    for (const join of this.joins) {
      const ids = results.map(node => node.id)
      const edges = join.direction === 'out'
        ? await this.edgesBySources(ids, join.edgeType)
        : await this.edgesByTargets(ids, join.edgeType)
      const connectedIds = [...new Set(edges.map(edge =>
        join.direction === 'out' ? edge.target : edge.source
      ))]
      const connected = new Map(
        (await this.adapter.getNodes(connectedIds)).map(node => [node.id, restoreNode(node, this.transform, this.sidecarData)]),
      )
      const matched = new Set<string>()
      for (const edge of edges) {
        const candidateId = join.direction === 'out' ? edge.source : edge.target
        const connectedId = join.direction === 'out' ? edge.target : edge.source
        const connectedNode = connected.get(connectedId)
        if (connectedNode && (!join.predicate || join.predicate(connectedNode))) {
          matched.add(candidateId)
        }
      }
      results = results.filter(node => matched.has(node.id))
    }
    return results
  }

  private async applyTraversals(nodes: SerializedNode[]): Promise<SerializedNode[]> {
    let currentIds = nodes.map(node => node.id)
    for (const traversal of this.traversals) {
      const visited = new Set(currentIds)
      let frontier = [...currentIds]
      for (let depth = 0; depth < traversal.depth && frontier.length > 0; depth++) {
        const edges = traversal.direction === 'out'
          ? await this.edgesBySources(frontier, traversal.edgeType)
          : await this.edgesByTargets(frontier, traversal.edgeType)
        const next: string[] = []
        for (const edge of edges) {
          const id = traversal.direction === 'out' ? edge.target : edge.source
          if (!visited.has(id)) {
            visited.add(id)
            next.push(id)
          }
        }
        frontier = next
      }
      currentIds = [...visited]
    }
    return this.adapter.getNodes(currentIds)
  }

  async toArray(): Promise<PolyNode[]> {
    const hasTraversal = this.traversals.length > 0
    const adapterCanPaginate = !this.similarVector && !this.edgeType && !this.edgeSource &&
      this.joins.length === 0 && !hasTraversal &&
      this.activationAbove === undefined && this.activationOrder === undefined
    let serialized = await this.serialized(!hasTraversal, adapterCanPaginate)
    serialized = await this.applyEdgeFilters(serialized)
    serialized = await this.applyJoins(serialized)
    if (hasTraversal) serialized = await this.applyTraversals(serialized)
    let results = serialized.map(n => restoreNode(n, this.transform, this.sidecarData))

    if (this.activationAbove !== undefined) {
      results = results.filter(n => activationScoreOf(n) >= this.activationAbove!)
    }
    if (this.activationOrder) {
      const now = Date.now()
      results = [...results].sort((a, b) => {
        const av = activationScoreOf(a, now)
        const bv = activationScoreOf(b, now)
        return this.activationOrder === 'desc' ? bv - av : av - bv
      })
    }

    if (hasTraversal && this.query.orderBy) {
      const { field, direction } = this.query.orderBy
      results.sort((a, b) => {
        const av = a.data[field]
        const bv = b.data[field]
        const an = typeof av === 'number' && Number.isFinite(av) ? av : 0
        const bn = typeof bv === 'number' && Number.isFinite(bv) ? bv : 0
        return direction === 'asc' ? an - bn : bn - an
      })
    }

    if (this.similarVector) {
      const { vector, threshold, topK } = this.similarVector
      const scored = results
        .filter((node): node is PolyNode & { vector: Float64Array } => node.vector !== undefined)
        .map(node => ({ node, score: cosineSimilarity(vector, node.vector) }))
        .filter(result => result.score >= threshold)
        .sort((a, b) => b.score - a.score)
      results = (topK === undefined ? scored : scored.slice(0, topK)).map(result => result.node)
    }

    if (!adapterCanPaginate) {
      if (this.resultOffset !== undefined) results = results.slice(this.resultOffset)
      if (this.resultLimit !== undefined) results = results.slice(0, this.resultLimit)
    }
    return results
  }

  async first(): Promise<PolyNode | null> {
    return (await this.toArray())[0] ?? null
  }

  async ids(): Promise<string[]> {
    return (await this.toArray()).map(node => node.id)
  }

  async count(): Promise<number> {
    if (this.similarVector || this.resultOffset !== undefined || this.resultLimit !== undefined ||
        this.edgeType || this.edgeSource || this.joins.length > 0 || this.traversals.length > 0 ||
        this.activationAbove !== undefined || this.activationOrder !== undefined) {
      return (await this.toArray()).length
    }
    if (this.adapter.countNodes) return this.adapter.countNodes(this.query)
    return (await this.serialized()).length
  }

  async collect(
    edgeType: string,
    direction: 'out' | 'in' = 'out',
    predicate?: (node: PolyNode) => boolean,
  ): Promise<PolyNode[]> {
    const seeds = await this.toArray()
    const seedIds = seeds.map(node => node.id)
    const edges = direction === 'out'
      ? await this.edgesBySources(seedIds, edgeType)
      : await this.edgesByTargets(seedIds, edgeType)
    const connectedIds = [...new Set(edges.map(edge => direction === 'out' ? edge.target : edge.source))]
    return (await this.adapter.getNodes(connectedIds))
      .map(n => restoreNode(n, this.transform, this.sidecarData))
      .filter(node => !predicate || predicate(node))
  }
}
