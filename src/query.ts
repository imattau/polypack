import type { PolyNode, DataTransform, IndexDefinition, QueryExplain, QueryMetrics } from './types.js'
import { cosineSimilarity } from './vector-index.js'
import { activationScoreOf, assertFiniteVector, assertNonNegativeInteger, cloneData, clonePolyNode } from './utils.js'

/**
 * Native query executor hook. Given a query-plan IR, serialized nodes, and
 * serialized edges, returns ordered node ids or `null` to fall back to the
 * TypeScript pipeline. Registered explicitly (e.g. by the polypack-native
 * package) so browsers and packages without the native binary keep the
 * TypeScript path.
 */
export type NativeQueryExecutor = (
  plan: Record<string, unknown>,
  nodes: unknown[],
  edges: unknown[],
) => string[] | null

let nativeQueryExecutor: NativeQueryExecutor | null = null

/** Register (or clear) the native query executor. */
export function setNativeQueryExecutor(fn: NativeQueryExecutor | null): void {
  nativeQueryExecutor = fn
}

/** Whether a native query executor is currently registered. */
export function isNativeQueryExecutorActive(): boolean {
  return nativeQueryExecutor !== null
}

type EdgeEntry = { id: string; target: string; type: string; data?: Record<string, unknown> }
type EdgeIndex = Map<string, Map<string, EdgeEntry>>
type CandidateResolver = (
  nodeTypes: string[] | undefined,
  attributes: Record<string, unknown> | undefined,
  ranges?: Record<string, { above?: number; below?: number }>,
) => string[] | undefined

interface TraversalStep {
  edgeType: string
  depth: number
  direction: 'out' | 'in'
}

export type AggregateOp = 'sum' | 'avg' | 'min' | 'max' | 'count'

export interface AggregateResult {
  value: number
  count: number
}

export interface GroupedRow {
  key: string
  value: number
  count: number
}

/** Mutable fluent query builder created by {@link PolyGraph.query}. */
export class GraphQuery {
  private opts: {
    nodeTypes?: string[]
    attributes?: Record<string, unknown>
    attributeRanges?: Record<string, { above?: number; below?: number }>
    edgeType?: string
    edgeTarget?: string
    edgeSource?: string
    orderBy?: { field: string; direction: 'asc' | 'desc' }
    limit?: number
    offset?: number
    afterSteps?: TraversalStep[]
    similarVector?: { vector: number[]; threshold: number; topK?: number }
    joinFilters?: Array<(node: PolyNode) => boolean>
    activationAbove?: number
    activationOrder?: 'asc' | 'desc'
  } = {}

  private nodes: Map<string, PolyNode>
  private edges: EdgeIndex
  private nodeToEdgeMap: Map<string, Set<string>>
  private transform?: DataTransform
  private sidecarData: Map<string, unknown>
  private indexes: IndexDefinition[]
  private candidateResolver?: CandidateResolver
  private onMetrics?: (metrics: QueryMetrics) => void

  constructor(
    nodes: Map<string, PolyNode>,
    edges: EdgeIndex,
    nodeToEdgeMap: Map<string, Set<string>>,
    transform?: DataTransform,
    sidecarData?: Map<string, unknown>,
    indexes: IndexDefinition[] = [],
    candidateResolver?: CandidateResolver,
    onMetrics?: (metrics: QueryMetrics) => void,
  ) {
    this.nodes = nodes
    this.edges = edges
    this.nodeToEdgeMap = nodeToEdgeMap
    this.transform = transform
    this.sidecarData = sidecarData ?? new Map()
    this.indexes = indexes
    this.candidateResolver = candidateResolver
    this.onMetrics = onMetrics
  }

  private readNode(node: PolyNode): PolyNode {
    const cloned = clonePolyNode(node)
    if (!this.transform?.deserialize) return cloned
    const sidecar = this.sidecarData.get(cloned.id)
    cloned.data = this.transform.deserialize(cloned.data, sidecar) as Record<string, unknown>
    return cloned
  }

  where(field: string, value: unknown): this {
    const attrs = this.opts.attributes ?? {}
    attrs[field] = value
    this.opts.attributes = attrs
    return this
  }

  whereAttribute(name: string, value: unknown): this {
    const attrs = this.opts.attributes ?? {}
    attrs[name] = value
    this.opts.attributes = attrs
    return this
  }

  whereAttributeRange(name: string, range: { above?: number; below?: number }): this {
    if (range.above !== undefined && !Number.isFinite(range.above)) throw new RangeError('Range above must be finite')
    if (range.below !== undefined && !Number.isFinite(range.below)) throw new RangeError('Range below must be finite')
    const ranges = this.opts.attributeRanges ?? {}
    ranges[name] = range
    this.opts.attributeRanges = ranges
    return this
  }

  whereNodeType(...types: string[]): this {
    this.opts.nodeTypes = types
    return this
  }

  whereEdge(type: string, target?: string): this {
    if (!type) throw new TypeError('Edge type must not be empty')
    this.opts.edgeType = type
    if (target) this.opts.edgeTarget = target
    return this
  }

  whereEdgeSource(source: string): this {
    if (!source) throw new TypeError('Edge source must not be empty')
    this.opts.edgeSource = source
    return this
  }

  orderBy(field: string, direction: 'asc' | 'desc' = 'asc'): this {
    this.opts.orderBy = { field, direction }
    return this
  }

  /** Keep only nodes whose current (decay-corrected) activation exceeds `above`. */
  whereActivated(above: number): this {
    if (!Number.isFinite(above)) throw new RangeError('above must be finite')
    this.opts.activationAbove = above
    return this
  }

  /** Order results by current (decay-corrected) activation instead of a data field. */
  orderByActivation(direction: 'asc' | 'desc' = 'desc'): this {
    this.opts.activationOrder = direction
    return this
  }

  limit(n: number): this {
    assertNonNegativeInteger(n, 'limit')
    this.opts.limit = n
    return this
  }

  offset(n: number): this {
    assertNonNegativeInteger(n, 'offset')
    this.opts.offset = n
    return this
  }

  traverse(edgeType: string, depth: number, direction: 'out' | 'in' = 'out'): this {
    if (!edgeType) throw new TypeError('Edge type must not be empty')
    assertNonNegativeInteger(depth, 'depth')
    const steps = this.opts.afterSteps ?? []
    steps.push({ edgeType, depth, direction })
    this.opts.afterSteps = steps
    return this
  }

  similarTo(vector: number[], threshold = 0, topK?: number): this {
    assertFiniteVector(vector, 'query vector')
    if (!Number.isFinite(threshold)) throw new RangeError('threshold must be finite')
    if (topK !== undefined) assertNonNegativeInteger(topK, 'topK')
    this.opts.similarVector = { vector, threshold, topK }
    return this
  }

  private match(node: PolyNode): boolean {
    if (this.opts.nodeTypes && !this.opts.nodeTypes.includes(node.type)) return false

    if (this.opts.attributes) {
      for (const [key, val] of Object.entries(this.opts.attributes)) {
        const nodeVal = (node.data as Record<string, unknown>)[key]
        if (key === 'type' && node.type !== val) return false
        if (key !== 'type' && nodeVal !== val) return false
      }
    }

    if (this.opts.attributeRanges) {
      for (const [key, range] of Object.entries(this.opts.attributeRanges)) {
        const val = (node.data as Record<string, unknown>)[key] as number | undefined
        if (val === undefined) return false
        if (range.above !== undefined && val <= range.above) return false
        if (range.below !== undefined && val >= range.below) return false
      }
    }

    if (this.opts.edgeType) {
      const nodeEdges = this.edges.get(node.id)
      if (!nodeEdges) return false
      let matched = false
      for (const e of nodeEdges.values()) {
        if (e.type === this.opts.edgeType && (!this.opts.edgeTarget || e.target === this.opts.edgeTarget)) {
          matched = true
          break
        }
      }
      if (!matched) return false
    }

    if (this.opts.edgeSource) {
      const nodeEdges = this.edges.get(this.opts.edgeSource)
      if (!nodeEdges) return false
      let matched = false
      for (const e of nodeEdges.values()) {
        if (e.target === node.id && (!this.opts.edgeType || e.type === this.opts.edgeType)) {
          matched = true
          break
        }
      }
      if (!matched) return false
    }

    if (this.opts.joinFilters) {
      for (const filter of this.opts.joinFilters) {
        if (!filter(node)) return false
      }
    }

    if (this.opts.activationAbove !== undefined && activationScoreOf(node) < this.opts.activationAbove) {
      return false
    }

    return true
  }

  private getSourceNodes(): PolyNode[] {
    const ids = new Set<string>()

    if (this.opts.edgeSource && this.opts.edgeType) {
      const sourceEdges = this.edges.get(this.opts.edgeSource)
      if (sourceEdges) {
        for (const e of sourceEdges.values()) {
          if (e.type === this.opts.edgeType) ids.add(e.target)
        }
      }
    } else if (this.opts.edgeTarget && this.opts.edgeType) {
      for (const [nodeId, nodeEdges] of this.edges) {
        for (const e of nodeEdges.values()) {
          if (e.type === this.opts.edgeType && e.target === this.opts.edgeTarget) {
            ids.add(nodeId)
          }
        }
      }
    }

    if (ids.size > 0) {
      return [...ids].map(id => this.nodes.get(id)).filter((n): n is PolyNode => !!n)
    }

    const indexedIds = this.candidateResolver?.(this.opts.nodeTypes, this.opts.attributes, this.opts.attributeRanges)
    if (indexedIds) return indexedIds.map(id => this.nodes.get(id)).filter((n): n is PolyNode => !!n)

    const allNodes = [...this.nodes.values()]
    if (this.opts.nodeTypes) return allNodes.filter(n => this.opts.nodeTypes!.includes(n.type))
    if (this.opts.edgeType) return allNodes
    return allNodes
  }

  private bfs(seeds: string[], step: TraversalStep): Set<string> {
    const visited = new Set<string>(seeds)
    let frontier = [...seeds]
    for (let d = 0; d < step.depth; d++) {
      if (frontier.length === 0) break
      const next: string[] = []
      for (const id of frontier) {
        if (step.direction === 'out') {
          const outEdges = this.edges.get(id)
          if (outEdges) {
            for (const e of outEdges.values()) {
              if (e.type === step.edgeType && !visited.has(e.target)) {
                visited.add(e.target)
                next.push(e.target)
              }
            }
          }
        } else {
          const sources = this.nodeToEdgeMap.get(id)
          if (sources) {
            for (const src of sources) {
              if (visited.has(src)) continue
              const srcEdges = this.edges.get(src)
              if (srcEdges) {
                for (const e of srcEdges.values()) {
                  if (e.type === step.edgeType && e.target === id && !visited.has(src)) {
                    visited.add(src)
                    next.push(src)
                    break
                  }
                }
              }
            }
          }
        }
      }
      frontier = next
    }
    return visited
  }

  private applyTraversals(ids: string[]): Set<string> {
    if (!this.opts.afterSteps || this.opts.afterSteps.length === 0) return new Set(ids)
    let current = new Set(ids)
    for (const step of this.opts.afterSteps) {
      current = this.bfs([...current], step)
    }
    return current
  }

  private resolve(ids: Set<string>): PolyNode[] {
    const result: PolyNode[] = []
    for (const id of ids) {
      const node = this.nodes.get(id)
      if (node) result.push(node)
    }
    return result
  }

  /**
   * Express this query as the shared query-plan IR, or `null` when it cannot
   * be represented (join predicates are host callbacks and not serialisable).
   */
  private toPlan(): Record<string, unknown> | null {
    if (this.opts.joinFilters && this.opts.joinFilters.length > 0) return null
    if (this.opts.edgeSource && !this.opts.edgeType) return null
    if (this.opts.activationAbove !== undefined || this.opts.activationOrder !== undefined) return null
    const plan: Record<string, unknown> = {}
    if (this.opts.nodeTypes) plan.nodeTypes = [...this.opts.nodeTypes]
    if (this.opts.attributes) {
      plan.attributes = Object.entries(this.opts.attributes).map(([field, value]) => ({
        field,
        operator: 'eq',
        value,
      }))
    }
    if (this.opts.attributeRanges) {
      const ranges = Object.entries(this.opts.attributeRanges).map(([field, r]) => ({
        field,
        operator: 'range',
        above: r.above,
        below: r.below,
      }))
      plan.attributes = [...(plan.attributes as Array<Record<string, unknown>> ?? []), ...ranges]
    }
    if (this.opts.edgeType) {
      const ef: Record<string, unknown> = { type: this.opts.edgeType }
      if (this.opts.edgeTarget) ef.target = this.opts.edgeTarget
      if (this.opts.edgeSource) ef.source = this.opts.edgeSource
      plan.edgeFilter = ef
    }
    if (this.opts.afterSteps?.length) {
      plan.traversal = this.opts.afterSteps.map(s => ({ edgeType: s.edgeType, direction: s.direction, depth: s.depth }))
    }
    if (this.opts.similarVector) {
      const s: Record<string, unknown> = {
        vector: [...this.opts.similarVector.vector],
        threshold: this.opts.similarVector.threshold,
      }
      if (this.opts.similarVector.topK !== undefined) s.topK = this.opts.similarVector.topK
      plan.similarity = s
    }
    if (this.opts.orderBy) plan.order = { field: this.opts.orderBy.field, direction: this.opts.orderBy.direction }
    if (this.opts.offset !== undefined) plan.offset = this.opts.offset
    if (this.opts.limit !== undefined) plan.limit = this.opts.limit
    return plan
  }

  private serializeNodes(): unknown[] {
    return [...this.nodes.values()].map(n => ({
      id: n.id,
      type: n.type,
      data: n.data,
      vector: n.vector ? [...n.vector] : null,
      insertedAt: n.insertedAt,
      updatedAt: n.updatedAt,
      activation: n.activation ? { ...n.activation } : undefined,
    }))
  }

  private serializeEdges(): unknown[] {
    const out: unknown[] = []
    for (const [source, edgeMap] of this.edges) {
      for (const e of edgeMap.values()) {
        out.push({
          id: e.id,
          source,
          target: e.target,
          type: e.type,
          data: e.data ?? null,
          createdAt: 0,
        })
      }
    }
    return out
  }

  private runNative(): string[] | null {
    const executor = nativeQueryExecutor
    if (!executor) return null
    const plan = this.toPlan()
    if (!plan) return null
    try {
      return executor(plan, this.serializeNodes(), this.serializeEdges())
    } catch {
      return null
    }
  }

  /** Explain the selected source index and the remaining query stages. */
  explain(): QueryExplain {
    const fields = new Set([
      ...Object.keys(this.opts.attributes ?? {}),
      ...Object.keys(this.opts.attributeRanges ?? {}),
    ])
    const nodeType = this.opts.nodeTypes?.length === 1 ? this.opts.nodeTypes[0] : undefined
    const index = this.indexes.find(candidate =>
      (!candidate.nodeType || candidate.nodeType === nodeType) &&
      candidate.fields.every(field => fields.has(field) || fields.has(field.replace(/^data\./, ''))),
    )
    const stages = [index ? `property-index(${index.name})` : 'record-scan']
    if (this.opts.nodeTypes?.length) stages.push(`type-filter(${this.opts.nodeTypes.join(',')})`)
    if (this.opts.afterSteps?.length) stages.push(`traversal(depth=${Math.max(...this.opts.afterSteps.map(step => step.depth))})`)
    if (this.opts.orderBy) stages.push(`order(${this.opts.orderBy.field},${this.opts.orderBy.direction})`)
    if (this.opts.limit !== undefined) stages.push(`limit(${this.opts.limit})`)
    return {
      index: index?.name,
      stages,
      loadedRecords: this.nodes.size,
      estimatedCost: Math.max(1, this.nodes.size * (index ? 0.25 : 1)),
    }
  }

  toArray(): PolyNode[] {
    const started = Date.now()
    const finish = (results: PolyNode[]): PolyNode[] => {
      const index = this.explain().index
      this.onMetrics?.({ durationMs: Date.now() - started, scannedRecords: this.nodes.size, index })
      return results
    }
    const nativeIds = this.runNative()
    if (nativeIds !== null) {
      return finish(nativeIds
        .map(id => this.nodes.get(id))
        .filter((n): n is PolyNode => !!n)
        .map(node => this.readNode(node)))
    }

    let results = this.getSourceNodes().filter(n => this.match(n))
    if (results.length === 0) return finish([])

    if (this.opts.afterSteps && this.opts.afterSteps.length > 0) {
      const ids = results.map(n => n.id)
      const expanded = this.applyTraversals(ids)
      results = this.resolve(expanded)
    }

    if (this.opts.orderBy) {
      const { field, direction } = this.opts.orderBy
      results = [...results].sort((a, b) => {
        const av = (a.data as Record<string, unknown>)[field] as number ?? 0
        const bv = (b.data as Record<string, unknown>)[field] as number ?? 0
        return direction === 'asc' ? av - bv : bv - av
      })
    }

    if (this.opts.activationOrder) {
      const now = Date.now()
      results = [...results].sort((a, b) => {
        const av = activationScoreOf(a, now)
        const bv = activationScoreOf(b, now)
        return this.opts.activationOrder === 'desc' ? bv - av : av - bv
      })
    }

    if (this.opts.similarVector) {
      const { vector, threshold, topK } = this.opts.similarVector
      const scored = results
        .filter((n): n is PolyNode & { vector: Float64Array } => n.vector !== undefined)
        .map(n => ({
          node: n,
          score: cosineSimilarity(vector, n.vector),
        }))
        .filter(s => s.score >= threshold)
        .sort((a, b) => b.score - a.score)
      results = topK !== undefined ? scored.slice(0, topK).map(s => s.node) : scored.map(s => s.node)
    }

    if (this.opts.offset !== undefined) results = results.slice(this.opts.offset)
    if (this.opts.limit !== undefined) results = results.slice(0, this.opts.limit)

    return finish(results.map(node => this.readNode(node)))
  }

  first(): PolyNode | null {
    return this.toArray()[0] ?? null
  }

  count(): number {
    if (this.opts.afterSteps?.length || this.opts.similarVector ||
        this.opts.offset !== undefined || this.opts.limit !== undefined) {
      return this.toArray().length
    }
    return this.getSourceNodes().filter(n => this.match(n)).length
  }

  ids(): string[] {
    return this.toArray().map(n => n.id)
  }

  // ── Relational extensions ──

  /** Project selected fields. Terminal — returns plain records instead of nodes. */
  pluck(...fields: string[]): Record<string, unknown>[] {
    return this.toArray().map(n => {
      const row: Record<string, unknown> = { id: n.id, type: n.type }
      const d = n.data as Record<string, unknown>
      for (const f of fields) {
        row[f] = d[f]
      }
      return row
    })
  }

  /** Aggregate a numeric field across all matched nodes. Terminal. */
  aggregate(field: string, op: AggregateOp): AggregateResult {
    const nodes = this.toArray()
    const values = nodes
      .map(n => (n.data as Record<string, unknown>)[field])
      .filter((v): v is number => typeof v === 'number' && Number.isFinite(v))
    if (values.length === 0) return { value: 0, count: 0 }
    let value: number
    switch (op) {
      case 'sum': value = values.reduce((a, b) => a + b, 0); break
      case 'avg': value = values.reduce((a, b) => a + b, 0) / values.length; break
      case 'min': value = Math.min(...values); break
      case 'max': value = Math.max(...values); break
      case 'count': value = values.length; break
    }
    return { value, count: values.length }
  }

  /** Group by a field and aggregate. Terminal. */
  groupAggregate(field: string, op: AggregateOp, groupByField: string): GroupedRow[] {
    const nodes = this.toArray()
    const groups = new Map<string, number[]>()

    for (const n of nodes) {
      const key = String((n.data as Record<string, unknown>)[groupByField] ?? 'null')
      const val = (n.data as Record<string, unknown>)[field]
      if (op === 'count' ? val !== undefined && val !== null : typeof val === 'number' && Number.isFinite(val)) {
        if (!groups.has(key)) groups.set(key, [])
        groups.get(key)!.push(op === 'count' ? 1 : val as number)
      }
    }

    return [...groups.entries()].map(([key, values]) => {
      let value: number
      switch (op) {
        case 'sum': value = values.reduce((a, b) => a + b, 0); break
        case 'avg': value = values.reduce((a, b) => a + b, 0) / values.length; break
        case 'min': value = Math.min(...values); break
        case 'max': value = Math.max(...values); break
        case 'count': value = values.length; break
      }
      return { key, value, count: values.length }
    })
  }

  /** Filter groups after groupAggregate by predicate. */
  having(groups: GroupedRow[], predicate: (row: GroupedRow) => boolean): GroupedRow[] {
    return groups.filter(predicate)
  }

  /**
   * Constrain results to nodes connected via `edgeType`. Unlike `whereEdge`
   * (which checks for a specific target ID), this checks for ANY edge of the
   * given type and optionally filters by the connected node's properties.
   * Chainable — returns `this`.
   *
   *   // All users who have rated a sci-fi book
   *   graph.query().whereNodeType('user')
   *     .join('RATED', 'out', b => (b.data as any).genre === 'sci-fi')
   *     .toArray()
   *
   *   // All books that have been rated by Alice
   *   graph.query().whereNodeType('book')
   *     .join('RATED', 'in', u => (u.data as any).name === 'Alice')
   *     .toArray()
   */
  join(edgeType: string, direction: 'out' | 'in' = 'out', predicate?: (connectedNode: PolyNode) => boolean): this {
    if (!edgeType) throw new TypeError('Edge type must not be empty')
    const filters = this.opts.joinFilters ?? []
    filters.push((node: PolyNode) => {
      const connected = direction === 'out'
        ? [...(this.edges.get(node.id)?.values() ?? [])]
            .filter((e: EdgeEntry) => e.type === edgeType)
            .map(e => this.nodes.get(e.target))
            .filter((n): n is PolyNode => !!n)
            .map(n => this.readNode(n))
        : this.getEdgeSourcesForTarget(node.id, edgeType)
            .map(id => this.nodes.get(id))
            .filter((n): n is PolyNode => !!n)
            .map(n => this.readNode(n))
      if (connected.length === 0) return false
      if (predicate) return connected.some(predicate)
      return true
    })
    this.opts.joinFilters = filters
    return this
  }

  /**
   * Group nodes by nearest-centroid vector clustering. Each node is assigned
   * to the group whose `centroid` vector is most similar (cosine) to the
   * node's own vector. Nodes below `threshold` go to an uncategorized group.
   * Nodes without vectors are excluded. Terminal.
   *
   *   graph.query().whereNodeType('book')
   *     .groupByVector(
   *       [
   *         { key: 'sci-fi', centroid: [0.9, 0.1, 0.2] },
   *         { key: 'fantasy', centroid: [0.3, 0.8, 0.7] },
   *       ],
   *       'rating', 'avg', 0.4
   *     )
   */
  groupByVector(
    groups: Array<{ key: string; centroid: number[] }>,
    field: string,
    op: AggregateOp,
    threshold = 0,
  ): GroupedRow[] {
    const nodes = this.toArray()
    const clusters = new Map<string, number[]>()

    for (const n of nodes) {
      if (!n.vector) continue
      let bestKey = 'null'
      let bestScore = threshold
      for (const g of groups) {
        const score = cosineSimilarity(g.centroid, n.vector)
        if (score > bestScore) {
          bestScore = score
          bestKey = g.key
        }
      }
      const val = (n.data as Record<string, unknown>)[field]
      if (op === 'count' ? val !== undefined && val !== null : typeof val === 'number' && Number.isFinite(val)) {
        if (!clusters.has(bestKey)) clusters.set(bestKey, [])
        clusters.get(bestKey)!.push(op === 'count' ? 1 : val as number)
      }
    }

    return [...clusters.entries()].map(([key, values]) => {
      let value: number
      switch (op) {
        case 'sum': value = values.reduce((a, b) => a + b, 0); break
        case 'avg': value = values.reduce((a, b) => a + b, 0) / values.length; break
        case 'min': value = Math.min(...values); break
        case 'max': value = Math.max(...values); break
        case 'count': value = values.length; break
      }
      return { key, value, count: values.length }
    })
  }

  private getEdgeSourcesForTarget(targetId: string, edgeType: string): string[] {
    const sources = this.nodeToEdgeMap.get(targetId)
    if (!sources) return []
    return [...sources].filter(src => {
      const edges = this.edges.get(src)
      if (!edges) return false
      for (const e of edges.values()) {
        if (e.type === edgeType && e.target === targetId) return true
      }
      return false
    })
  }

  uniqueKeys(field: string): unknown[] {
    const keys = new Set<unknown>()
    for (const node of this.nodes.values()) {
      const val = (node.data as Record<string, unknown>)[field]
      if (val !== undefined) keys.add(cloneData(val))
    }
    return [...keys]
  }

  collect(
    edgeType: string,
    direction: 'out' | 'in' = 'out',
    predicate?: (node: PolyNode) => boolean
  ): PolyNode[] {
    const seeds = this.toArray()
    const collected: PolyNode[] = []
    const seen = new Set<string>()

    for (const seed of seeds) {
      if (direction === 'out') {
        const outEdges = this.edges.get(seed.id)
        if (outEdges) {
          for (const e of outEdges.values()) {
            if (e.type !== edgeType) continue
            if (seen.has(e.target)) continue
            seen.add(e.target)
            const node = this.nodes.get(e.target)
            if (node) {
              const snapshot = this.readNode(node)
              if (!predicate || predicate(snapshot)) collected.push(snapshot)
            }
          }
        }
      } else {
        const sources = this.nodeToEdgeMap.get(seed.id)
        if (sources) {
          for (const src of sources) {
            if (seen.has(src)) continue
            const srcEdges = this.edges.get(src)
            if (!srcEdges) continue
            let matched = false
            for (const e of srcEdges.values()) {
              if (e.type === edgeType && e.target === seed.id) { matched = true; break }
            }
            if (!matched) continue
            seen.add(src)
            const node = this.nodes.get(src)
            if (node) {
              const snapshot = this.readNode(node)
              if (!predicate || predicate(snapshot)) collected.push(snapshot)
            }
          }
        }
      }
    }

    return collected
  }
}
