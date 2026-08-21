/**
 * Conformance fixture runner.
 *
 * Loads language-neutral JSON fixtures from `fixtures/` and drives the public
 * TypeScript API only (PolyGraph, GraphQuery, VectorIndex, HNSWIndex), so the
 * same fixtures can be consumed by the Rust and Python implementations later.
 *
 * Fixture shape:
 *
 *   {
 *     schemaVersion: 1,
 *     name: string,
 *     group: string,
 *     orphanAware?: boolean,
 *     graphOptions?: { hotCacheMax?: number },
 *     setup?: { nodes: SerializedNode[], edges: EdgeInput[] },
 *     operations: Operation[],
 *     expect: Expectations
 *   }
 */
import { readFileSync, readdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { PolyGraph, HNSWIndex, cosineSimilarity, ConflictError, mergeActivation } from '../../src/index'
import type { EdgeOwnership, PolyNode, SerializedNode, VectorIndexLike, NodeActivation } from '../../src/index'
import { decayActivationState, reinforceActivation, suppressActivation } from '../../src/utils'
import type { QueryPlan } from './query-plan'

export type ErrorCode =
  | 'invalid_argument'
  | 'conflict'
  | 'dimension_mismatch'
  | 'range_out_of_bounds'
  | 'closed'
  | 'format_version'
  | 'corrupt_data'
  | 'storage'
  | 'not_implemented'

export interface EdgeInput {
  source: string
  type: string
  target: string
  data?: Record<string, unknown>
  ownership?: EdgeOwnership
}

interface NodeInput {
  id: string
  type: string
  data: Record<string, unknown>
  vector?: number[] | null
  insertedAt: number
  updatedAt: number
  revision?: number
}

export type Operation =
  | { op: 'addNode'; node: NodeInput; expectError?: ErrorCode }
  | { op: 'updateNode'; id: string; data: Record<string, unknown>; vector?: number[]; expectedRevision?: number; expectError?: ErrorCode }
  | { op: 'patchNode'; id: string; patch: { set?: Record<string, unknown>; unset?: string[]; increment?: Record<string, number>; compareAndSet?: Record<string, { expected: unknown; value: unknown }> }; expectedRevision?: number; expectError?: ErrorCode }
  | { op: 'addEdge'; source: string; type: string; target: string; data?: Record<string, unknown>; ownership?: EdgeOwnership; expectError?: ErrorCode }
  | { op: 'removeNode'; id: string }
  | { op: 'removeEdges'; source: string; type?: string; target?: string }
  | { op: 'hnswAdd'; id: string; vector: number[] }
  | { op: 'hnswRemove'; id: string }
  | { op: 'hnswUpdate'; id: string; vector: number[] }
  | { op: 'mutateDetached'; id: string }

export interface Expectations {
  presentNodeIds?: string[]
  absentNodeIds?: string[]
  nodeCount?: number
  loadedSize?: number
  nodeData?: Record<string, Record<string, unknown>>
  nodeVector?: Record<string, number[]>
  nodeRevision?: Record<string, number>
  edgeTargets?: Array<{ source: string; type: string; targets: string[] }>
  orphanEvents?: string[]
  queries?: Array<{ plan: QueryPlan; resultIds: string[] }>
  exactSearches?: Array<{ vector: number[]; topK: number; threshold?: number; resultIds: string[]; expectError?: ErrorCode }>
  hnswSearches?: Array<{ vector: number[]; topK: number; minOverlap?: number; resultIds: string[]; expectError?: ErrorCode }>
  hnswVector?: Record<string, number[]>
  hnswSize?: number
  aggregate?: { plan?: QueryPlan; field: string; op: 'sum' | 'avg' | 'min' | 'max' | 'count'; value: number; count: number }
}

export type ActivationCase =
  | {
      name: string
      kind: 'decay'
      input: NodeActivation
      now: number
      scoreHalfLifeMs?: number
      importanceHalfLifeMs?: number
      expect: NodeActivation
    }
  | {
      name: string
      kind: 'reinforce'
      previous: NodeActivation | null
      delta: number
      now: number
      context?: string
      expect: NodeActivation
    }
  | { name: string; kind: 'merge'; existing: NodeActivation; incoming: NodeActivation; now: number; expect: NodeActivation }
  | {
      name: string
      kind: 'suppress'
      previous: NodeActivation | null
      delta: number
      now: number
      inhibitionHalfLifeMs?: number
      expect: NodeActivation
    }

export interface Fixture {
  schemaVersion: number
  name: string
  group: string
  orphanAware?: boolean
  graphOptions?: { hotCacheMax?: number }
  setup?: { nodes?: NodeInput[]; edges?: EdgeInput[] }
  operations?: Operation[]
  expect?: Expectations
  /** Pure activation-math cases (group `activation`) — no PolyGraph instance involved. */
  activationCases?: ActivationCase[]
}

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'fixtures', 'conformance')

export function loadFixtures(): Fixture[] {
  return readdirSync(FIXTURES_DIR)
    .filter(f => f.endsWith('.json'))
    .sort()
    .map(f => JSON.parse(readFileSync(join(FIXTURES_DIR, f), 'utf8')) as Fixture)
}

export class OrphanAwareGraph extends PolyGraph {
  readonly orphaned: string[] = []
  protected onOrphan(id: string): void {
    this.orphaned.push(id)
  }
}

export class ConformanceError extends Error {}

function mapError(err: unknown): ErrorCode {
  if (err instanceof ConflictError) return 'conflict'
  if (err instanceof TypeError) return 'invalid_argument'
  if (err instanceof RangeError) {
    return /dimension/i.test(String((err as Error).message)) ? 'dimension_mismatch' : 'range_out_of_bounds'
  }
  return 'storage'
}

export function fixtureError(e: Error): string {
  return `  error: ${e instanceof Error ? e.stack ?? e.message : String(e)}`
}

export function assertError(err: unknown, expected?: ErrorCode): void {
  if (!expected) {
    if (err !== undefined) throw new ConformanceError(`operation threw unexpectedly: ${fixtureError(err as Error)}`)
    return
  }
  if (err === undefined) throw new ConformanceError(`expected error ${expected} but operation succeeded`)
  const got = mapError(err)
  if (got !== expected) {
    throw new ConformanceError(`expected error ${expected}, got ${got} (${err instanceof Error ? err.message : String(err)})`)
  }
}

function toGraphNode(n: NodeInput): PolyNode {
  const node: PolyNode = {
    id: n.id,
    type: n.type,
    data: n.data,
    insertedAt: n.insertedAt,
    updatedAt: n.updatedAt,
    revision: n.revision,
  }
  if (n.vector) node.vector = new Float64Array(n.vector)
  return node
}

function runOperation(graph: PolyGraph, hnsw: HNSWIndex | null, op: Operation): void {
  switch (op.op) {
    case 'addNode': {
      let err: unknown
      try {
        graph.addNode(toGraphNode(op.node))
      } catch (e) {
        err = e
      }
      assertError(err, op.expectError)
      break
    }
    case 'updateNode': {
      let err: unknown
      try {
        graph.updateNode(op.id, op.data, op.vector ? new Float64Array(op.vector) : undefined, undefined, { expectedRevision: op.expectedRevision })
      } catch (e) {
        err = e
      }
      assertError(err, op.expectError)
      break
    }
    case 'patchNode': {
      let err: unknown
      try {
        graph.patchNode(op.id, op.patch, { expectedRevision: op.expectedRevision })
      } catch (e) {
        err = e
      }
      assertError(err, op.expectError)
      break
    }
    case 'addEdge': {
      let err: unknown
      try {
        graph.addEdge(op.source, op.type, op.target, op.data, op.ownership)
      } catch (e) {
        err = e
      }
      assertError(err, op.expectError)
      break
    }
    case 'removeNode':
      graph.removeNode(op.id)
      break
    case 'removeEdges':
      graph.removeEdges(op.source, op.type, op.target)
      break
    case 'hnswAdd':
      if (!hnsw) throw new ConformanceError('hnswAdd requires hnswConfig in the fixture')
      hnsw.add(op.id, op.vector)
      break
    case 'hnswRemove':
      if (!hnsw) throw new ConformanceError('hnswRemove requires hnswConfig in the fixture')
      hnsw.remove(op.id)
      break
    case 'hnswUpdate':
      if (!hnsw) throw new ConformanceError('hnswUpdate requires hnswConfig in the fixture')
      hnsw.update(op.id, op.vector)
      break
    case 'mutateDetached': {
      const node = graph.getNode(op.id)
      if (!node) throw new ConformanceError(`mutateDetached: node ${op.id} missing`)
      node.data = { ...node.data, __tampered: true }
      if (node.vector) node.vector[0] = 999
      break
    }
  }
}

/** Translate a query-plan IR into the fluent GraphQuery builder. */
function applyPlan(query: ReturnType<PolyGraph['query']>, plan: QueryPlan): void {
  if (plan.nodeTypes?.length) query.whereNodeType(...plan.nodeTypes)
  for (const attr of plan.attributes ?? []) {
    if (attr.operator === 'eq') query.where(attr.field, attr.value)
    else query.whereAttributeRange(attr.field, { above: attr.above, below: attr.below })
  }
  if (plan.edgeFilter) {
    if (plan.edgeFilter.source) query.whereEdgeSource(plan.edgeFilter.source)
    query.whereEdge(plan.edgeFilter.type, plan.edgeFilter.target)
  }
  for (const step of plan.traversal ?? []) {
    query.traverse(step.edgeType, step.depth, step.direction)
  }
  for (const j of plan.joins ?? []) {
    query.join(j.edgeType, j.direction)
  }
  if (plan.similarity) {
    query.similarTo(plan.similarity.vector, plan.similarity.threshold, plan.similarity.topK)
  }
  if (plan.order) query.orderBy(plan.order.field, plan.order.direction)
  if (plan.offset !== undefined) query.offset(plan.offset)
  if (plan.limit !== undefined) query.limit(plan.limit)
}

function assertExpectations(graph: PolyGraph, orphanGraph: OrphanAwareGraph | null, hnsw: HNSWIndex | null, expect: Expectations): void {
  if (expect.presentNodeIds) {
    for (const id of expect.presentNodeIds) {
      if (!graph.getNode(id)) throw new ConformanceError(`expected node ${id} to exist`)
    }
  }
  if (expect.absentNodeIds) {
    for (const id of expect.absentNodeIds) {
      if (graph.getNode(id)) throw new ConformanceError(`expected node ${id} to be absent`)
    }
  }
  if (expect.nodeCount !== undefined) {
    const count = graph.size
    if (count !== expect.nodeCount) throw new ConformanceError(`nodeCount ${count} !== ${expect.nodeCount}`)
  }
  if (expect.loadedSize !== undefined) {
    if (graph.loadedSize !== expect.loadedSize) {
      throw new ConformanceError(`loadedSize ${graph.loadedSize} !== ${expect.loadedSize}`)
    }
  }
  if (expect.nodeData) {
    for (const [id, fields] of Object.entries(expect.nodeData)) {
      const node = graph.getNode(id)
      if (!node) throw new ConformanceError(`nodeData: node ${id} missing`)
      for (const [k, v] of Object.entries(fields)) {
        if ((node.data as Record<string, unknown>)[k] !== v) {
          throw new ConformanceError(`nodeData[${id}].${k} = ${String((node.data as Record<string, unknown>)[k])} !== ${String(v)}`)
        }
      }
    }
  }
  if (expect.nodeVector) {
    for (const [id, vector] of Object.entries(expect.nodeVector)) {
      const node = graph.getNode(id)
      if (!node?.vector) throw new ConformanceError(`nodeVector: node ${id} has no vector`)
      if ([...node.vector].join(',') !== vector.join(',')) {
        throw new ConformanceError(`nodeVector[${id}] mismatch`)
      }
    }
  }
  if (expect.nodeRevision) {
    for (const [id, revision] of Object.entries(expect.nodeRevision)) {
      const node = graph.getNode(id)
      if (!node) throw new ConformanceError(`nodeRevision: node ${id} missing`)
      if (node.revision !== revision) throw new ConformanceError(`nodeRevision[${id}] ${node.revision} !== ${revision}`)
    }
  }
  if (expect.edgeTargets) {
    for (const { source, type, targets } of expect.edgeTargets) {
      const got = graph.getEdgeTargets(source, type).sort()
      if (got.join(',') !== [...targets].sort().join(',')) {
        throw new ConformanceError(`edgeTargets ${source}/${type}: ${got.join(',')} !== ${targets.join(',')}`)
      }
    }
  }
  if (expect.orphanEvents) {
    if (!orphanGraph) throw new ConformanceError('orphanEvents requires orphanAware fixture')
    if (orphanGraph.orphaned.join(',') !== expect.orphanEvents.join(',')) {
      throw new ConformanceError(`orphanEvents ${orphanGraph.orphaned.join(',')} !== ${expect.orphanEvents.join(',')}`)
    }
  }
  if (expect.queries) {
    for (const { plan, resultIds } of expect.queries) {
      const query = graph.query()
      applyPlan(query, plan)
      const got = query.ids()
      if (got.join(',') !== resultIds.join(',')) {
        throw new ConformanceError(`query result ${got.join(',')} !== ${resultIds.join(',')}`)
      }
    }
  }
  if (expect.exactSearches) {
    for (const { vector, topK, threshold, resultIds, expectError } of expect.exactSearches) {
      let err: unknown
      let got: string[] = []
      try {
        got = graph.vectors.query(vector, topK, threshold ?? 0).map(r => r.id)
      } catch (e) {
        err = e
      }
      if (expectError) {
        assertError(err, expectError)
      } else {
        assertError(err, undefined)
        if (got.join(',') !== resultIds.join(',')) {
          throw new ConformanceError(`exactSearch ${got.join(',')} !== ${resultIds.join(',')}`)
        }
      }
    }
  }
  if (expect.hnswSearches) {
    if (!hnsw) throw new ConformanceError('hnswSearch requires hnswConfig in the fixture')
    for (const { vector, topK, minOverlap, resultIds, expectError } of expect.hnswSearches) {
      let err: unknown
      let got: string[] = []
      try {
        got = hnsw.query(vector, topK, 0).map(r => r.id)
      } catch (e) {
        err = e
      }
      if (expectError) {
        assertError(err, expectError)
        continue
      }
      assertError(err, undefined)
      const min = minOverlap ?? topK
      const overlap = got.filter(id => resultIds.includes(id)).length
      if (overlap < min) {
        throw new ConformanceError(`hnswSearch overlap ${overlap} < ${min} (got ${got.join(',')}, expected ${resultIds.join(',')})`)
      }
    }
  }
  if (expect.hnswVector) {
    if (!hnsw) throw new ConformanceError('hnswVector requires hnswConfig in the fixture')
    for (const [id, vector] of Object.entries(expect.hnswVector)) {
      const got = hnsw.get(id)
      if (!got) throw new ConformanceError(`hnswVector: id ${id} missing`)
      if ([...got].join(',') !== vector.join(',')) {
        throw new ConformanceError(`hnswVector[${id}] mismatch`)
      }
    }
  }
  if (expect.hnswSize !== undefined) {
    if (!hnsw) throw new ConformanceError('hnswSize requires hnswConfig in the fixture')
    if (hnsw.size !== expect.hnswSize) throw new ConformanceError(`hnswSize ${hnsw.size} !== ${expect.hnswSize}`)
  }
  if (expect.aggregate) {
    const { plan, field, op, value, count } = expect.aggregate
    const query = graph.query()
    if (plan) applyPlan(query, plan)
    const result = query.aggregate(field, op)
    if (result.value !== value || result.count !== count) {
      throw new ConformanceError(`aggregate ${op}(${field}) = ${result.value} (${result.count}) !== ${value} (${count})`)
    }
  }
}

const ACTIVATION_EPSILON = 1e-9

function assertActivationEqual(name: string, got: NodeActivation, expect: NodeActivation): void {
  for (const key of ['score', 'importance', 'reinforcementCount', 'lastMeaningfulActivation'] as const) {
    if (Math.abs(got[key] - expect[key]) > ACTIVATION_EPSILON) {
      throw new ConformanceError(`${name}: ${key} = ${got[key]} !== ${expect[key]}`)
    }
  }
  const gotInhibition = got.inhibition ?? 0
  const expectInhibition = expect.inhibition ?? 0
  if (Math.abs(gotInhibition - expectInhibition) > ACTIVATION_EPSILON) {
    throw new ConformanceError(`${name}: inhibition = ${gotInhibition} !== ${expectInhibition}`)
  }
  const expectContext = expect.context ?? {}
  const gotContext = got.context ?? {}
  for (const key of Object.keys(expectContext)) {
    const gotScore = gotContext[key]?.score ?? 0
    const expectScore = expectContext[key].score
    if (Math.abs(gotScore - expectScore) > ACTIVATION_EPSILON) {
      throw new ConformanceError(`${name}: context[${key}].score = ${gotScore} !== ${expectScore}`)
    }
  }
  for (const key of Object.keys(gotContext)) {
    if (!(key in expectContext) && (gotContext[key]?.score ?? 0) > ACTIVATION_EPSILON) {
      throw new ConformanceError(`${name}: unexpected context[${key}] with score ${gotContext[key].score}`)
    }
  }
}

export function runActivationCases(cases: ActivationCase[]): void {
  for (const c of cases) {
    if (c.kind === 'decay') {
      const got = decayActivationState(c.input, c.now, c.scoreHalfLifeMs, c.importanceHalfLifeMs)
      assertActivationEqual(c.name, got, c.expect)
    } else if (c.kind === 'reinforce') {
      const got = reinforceActivation(c.previous ?? undefined, c.delta, c.now, undefined, c.context)
      assertActivationEqual(c.name, got, c.expect)
    } else if (c.kind === 'suppress') {
      const got = suppressActivation(c.previous ?? undefined, c.delta, c.now, c.inhibitionHalfLifeMs)
      assertActivationEqual(c.name, got, c.expect)
    } else {
      const got = mergeActivation(c.existing, c.incoming, c.now)
      assertActivationEqual(c.name, got, c.expect)
    }
  }
}

export function runFixture(fixture: Fixture, createIndex?: (onChange: (id: string) => void) => VectorIndexLike): void {
  if (fixture.activationCases) {
    runActivationCases(fixture.activationCases)
    return
  }
  const graph = fixture.orphanAware
    ? new OrphanAwareGraph(undefined, fixture.graphOptions?.hotCacheMax, undefined, undefined, createIndex)
    : new PolyGraph(undefined, fixture.graphOptions?.hotCacheMax, undefined, undefined, createIndex)
  const orphanGraph = fixture.orphanAware ? graph as OrphanAwareGraph : null

  const operations = fixture.operations ?? []
  const expect = fixture.expect ?? {}
  const needsHnsw = operations.some(op => op.op.startsWith('hnsw')) || (expect.hnswSearches?.length ?? 0) > 0
  const hnsw = needsHnsw ? new HNSWIndex(undefined, cosineSimilarity, { M: 16, efConstruction: 200, efSearch: 300 }) : null

  if (hnsw) {
    for (const node of fixture.setup?.nodes ?? []) {
      if (node.vector) hnsw.add(node.id, node.vector)
    }
  }

  for (const node of fixture.setup?.nodes ?? []) {
    graph.addNode(toGraphNode(node))
  }
  for (const edge of fixture.setup?.edges ?? []) {
    graph.addEdge(edge.source, edge.type, edge.target, edge.data, edge.ownership)
  }
  for (const op of operations) {
    runOperation(graph, hnsw, op)
  }
  assertExpectations(graph, orphanGraph, hnsw, expect)
}
