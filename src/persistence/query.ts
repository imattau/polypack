import type { IndexDefinition, SerializedNode } from '../types.js'
import type { PersistedNodeQuery } from './adapter.js'
import { QueryAbortedError } from '../query-errors.js'

export function assertQueryActive(signal?: AbortSignal): void {
  if (signal?.aborted) throw new QueryAbortedError()
}

function fieldValue(node: SerializedNode, field: string): unknown {
  if (field === 'type') return node.type
  const path = field.replace(/^data\./, '').split('.')
  let value: unknown = node.data
  for (const part of path) {
    if (!value || typeof value !== 'object') return undefined
    value = (value as Record<string, unknown>)[part]
  }
  return value
}

function queryField(query: PersistedNodeQuery, field: string): { exact?: unknown; range?: { above?: number; below?: number } } | undefined {
  const aliases = [field, field.replace(/^data\./, '')]
  for (const alias of aliases) {
    if (query.attributes && Object.prototype.hasOwnProperty.call(query.attributes, alias)) return { exact: query.attributes[alias] }
    if (query.attributeRanges?.[alias]) return { range: query.attributeRanges[alias] }
  }
  return undefined
}

function bucketKey(values: unknown[]): string {
  return JSON.stringify(values, (_key, value) => typeof value === 'bigint' ? `${value}n` : value)
}

/** Maintained candidate buckets for configurable persisted node indexes. */
export class SecondaryIndexBuckets {
  private definitions: IndexDefinition[] = []
  private buckets = new Map<string, Map<string, Set<string>>>()

  setDefinitions(definitions: IndexDefinition[]): void {
    this.definitions = definitions.map(definition => ({ ...definition, fields: [...definition.fields] }))
    this.buckets = new Map(this.definitions.map(definition => [definition.name, new Map()]))
  }

  add(node: SerializedNode): void {
    for (const definition of this.definitions) {
      if (definition.nodeType && definition.nodeType !== node.type) continue
      const values = definition.fields.map(field => fieldValue(node, field))
      if (definition.sparse && values.some(value => value === undefined)) continue
      const key = bucketKey(values)
      const buckets = this.buckets.get(definition.name)!
      let ids = buckets.get(key)
      if (!ids) { ids = new Set(); buckets.set(key, ids) }
      ids.add(node.id)
    }
  }

  remove(node: SerializedNode | undefined): void {
    if (!node) return
    for (const definition of this.definitions) {
      if (definition.nodeType && definition.nodeType !== node.type) continue
      const values = definition.fields.map(field => fieldValue(node, field))
      if (definition.sparse && values.some(value => value === undefined)) continue
      const buckets = this.buckets.get(definition.name)
      const ids = buckets?.get(bucketKey(values))
      ids?.delete(node.id)
      if (ids && ids.size === 0) buckets?.delete(bucketKey(values))
    }
  }

  rebuild(nodes: Iterable<SerializedNode>): void {
    for (const buckets of this.buckets.values()) buckets.clear()
    for (const node of nodes) this.add(node)
  }

  verify(nodes: Iterable<SerializedNode>): string[] {
    const expected = new SecondaryIndexBuckets()
    expected.setDefinitions(this.definitions)
    expected.rebuild(nodes)
    const errors: string[] = []
    for (const definition of this.definitions) {
      const actualBuckets = this.buckets.get(definition.name) ?? new Map()
      const expectedBuckets = expected.buckets.get(definition.name) ?? new Map()
      for (const [key, ids] of expectedBuckets) {
        if (definition.unique && ids.size > 1) errors.push(`unique index violation: ${definition.name}/${key}`)
        const actual = actualBuckets.get(key)
        if (!actual || actual.size !== ids.size || [...ids].some(id => !actual.has(id))) errors.push(`secondary index mismatch: ${definition.name}/${key}`)
      }
      for (const key of actualBuckets.keys()) if (!expectedBuckets.has(key)) errors.push(`stale secondary index entry: ${definition.name}/${key}`)
    }
    return errors
  }

  /** Return the intersection of all usable index candidates, or null for a scan. */
  candidates(query: PersistedNodeQuery): { ids: Set<string> | null; names: string[] } {
    if (query.nodeTypes && query.nodeTypes.length > 1) return { ids: null, names: [] }
    const candidates: Set<string>[] = []
    const names: string[] = []
    for (const definition of this.definitions) {
      if (definition.nodeType && (!query.nodeTypes || query.nodeTypes.length !== 1 || definition.nodeType !== query.nodeTypes[0])) continue
      const constraints = definition.fields.map(field => queryField(query, field))
      if (constraints.some(constraint => !constraint)) continue
      const buckets = this.buckets.get(definition.name)
      if (!buckets) continue
      const ids = new Set<string>()
      for (const [key, bucketIds] of buckets) {
        const values = JSON.parse(key) as unknown[]
        const matches = constraints.every((constraint, index) => {
          if (constraint?.exact !== undefined) return Object.is(values[index], constraint.exact)
          const value = values[index]
          if (typeof value !== 'number' || !Number.isFinite(value)) return false
          return (constraint?.range?.above === undefined || value > constraint.range.above) &&
            (constraint?.range?.below === undefined || value < constraint.range.below)
        })
        if (matches) for (const id of bucketIds) ids.add(id)
      }
      candidates.push(ids)
      names.push(definition.name)
    }
    if (candidates.length === 0) return { ids: null, names: [] }
    candidates.sort((a, b) => a.size - b.size)
    const result = new Set(candidates[0])
    for (const candidate of candidates.slice(1)) for (const id of result) if (!candidate.has(id)) result.delete(id)
    return { ids: result, names }
  }
}

export function matchesPersistedNode(node: SerializedNode, query: PersistedNodeQuery): boolean {
  if (query.nodeTypes && !query.nodeTypes.includes(node.type)) return false

  if (query.attributes) {
    for (const [key, expected] of Object.entries(query.attributes)) {
      const actual = fieldValue(node, key)
      if (actual !== expected) return false
    }
  }

  if (query.attributeRanges) {
    for (const [key, range] of Object.entries(query.attributeRanges)) {
      const value = fieldValue(node, key)
      if (typeof value !== 'number' || !Number.isFinite(value)) return false
      if (range.above !== undefined && value <= range.above) return false
      if (range.below !== undefined && value >= range.below) return false
    }
  }

  return true
}

export function applyPersistedNodeQuery(
  nodes: SerializedNode[],
  query: PersistedNodeQuery,
): SerializedNode[] {
  let results: SerializedNode[] = []
  for (const node of nodes) {
    assertQueryActive(query.signal)
    if (matchesPersistedNode(node, query)) results.push(node)
  }
  if (query.orderBy) {
    const { field, direction } = query.orderBy
    results = [...results].sort((a, b) => {
      const av = a.data[field]
      const bv = b.data[field]
      const an = typeof av === 'number' && Number.isFinite(av) ? av : 0
      const bn = typeof bv === 'number' && Number.isFinite(bv) ? bv : 0
      return direction === 'asc' ? an - bn : bn - an
    })
  }
  if (query.offset !== undefined) results = results.slice(query.offset)
  if (query.limit !== undefined) results = results.slice(0, query.limit)
  return results
}

export function applyPersistedCountPagination(count: number, query: PersistedNodeQuery): number {
  const afterOffset = query.offset === undefined
    ? count
    : Math.max(0, count - (query.offset < 0 ? Math.max(0, count + query.offset) : query.offset))
  return query.limit === undefined ? afterOffset : Math.max(0, Math.min(afterOffset, query.limit))
}
