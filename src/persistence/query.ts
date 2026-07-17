import type { SerializedNode } from '../types.js'
import type { PersistedNodeQuery } from './adapter.js'

export function matchesPersistedNode(node: SerializedNode, query: PersistedNodeQuery): boolean {
  if (query.nodeTypes && !query.nodeTypes.includes(node.type)) return false

  if (query.attributes) {
    for (const [key, expected] of Object.entries(query.attributes)) {
      const actual = key === 'type' ? node.type : node.data[key]
      if (actual !== expected) return false
    }
  }

  if (query.attributeRanges) {
    for (const [key, range] of Object.entries(query.attributeRanges)) {
      const value = node.data[key]
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
  let results = nodes.filter(node => matchesPersistedNode(node, query))
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
  return results
}
