import type { PolyNode, SerializedNode } from './types.js'
import type { PersistenceAdapter, PersistedNodeQuery } from './persistence/adapter.js'
import { applyPersistedNodeQuery } from './persistence/query.js'
import { cosineSimilarity } from './vector-index.js'

function restoreNode(node: SerializedNode): PolyNode {
  return {
    id: node.id,
    type: node.type,
    data: { ...node.data },
    vector: node.vector ? new Float64Array(node.vector) : undefined,
    insertedAt: node.insertedAt,
    updatedAt: node.updatedAt,
  }
}

/** Chainable asynchronous query over all persisted nodes. Results are detached. */
export class PersistedGraphQuery {
  private readonly adapter: PersistenceAdapter
  private query: PersistedNodeQuery = {}
  private resultOffset?: number
  private resultLimit?: number
  private similarVector?: { vector: number[]; threshold: number; topK?: number }

  constructor(adapter: PersistenceAdapter) {
    this.adapter = adapter
  }

  where(field: string, value: unknown): this {
    this.query.attributes = { ...this.query.attributes, [field]: value }
    return this
  }

  whereAttribute(name: string, value: unknown): this {
    return this.where(name, value)
  }

  whereAttributeRange(name: string, range: { above?: number; below?: number }): this {
    this.query.attributeRanges = { ...this.query.attributeRanges, [name]: range }
    return this
  }

  whereNodeType(...types: string[]): this {
    this.query.nodeTypes = types
    return this
  }

  orderBy(field: string, direction: 'asc' | 'desc' = 'asc'): this {
    this.query.orderBy = { field, direction }
    return this
  }

  offset(n: number): this {
    this.resultOffset = n
    return this
  }

  limit(n: number): this {
    this.resultLimit = n
    return this
  }

  similarTo(vector: number[], threshold = 0, topK?: number): this {
    this.similarVector = { vector, threshold, topK }
    return this
  }

  private async serialized(): Promise<SerializedNode[]> {
    if (this.adapter.queryNodes) return this.adapter.queryNodes(this.query)
    const ids = await this.adapter.allNodeIds()
    return applyPersistedNodeQuery(await this.adapter.getNodes(ids), this.query)
  }

  async toArray(): Promise<PolyNode[]> {
    let results = (await this.serialized()).map(restoreNode)

    if (this.similarVector) {
      const { vector, threshold, topK } = this.similarVector
      const scored = results
        .filter((node): node is PolyNode & { vector: Float64Array } => node.vector !== undefined)
        .map(node => ({ node, score: cosineSimilarity(vector, node.vector) }))
        .filter(result => result.score >= threshold)
        .sort((a, b) => b.score - a.score)
      results = (topK === undefined ? scored : scored.slice(0, topK)).map(result => result.node)
    }

    if (this.resultOffset !== undefined) results = results.slice(this.resultOffset)
    if (this.resultLimit !== undefined) results = results.slice(0, this.resultLimit)
    return results
  }

  async first(): Promise<PolyNode | null> {
    return (await this.toArray())[0] ?? null
  }

  async ids(): Promise<string[]> {
    return (await this.toArray()).map(node => node.id)
  }

  async count(): Promise<number> {
    if (this.similarVector || this.resultOffset !== undefined || this.resultLimit !== undefined) {
      return (await this.toArray()).length
    }
    if (this.adapter.countNodes) return this.adapter.countNodes(this.query)
    return (await this.serialized()).length
  }
}
