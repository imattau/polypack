import { describe, expect, it } from 'vitest'
import { FeatureHashEmbedding, createEmbedding } from '../src/embedding'
import type { EmbeddingProvider } from '../src/embedding'
import { PolyGraph } from '../src/graph'
import { cosineSimilarity } from '../src/vector-index'

describe('embeddings', () => {
  it('creates deterministic normalized model-free embeddings', () => {
    const provider = new FeatureHashEmbedding()
    const first = provider.embed('Graph databases and vector search')
    const second = provider.embed('Graph databases and vector search')
    const related = provider.embed('Vector search for graph databases')
    const unrelated = provider.embed('Cooking pasta with tomato sauce')

    expect(first).toEqual(second)
    expect(first).toHaveLength(384)
    expect(Math.hypot(...first)).toBeCloseTo(1)
    expect(cosineSimilarity(first, related)).toBeGreaterThan(cosineSimilarity(first, unrelated))
  })

  it('supports configurable dimensions and empty text', () => {
    const provider = new FeatureHashEmbedding({ dimensions: 16 })
    expect(provider.embed('')).toEqual(new Float64Array(16))
    expect(() => new FeatureHashEmbedding({ dimensions: 0 })).toThrow(RangeError)
  })

  it('validates custom provider output and returns an owned Float64Array', async () => {
    const source = new Float32Array([1, 2, 3])
    const provider: EmbeddingProvider = { dimensions: 3, embed: async () => source }
    const vector = await createEmbedding(provider, 'hello')
    source[0] = 9

    expect(vector).toEqual(new Float64Array([1, 2, 3]))
    await expect(createEmbedding({ dimensions: 2, embed: () => [1] }, 'x')).rejects.toThrow(RangeError)
    await expect(createEmbedding({ embed: () => [Number.NaN] }, 'x')).rejects.toThrow(RangeError)
    await expect(createEmbedding({ embed: () => [] }, 'x')).rejects.toThrow(RangeError)
  })

  it('uses a supplied async provider for node mutation and text queries', async () => {
    const provider: EmbeddingProvider = {
      dimensions: 2,
      async embed(text) {
        return text.includes('graph') ? [1, 0] : [0, 1]
      },
    }
    const graph = new PolyGraph(undefined, undefined, provider)
    await graph.addNodeWithEmbedding({
      id: 'graph', type: 'doc', data: { title: 'Graphs' }, insertedAt: 1, updatedAt: 1,
    }, 'graph database')
    await graph.addNodeWithEmbedding({
      id: 'food', type: 'doc', data: { title: 'Food' }, insertedAt: 2, updatedAt: 2,
    }, 'pasta recipe')

    const results = (await graph.queryText('graph theory', 0, 1)).toArray()
    expect(results.map(node => node.id)).toEqual(['graph'])

    await graph.flush()
    const persisted = await (await graph.queryPersistedText('graph theory', 0, 1)).ids()
    expect(persisted).toEqual(['graph'])

    await graph.updateNodeWithEmbedding('food', { title: 'Graph food' }, 'graph recipe')
    expect(graph.getNode('food')?.vector).toEqual(new Float64Array([1, 0]))
  })
})
