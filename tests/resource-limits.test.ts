import { describe, expect, it } from 'vitest'
import { PolyGraph, ResourceLimitError } from '../src/index'

const node = (id: string, data: Record<string, unknown> = {}, vector?: number[]) => ({
  id,
  type: 'record',
  data,
  vector: vector ? new Float64Array(vector) : undefined,
  insertedAt: 1,
  updatedAt: 1,
})

describe('graph write resource limits', () => {
  it('rejects oversized payloads and vectors before mutation', () => {
    const graph = new PolyGraph()
    graph.setResourceLimits({ maxNodePayloadBytes: 20, maxVectorDimensions: 2 })
    expect(() => graph.addNode(node('large', { value: 'this is too large' }))).toThrow(ResourceLimitError)
    expect(() => graph.addNode(node('wide', {}, [1, 2, 3]))).toThrow(ResourceLimitError)
    expect(graph.getNode('large')).toBeUndefined()
    expect(graph.getNode('wide')).toBeUndefined()
  })

  it('rejects oversized batches and permits valid writes', () => {
    const graph = new PolyGraph()
    graph.setResourceLimits({ maxBatchSize: 1 })
    expect(() => graph.addNodes([node('a'), node('b')])).toThrow(ResourceLimitError)
    graph.addNodes([node('a')])
    expect(graph.getNode('a')).toBeDefined()
  })

  it('limits transaction mutations before exceeding the batch bound', async () => {
    const graph = new PolyGraph()
    graph.setResourceLimits({ maxBatchSize: 1 })
    await expect(graph.transaction(tx => {
      tx.addNode(node('a'))
      tx.addNode(node('b'))
    })).rejects.toMatchObject({ limitName: 'maxBatchSize' })
    expect(graph.getNode('a')).toBeUndefined()
  })

  it('applies limits to updates before changing the existing node', () => {
    const graph = new PolyGraph()
    graph.addNode(node('a', { value: 'ok' }))
    graph.setResourceLimits({ maxNodePayloadBytes: 20 })
    expect(() => graph.updateNode('a', { value: 'this is too large' })).toThrow(ResourceLimitError)
    expect(graph.getNode('a')?.data.value).toBe('ok')
  })
})
