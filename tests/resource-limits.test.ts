import { describe, expect, it } from 'vitest'
import { PolyGraph, ResourceLimitError } from '../src/index'
import fixture from '../fixtures/database-core/resource-limits.json'

const node = (id: string, data: Record<string, unknown> = {}, vector?: number[]) => ({
  id,
  type: 'record',
  data,
  vector: vector ? new Float64Array(vector) : undefined,
  insertedAt: 1,
  updatedAt: 1,
})

describe('graph write resource limits', () => {
  it('matches the shared resource-limit fixture', async () => {
    const graph = new PolyGraph()
    graph.setResourceLimits(fixture.limits)

    const rejected: string[] = []
    for (const candidate of [fixture.payloadNode, fixture.vectorNode]) {
      try {
        graph.addNode(node(candidate.id, candidate.data, candidate.vector))
      } catch (error) {
        expect(error).toBeInstanceOf(ResourceLimitError)
        rejected.push((error as ResourceLimitError).limitName)
      }
    }
    try {
      graph.addNodes(fixture.batchNodes.map(candidate => node(candidate.id, candidate.data)))
    } catch (error) {
      expect(error).toBeInstanceOf(ResourceLimitError)
      rejected.push((error as ResourceLimitError).limitName)
    }

    await expect(graph.transaction(tx => {
      tx.addNode(node(fixture.transactionNodes[0].id, fixture.transactionNodes[0].data))
      tx.addNode(node(fixture.transactionNodes[1].id, fixture.transactionNodes[1].data))
    })).rejects.toMatchObject({ limitName: 'maxBatchSize' })
    for (const id of fixture.expect.absentNodeIds) expect(graph.getNode(id)).toBeUndefined()

    await graph.transaction(tx => tx.addNode(node(fixture.afterRollbackNode.id, fixture.afterRollbackNode.data)))
    expect(() => graph.patchNode(fixture.afterRollbackNode.id, fixture.oversizedPatch)).toThrow(ResourceLimitError)
    expect(graph.getNode(fixture.afterRollbackNode.id)?.data).toEqual({})
    expect(rejected).toEqual(fixture.expect.rejectedLimitNames)
    for (const id of fixture.expect.presentNodeIds) expect(graph.getNode(id)).toBeDefined()
  })

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
