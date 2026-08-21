import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { PolyGraph } from '../src/graph'

const fixture = JSON.parse(readFileSync(join(process.cwd(), 'fixtures/database-core/transaction.json'), 'utf8'))

describe('transaction conformance fixture', () => {
  it('provides read-your-own-writes and commits node/edge mutations atomically', async () => {
    const graph = new PolyGraph()
    graph.addNode(fixture.setup.nodes[0])
    const tx = fixture.transaction
    await graph.transaction(context => {
      context.patchNode(tx.patch.id, { increment: tx.patch.increment }, { expectedRevision: tx.patch.expectedRevision })
      context.addNode(tx.addNode)
      context.addEdge({
        id: 'person-1::RELATED_TO::person-2',
        source: tx.addEdge.source,
        type: tx.addEdge.type,
        target: tx.addEdge.target,
        createdAt: 1,
      })
      expect(context.getNode(tx.readYourWrites.id)?.data.count).toBe(tx.readYourWrites.count)
    })

    expect(graph.size).toBe(fixture.expect.nodeCount)
    expect(graph.getNode('person-1')?.data.count).toBe(fixture.expect.person1Count)
    expect(graph.getNode('person-1')?.revision).toBe(fixture.expect.person1Revision)
    expect(graph.getEdgeTargets('person-1', 'RELATED_TO')).toEqual(fixture.expect.edgeTargets)

    await expect(graph.transaction(context => {
      context.patchNode(fixture.rollback.patch.id, { set: fixture.rollback.patch.set }, { expectedRevision: fixture.rollback.patch.expectedRevision })
      context.addNode(fixture.rollback.addNode)
      throw new Error('rollback fixture failure')
    })).rejects.toThrow('rollback fixture failure')
    expect(graph.size).toBe(fixture.expect.rollbackCount)
    expect(graph.getNode('temporary')).toBeUndefined()
    expect(graph.getNode('person-1')?.data.count).toBe(fixture.expect.person1Count)
    expect(graph.getNode('person-1')?.revision).toBe(fixture.expect.rollbackRevision)
  })
})
