import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { PolyGraph } from '../src/graph'

const fixture = JSON.parse(readFileSync(join(process.cwd(), 'fixtures/database-core/snapshot-isolation.json'), 'utf8'))

describe('snapshot isolation conformance fixture', () => {
  it('keeps captured query state detached from later writes', async () => {
    const graph = new PolyGraph()
    for (const node of fixture.nodes) graph.addNode(node)
    const snapshot = await graph.snapshot()
    graph.addNode(fixture.mutation.add)
    graph.removeNode(fixture.mutation.remove)
    expect(snapshot.query().ids().sort()).toEqual(fixture.expect.snapshotIds)
    expect(graph.query().ids().sort()).toEqual(fixture.expect.liveIds)
  })
})
