import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { PolyGraph } from '../src/graph'

const fixture = JSON.parse(readFileSync(join(process.cwd(), 'fixtures/database-core/parallel-edges.json'), 'utf8'))

describe('edge identity conformance fixture', () => {
  it('keeps parallel edges independent and conditionally mutable', () => {
    const graph = new PolyGraph()
    for (const node of fixture.nodes) graph.addNode(node)
    for (const edge of fixture.edges) graph.addEdge(edge)
    graph.updateEdge(fixture.update.id, fixture.update.data, { expectedRevision: fixture.update.expectedRevision })
    expect(graph.removeEdge(fixture.remove.id, { expectedRevision: fixture.remove.expectedRevision })).toBe(true)
    const edges = graph.getEdges('a', 'RELATED')
    expect(edges.map(edge => edge.id)).toEqual(fixture.expect.edgeIds)
    expect(edges[0].revision).toBe(fixture.expect.revision)
    expect(edges[0].data?.confidence).toBe(fixture.expect.confidence)
  })
})
