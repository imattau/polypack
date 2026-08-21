import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { PolyGraph } from '../src/graph'

const fixture = JSON.parse(readFileSync(join(process.cwd(), 'fixtures/database-core/secondary-indexes.json'), 'utf8'))

describe('secondary-index conformance fixture', () => {
  it('intersects equality indexes and explains the selected stages', () => {
    const graph = new PolyGraph()
    for (const index of fixture.indexes) graph.defineIndex(index)
    for (const node of fixture.nodes) graph.addNode(node)
    const query = graph.query().where('surname', fixture.query.surname).where('birthYear', fixture.query.birthYear)
    const explanation = query.explain()
    expect(query.ids()).toEqual(fixture.expect.ids)
    expect(explanation.indexes).toEqual(fixture.expect.indexes)
    expect(explanation.stages).toContain(fixture.expect.intersectionStage)
  })
})
