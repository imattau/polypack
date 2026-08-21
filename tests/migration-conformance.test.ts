import { describe, expect, it } from 'vitest'
import { readFileSync } from 'node:fs'
import { join } from 'node:path'
import { PolyGraph } from '../src/graph'

const fixture = JSON.parse(readFileSync(join(process.cwd(), 'fixtures/database-core/migration.json'), 'utf8'))

describe('migration conformance fixture', () => {
  it('runs the shared migration in batches and preserves identity', async () => {
    const graph = new PolyGraph()
    for (const node of fixture.nodes) graph.addNode(node)
    graph.migrations.register({ from: fixture.from, to: fixture.to, migrateNode: node => ({ ...node, data: { ...node.data, displayName: node.data.name } }) })
    const report = await graph.migrations.run(graph, fixture.from, fixture.to, { batchSize: 1 })
    expect(report.migrated).toBe(fixture.expect.migrated)
    expect(graph.query().ids().sort()).toEqual(fixture.expect.ids)
    for (const id of fixture.expect.ids) expect(graph.getNode(id)?.data.displayName).toBe(fixture.expect.displayNames[id])
  })
})
