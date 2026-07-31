import { describe, it, expect } from 'vitest'
import { readFileSync, readdirSync } from 'node:fs'
import { join, dirname } from 'node:path'
import { fileURLToPath } from 'node:url'
import { BinaryStoreAdapter } from '../../src/persistence/binary-store'
import { MemoryFileIO } from '../../src/persistence/file-io'
import { encodeWalEntries, encodeSnapshot } from '../../src/persistence/binary-format'
import type { WalEntry } from '../../src/persistence/binary-format'

const FIXTURES_DIR = join(dirname(fileURLToPath(import.meta.url)), '..', '..', 'fixtures', 'recovery')

interface RecoveryFixture {
  schemaVersion: number
  name: string
  group: string
  store: {
    snapshot?: { nodes: unknown[]; edges: unknown[]; vectors: Array<[string, number[]]> }
    wal?: WalEntry[]
    corruptTailHex?: string
  }
  expect: {
    presentNodeIds: string[]
    absentNodeIds?: string[]
    vectors?: Record<string, number[]>
    walRemovedAfterRecovery: boolean
    snapshotPresentAfterRecovery: boolean
  }
}

function loadRecoveryFixtures(): RecoveryFixture[] {
  return readdirSync(FIXTURES_DIR)
    .filter(f => f.endsWith('.json'))
    .sort()
    .map(f => JSON.parse(readFileSync(join(FIXTURES_DIR, f), 'utf8')) as RecoveryFixture)
}

async function runRecoveryFixture(fixture: RecoveryFixture): Promise<void> {
  const io = new MemoryFileIO()

  if (fixture.store.snapshot) {
    const nodes = new Map((fixture.store.snapshot.nodes as Array<{ id: string }>).map(n => [n.id, n]))
    const edges = new Map((fixture.store.snapshot.edges as Array<{ id: string }>).map(e => [e.id, e]))
    const vectors = new Map(fixture.store.snapshot.vectors ?? [])
    await io.writeFile('snapshot.msgpack', encodeSnapshot(nodes as never, edges as never, vectors as never))
  }
  if (fixture.store.wal) {
    let data = encodeWalEntries(fixture.store.wal)
    if (fixture.store.corruptTailHex) {
      const tail = Buffer.from(fixture.store.corruptTailHex, 'hex')
      const combined = new Uint8Array(data.length + tail.length)
      combined.set(data)
      combined.set(tail, data.length)
      data = combined
    }
    await io.writeFile('wal.msgpack', data)
  }

  const adapter = new BinaryStoreAdapter({ storeDir: 'recovery', fileIO: io, compactThreshold: 1_000_000 })
  const ids = await adapter.allNodeIds()
  expect(ids.sort()).toEqual([...fixture.expect.presentNodeIds].sort())
  if (fixture.expect.absentNodeIds) {
    for (const id of fixture.expect.absentNodeIds) expect(ids).not.toContain(id)
  }
  if (fixture.expect.vectors) {
    const all = await adapter.getAllVectors()
    for (const [id, vector] of Object.entries(fixture.expect.vectors)) {
      expect(all.find(v => v.id === id)?.vector).toEqual(vector)
    }
  }
  await adapter.close()

  expect(await io.fileExists('snapshot.msgpack')).toBe(fixture.expect.snapshotPresentAfterRecovery)
  const walNow = await io.readFile('wal.msgpack')
  const walEmpty = walNow === null || walNow.length === 0
  expect(walEmpty).toBe(fixture.expect.walRemovedAfterRecovery)
}

const fixtures = loadRecoveryFixtures()

describe('recovery fixtures', () => {
  it('loads recovery fixtures', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(4)
  })

  for (const fixture of fixtures) {
    it(`${fixture.group} / ${fixture.name}`, async () => {
      await runRecoveryFixture(fixture)
    })
  }
})
