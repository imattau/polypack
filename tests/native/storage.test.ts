import { describe, it, expect, beforeAll } from 'vitest'
import { mkdtempSync, rmSync, readdirSync, writeFileSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { NativeStore, isNativeAvailable } from '../../packages/node-native/src/index'
import { BinaryStoreAdapter } from '../../src/persistence/binary-store'
import { NodeFileIO } from '../../src/persistence/node-file-io'

const available = isNativeAvailable()

beforeAll(() => {
  if (!available) console.warn('native binary unavailable — skipping native storage tests')
})

function node(id: string): Record<string, unknown> {
  return {
    id,
    type: 'doc',
    data: { title: `Node ${id}` },
    vector: null,
    insertedAt: 1,
    updatedAt: 1,
  }
}

function tempDir(prefix: string): string {
  return mkdtempSync(join(tmpdir(), `polypack-native-${prefix}-`))
}

describe('NativeStore', () => {
  it('persists and reloads from a directory', () => {
    if (!available) return
    const dir = tempDir('roundtrip')
    try {
      const store = new NativeStore(dir)
      expect(store.capabilities()).toMatchObject({
        atomicBatches: true,
        fsync: true,
        snapshots: true,
        concurrentWriters: false,
        vectorSearch: 'exact',
      })
      store.apply({
        putNodes: [
          { ...node('n1'), vector: [0.1, 0.2, 0.3] },
          node('n2'),
        ],
        putEdges: [{ id: 'n1::LINKS::n2', source: 'n1', target: 'n2', type: 'LINKS', data: null, createdAt: 1 }],
      })
      store.close()

      const reopened = new NativeStore(dir)
      expect(reopened.nodeIds().sort()).toEqual(['n1', 'n2'])
      expect(reopened.getNode('n1')).toMatchObject({ vector: [0.1, 0.2, 0.3] })
      expect(reopened.allEdges()).toHaveLength(1)
      reopened.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('rejects use after close', () => {
    if (!available) return
    const dir = tempDir('closed')
    try {
      const store = new NativeStore(dir)
      store.close()
      expect(() => store.nodeIds()).toThrow(/closed/)
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('rejects a second writer for the same directory', () => {
    if (!available) return
    const dir = tempDir('lock')
    try {
      const first = new NativeStore(dir)
      expect(() => new NativeStore(dir)).toThrow(/already locked/)
      first.close()
      const reopened = new NativeStore(dir)
      reopened.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('recovers an abandoned stale lock', () => {
    if (!available) return
    const dir = tempDir('stale-lock')
    try {
      writeFileSync(join(dir, 'store.lock'), JSON.stringify({
        pid: 999999,
        startedAt: Date.now() - 25 * 60 * 60 * 1000,
        token: 'abandoned',
      }))
      const store = new NativeStore(dir)
      expect(store.nodeIds()).toEqual([])
      store.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})

describe('cross-language byte compatibility', () => {
  it('reads files written by the Rust store with the TS BinaryStoreAdapter', async () => {
    if (!available) return
    const dir = tempDir('rust-to-ts')
    try {
      const native = new NativeStore(dir)
      native.apply({ putNodes: [{ ...node('n1'), vector: [0.5, 0.25] }] })
      native.close()

      const ts = new BinaryStoreAdapter({ storeDir: dir, fileIO: new NodeFileIO(dir) })
      expect((await ts.allNodeIds()).sort()).toEqual(['n1'])
      const got = await ts.getNode('n1')
      expect(got?.vector).toEqual([0.5, 0.25])
      await ts.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('reads files written by the TS BinaryStoreAdapter with the Rust store', async () => {
    if (!available) return
    const dir = tempDir('ts-to-rust')
    try {
      const ts = new BinaryStoreAdapter({ storeDir: dir, fileIO: new NodeFileIO(dir) })
      await ts.bulkPutNodes([
        { id: 'a', type: 'doc', data: { v: 1 }, vector: [1, 0], insertedAt: 1, updatedAt: 1 },
        { id: 'b', type: 'doc', data: { v: 2 }, vector: null, insertedAt: 2, updatedAt: 2 },
      ])
      await ts.close()

      const native = new NativeStore(dir)
      expect(native.nodeIds().sort()).toEqual(['a', 'b'])
      expect(native.getNode('a')).toMatchObject({ vector: [1, 0] })
      native.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('produces snapshot and wal files in the TS store directory', () => {
    if (!available) return
    const dir = tempDir('files')
    try {
      const native = new NativeStore(dir)
      native.apply({ putNodes: [node('x')] })
      native.close()
      const files = readdirSync(dir)
      expect(files).toContain('snapshot.msgpack')
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('checkpoints and verifies durable state', () => {
    if (!available) return
    const dir = tempDir('verify')
    try {
      const store = new NativeStore(dir)
      expect(store.stats()).toMatchObject({ persistedNodeCount: 0, edgeCount: 0, vectorCount: 0 })
      store.apply({
        putNodes: [node('n1'), node('n2')],
        putEdges: [{ id: 'n1::LINKS::n2', source: 'n1', target: 'n2', type: 'LINKS', data: null, createdAt: 1 }],
      })
      expect(store.stats()).toMatchObject({ persistedNodeCount: 2, edgeCount: 1, vectorCount: 0, mutationCount: 1 })
      expect(store.latestMutationSequence()).toBe(1n)
      expect(store.mutationLogSince(0n, 1)).toHaveLength(1)
      expect(store.mutationLogSince(1n)).toEqual([])
      expect(store.verify()).toMatchObject({ ok: true, nodeCount: 0, edgeCount: 0 })
      store.checkpoint()
      expect(store.verify()).toMatchObject({ ok: true, nodeCount: 2, edgeCount: 1 })
      store.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('creates a restorable directory backup', () => {
    if (!available) return
    const dir = tempDir('backup-source')
    const backup = tempDir('backup-destination')
    const restoredDir = tempDir('backup-restored')
    try {
      const store = new NativeStore(dir)
      store.apply({ putNodes: [node('backup-node')] })
      store.backup(backup)
      store.close()

      const restored = NativeStore.restore(backup, restoredDir)
      expect(restored.nodeIds()).toEqual(['backup-node'])
      expect(restored.verify()).toMatchObject({ ok: true, nodeCount: 1 })
      restored.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
      rmSync(backup, { recursive: true, force: true })
      rmSync(restoredDir, { recursive: true, force: true })
    }
  })
})
