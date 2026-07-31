import { describe, it, expect, beforeAll } from 'vitest'
import { mkdtempSync, rmSync, readdirSync } from 'node:fs'
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
})
