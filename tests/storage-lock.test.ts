import { describe, expect, it } from 'vitest'
import { mkdtempSync, rmSync } from 'node:fs'
import { join } from 'node:path'
import { tmpdir } from 'node:os'
import { ReadOnlyStoreError, StoreLockError } from '../src/index'
import { BinaryStoreAdapter } from '../src/persistence/binary-store'
import { NodeFileIO } from '../src/persistence/node-file-io'

describe('binary store process safety', () => {
  it('holds an exclusive filesystem lock and releases it on close', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'polypack-lock-'))
    try {
      const first = new BinaryStoreAdapter({ storeDir: dir })
      await first.putNode({ id: 'a', type: 'record', data: {}, vector: null, insertedAt: 1, updatedAt: 1 })
      const second = new BinaryStoreAdapter({ storeDir: dir })
      await expect(second.allNodeIds()).rejects.toBeInstanceOf(StoreLockError)
      await first.close()
      expect(await second.allNodeIds()).toEqual(['a'])
      await second.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('supports read-only access without acquiring the writer lock', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'polypack-readonly-'))
    try {
      const writer = new BinaryStoreAdapter({ storeDir: dir })
      await writer.putNode({ id: 'a', type: 'record', data: {}, vector: null, insertedAt: 1, updatedAt: 1 })
      const reader = new BinaryStoreAdapter({ storeDir: dir, readOnly: true })
      expect(await reader.allNodeIds()).toEqual(['a'])
      await expect(reader.putNode({ id: 'b', type: 'record', data: {}, vector: null, insertedAt: 1, updatedAt: 1 }))
        .rejects.toBeInstanceOf(ReadOnlyStoreError)
      await reader.close()
      await writer.close()
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })

  it('recovers a stale lock with explicit metadata', async () => {
    const dir = mkdtempSync(join(tmpdir(), 'polypack-stale-lock-'))
    try {
      const io = new NodeFileIO(dir)
      await io.writeFile('store.lock', new TextEncoder().encode(JSON.stringify({ pid: 1, startedAt: 1 })))
      const release = await io.acquireExclusiveLock!('store.lock', { storeDir: dir }, 1)
      await release()
      expect(await io.fileExists('store.lock')).toBe(false)
    } finally {
      rmSync(dir, { recursive: true, force: true })
    }
  })
})
