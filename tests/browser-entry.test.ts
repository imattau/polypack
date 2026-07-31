import { describe, it, expect } from 'vitest'
import { PolyGraph, VectorIndex, HNSWIndex, MemoryAdapter, GraphQuery } from '../src/index'
import { BinaryStoreAdapter, OPFSFileIO } from '../src/persistence/opfs'
import { MemoryFileIO } from '../src/persistence/file-io'
import type { PersistenceAdapter } from '../src/persistence/opfs'
import type { SerializedNode } from '../src/types'

describe('browser-safe entry points', () => {
  it('exports core and OPFS persistence from their public subpaths', () => {
    expect(typeof PolyGraph).toBe('function')
    expect(typeof VectorIndex).toBe('function')
    expect(typeof HNSWIndex).toBe('function')
    expect(typeof MemoryAdapter).toBe('function')
    expect(typeof GraphQuery).toBe('function')
    expect(typeof BinaryStoreAdapter).toBe('function')
    expect(typeof OPFSFileIO).toBe('function')
  })

  it('BinaryStoreAdapter works under a browser-like environment with an injected FileIO', async () => {
    const node: SerializedNode = { id: 'n1', type: 't', data: { v: 1 }, vector: null, insertedAt: 1, updatedAt: 1 }
    const adapter: PersistenceAdapter = new BinaryStoreAdapter({
      storeDir: 'browser-entry',
      fileIO: new MemoryFileIO(),
    })
    await adapter.putNode(node)
    expect(await adapter.getNode('n1')).toMatchObject({ id: 'n1' })
    await adapter.close()
  })
})
