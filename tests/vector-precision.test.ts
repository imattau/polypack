import { describe, expect, it } from 'vitest'
import { encode as msgpackEncode } from '@msgpack/msgpack'
import { decodeSnapshot, encodeSnapshot, decodeWalEntries, encodeWalEntries } from '../src/persistence/binary-format'

describe('vector precision persistence', () => {
  it('reads legacy v1 snapshots as Float64', () => {
    const bytes = msgpackEncode({
      version: 1,
      nodes: [['n', { id: 'n', type: 'doc', data: {}, vector: [1 / 3], insertedAt: 1, updatedAt: 1 }]],
      edges: [],
      vectors: [['n', [1 / 3]]],
    })
    const decoded = decodeSnapshot(bytes)
    expect(decoded.vectorPrecision).toBe('float64')
    expect(decoded.nodes.get('n')!.vector![0]).toBe(1 / 3)
  })

  it('round-trips v2 Float32 snapshot/node/WAL vectors in the public numeric shape', () => {
    const vector = Array(384).fill(1 / 3)
    const node = { id: 'n', type: 'doc', data: {}, vector, insertedAt: 1, updatedAt: 1 }
    const encoded = encodeSnapshot(new Map([['n', node]]), new Map(), new Map([['n', vector]]), [], undefined, 'float32')
    const decoded = decodeSnapshot(encoded)
    expect(decoded.vectorPrecision).toBe('float32')
    expect(decoded.nodes.get('n')!.vector).toEqual(vector.map(Math.fround))
    expect(decoded.vectors.get('n')).toEqual(vector.map(Math.fround))

    const wal = encodeWalEntries([
      { kind: 'putVector', id: 'n', vector },
      { kind: 'putNode', node },
    ], 'float32')
    const entries = [...decodeWalEntries(wal)]
    expect(entries[0]).toEqual({ kind: 'putVector', id: 'n', vector: vector.map(Math.fround) })
    expect(entries[1]).toMatchObject({ kind: 'putNode', node: { vector: vector.map(Math.fround) } })
    expect(encoded.byteLength).toBeLessThan(encodeSnapshot(new Map([['n', node]]), new Map(), new Map([['n', vector]])).byteLength)
  })
})
