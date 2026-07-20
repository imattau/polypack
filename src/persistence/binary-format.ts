import { encode as msgpackEncode, decode as msgpackDecode } from '@msgpack/msgpack'
import type { SerializedNode, SerializedEdge } from '../types.js'

export type WalEntryKind =
  | 'putNode'
  | 'deleteNode'
  | 'putEdge'
  | 'deleteEdge'
  | 'putVector'
  | 'deleteVector'
  | 'clearAll'

export type WalEntry =
  | { kind: 'putNode'; node: SerializedNode }
  | { kind: 'deleteNode'; id: string }
  | { kind: 'putEdge'; edge: SerializedEdge }
  | { kind: 'deleteEdge'; id: string }
  | { kind: 'putVector'; id: string; vector: number[] }
  | { kind: 'deleteVector'; id: string }
  | { kind: 'clearAll' }

export interface SnapshotData {
  version: 1
  nodes: Array<[string, SerializedNode]>
  edges: Array<[string, SerializedEdge]>
  vectors: Array<[string, number[]]>
}

export function encodeWalEntries(entries: WalEntry[]): Uint8Array {
  const parts: Uint8Array[] = []
  let totalLen = 0
  for (const entry of entries) {
    const body = msgpackEncode(entry)
    const header = new Uint8Array(4)
    new DataView(header.buffer).setUint32(0, body.length, false)
    parts.push(header, body)
    totalLen += 4 + body.length
  }
  const result = new Uint8Array(totalLen)
  let offset = 0
  for (const part of parts) {
    result.set(part, offset)
    offset += part.length
  }
  return result
}

export function* decodeWalEntries(data: Uint8Array): Generator<WalEntry, void, unknown> {
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength)
  let offset = 0
  while (offset + 4 <= data.length) {
    const len = dv.getUint32(offset, false)
    offset += 4
    if (offset + len > data.length) break
    const entry = msgpackDecode(data.subarray(offset, offset + len)) as WalEntry
    offset += len
    yield entry
  }
}

export function encodeSnapshot(
  nodes: Map<string, SerializedNode>,
  edges: Map<string, SerializedEdge>,
  vectors: Map<string, number[]>,
): Uint8Array {
  const snapshot: SnapshotData = {
    version: 1,
    nodes: [],
    edges: [],
    vectors: [],
  }
  for (const [id, node] of nodes) snapshot.nodes.push([id, node])
  for (const [id, edge] of edges) snapshot.edges.push([id, edge])
  for (const [id, vector] of vectors) snapshot.vectors.push([id, vector])
  return msgpackEncode(snapshot)
}

export function decodeSnapshot(data: Uint8Array): {
  nodes: Map<string, SerializedNode>
  edges: Map<string, SerializedEdge>
  vectors: Map<string, number[]>
} {
  const snapshot = msgpackDecode(data) as SnapshotData
  const nodes = new Map<string, SerializedNode>()
  const edges = new Map<string, SerializedEdge>()
  const vectors = new Map<string, number[]>()
  if (snapshot.nodes) for (const [id, node] of snapshot.nodes) nodes.set(id, node)
  if (snapshot.edges) for (const [id, edge] of snapshot.edges) edges.set(id, edge)
  if (snapshot.vectors) for (const [id, vector] of snapshot.vectors) vectors.set(id, vector)
  return { nodes, edges, vectors }
}
