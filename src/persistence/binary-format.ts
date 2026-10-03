import { encode as msgpackEncode, decode as msgpackDecode } from '@msgpack/msgpack'
import type { SerializedNode, SerializedEdge, IndexDefinition, MutationRecord, PersistedSchemaDefinitions } from '../types.js'

export type VectorPrecision = 'float64' | 'float32'
const PACKED_F32 = '__polypack_f32'
type PackedVector = { [PACKED_F32]: Uint8Array }

export type WalEntryKind = 'putNode' | 'deleteNode' | 'putEdge' | 'deleteEdge' | 'putVector' | 'deleteVector' | 'clearAll' | 'setIndexes' | 'setSchema' | 'setPrecision'
export type WalEntry =
  | { kind: 'putNode'; node: SerializedNode }
  | { kind: 'deleteNode'; id: string }
  | { kind: 'putEdge'; edge: SerializedEdge }
  | { kind: 'deleteEdge'; id: string }
  | { kind: 'putVector'; id: string; vector: number[] }
  | { kind: 'deleteVector'; id: string }
  | { kind: 'clearAll' }
  | { kind: 'setIndexes'; indexes: IndexDefinition[] }
  | { kind: 'setSchema'; schema: PersistedSchemaDefinitions }
  | { kind: 'setPrecision'; precision: VectorPrecision }

export interface SnapshotData {
  version: 1 | 2
  vectorPrecision?: VectorPrecision
  nodes: Array<[string, SerializedNode]>
  edges: Array<[string, SerializedEdge]>
  vectors: Array<[string, number[]]>
  indexes?: IndexDefinition[]
  schemaDefinitions?: PersistedSchemaDefinitions
}

function pack(vector: number[], precision: VectorPrecision): number[] | PackedVector {
  if (precision === 'float64') return vector
  const bytes = new Uint8Array(vector.length * 4)
  const view = new DataView(bytes.buffer)
  vector.forEach((value, i) => view.setFloat32(i * 4, value, true))
  return { [PACKED_F32]: bytes }
}

function unpack(value: unknown): number[] {
  if (!value || typeof value !== 'object' || !(PACKED_F32 in value)) return value as number[]
  const bytes = (value as PackedVector)[PACKED_F32]
  if (bytes.byteLength % 4 !== 0) throw new Error('Invalid packed Float32 vector payload')
  const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength)
  return Array.from({ length: bytes.byteLength / 4 }, (_, i) => view.getFloat32(i * 4, true))
}

function encodeNode(node: SerializedNode, precision: VectorPrecision): SerializedNode {
  return node.vector ? { ...node, vector: pack(node.vector, precision) as unknown as number[] } : node
}
function decodeNode(node: SerializedNode): SerializedNode {
  return node.vector ? { ...node, vector: unpack(node.vector) } : node
}
function encodeWalEntry(entry: WalEntry, precision: VectorPrecision): unknown {
  if (entry.kind === 'putVector') return { ...entry, vector: pack(entry.vector, precision) }
  if (entry.kind === 'putNode') return { ...entry, node: encodeNode(entry.node, precision) }
  return entry
}
function decodeWalEntry(value: any): WalEntry {
  if (value.kind === 'putVector') return { ...value, vector: unpack(value.vector) }
  if (value.kind === 'putNode') return { ...value, node: decodeNode(value.node) }
  return value as WalEntry
}

export function encodeWalEntries(entries: WalEntry[], precision: VectorPrecision = 'float64'): Uint8Array {
  const parts: Uint8Array[] = []
  let totalLen = 0
  for (const entry of entries) {
    const body = msgpackEncode(encodeWalEntry(entry, precision))
    const header = new Uint8Array(4)
    new DataView(header.buffer).setUint32(0, body.length, false)
    parts.push(header, body)
    totalLen += 4 + body.length
  }
  const result = new Uint8Array(totalLen)
  let offset = 0
  for (const part of parts) { result.set(part, offset); offset += part.length }
  return result
}

export function* decodeWalEntries(data: Uint8Array): Generator<WalEntry, void, unknown> {
  const dv = new DataView(data.buffer, data.byteOffset, data.byteLength)
  let offset = 0
  while (offset + 4 <= data.length) {
    const len = dv.getUint32(offset, false); offset += 4
    if (offset + len > data.length) break
    yield decodeWalEntry(msgpackDecode(data.subarray(offset, offset + len)))
    offset += len
  }
}

export function encodeMutationRecords(records: MutationRecord[], precision: VectorPrecision = 'float64'): Uint8Array {
  const parts: Uint8Array[] = []
  let total = 0
  for (const record of records) {
    const compact = structuredClone(record)
    for (const operation of compact.operations) {
      if (operation.type === 'putVector' && Array.isArray(operation.payload.vector)) operation.payload.vector = pack(operation.payload.vector as number[], precision)
      if (operation.type === 'putNode' && Array.isArray(operation.payload.vector)) operation.payload.vector = pack(operation.payload.vector as number[], precision)
    }
    const body = msgpackEncode({ ...compact, sequence: record.sequence.toString() })
    const header = new Uint8Array(4)
    new DataView(header.buffer).setUint32(0, body.length, false)
    parts.push(header, body); total += header.length + body.length
  }
  const result = new Uint8Array(total)
  let offset = 0
  for (const part of parts) { result.set(part, offset); offset += part.length }
  return result
}

export function* decodeMutationRecords(data: Uint8Array): Generator<MutationRecord, void, unknown> {
  const view = new DataView(data.buffer, data.byteOffset, data.byteLength)
  let offset = 0
  while (offset + 4 <= data.length) {
    const length = view.getUint32(offset, false); offset += 4
    if (offset + length > data.length) break
    const decoded = msgpackDecode(data.subarray(offset, offset + length)) as Omit<MutationRecord, 'sequence'> & { sequence: string | number | bigint }
    offset += length
    for (const operation of decoded.operations) {
      if ((operation.type === 'putVector' || operation.type === 'putNode') && operation.payload.vector) operation.payload.vector = unpack(operation.payload.vector)
    }
    yield { ...decoded, sequence: BigInt(decoded.sequence) }
  }
}

export function encodeSnapshot(nodes: Map<string, SerializedNode>, edges: Map<string, SerializedEdge>, vectors: Map<string, number[]>, indexes: IndexDefinition[] = [], schemaDefinitions?: PersistedSchemaDefinitions, precision: VectorPrecision = 'float64'): Uint8Array {
  const snapshot: SnapshotData = {
    version: 2,
    vectorPrecision: precision,
    nodes: [...nodes].map(([id, node]) => [id, encodeNode(node, precision)]),
    edges: [...edges],
    vectors: [...vectors].map(([id, vector]) => [id, pack(vector, precision) as unknown as number[]]),
    indexes: indexes.map(index => ({ ...index, fields: [...index.fields] })),
    schemaDefinitions: schemaDefinitions ? structuredClone(schemaDefinitions) : undefined,
  }
  return msgpackEncode(snapshot)
}

export function decodeSnapshot(data: Uint8Array): { nodes: Map<string, SerializedNode>; edges: Map<string, SerializedEdge>; vectors: Map<string, number[]>; indexes: IndexDefinition[]; schemaDefinitions?: PersistedSchemaDefinitions; vectorPrecision: VectorPrecision } {
  const snapshot = msgpackDecode(data) as SnapshotData
  if (snapshot.version !== 1 && snapshot.version !== 2) throw new Error(`Unsupported snapshot version: ${snapshot.version}`)
  const precision = snapshot.version === 1 ? 'float64' : snapshot.vectorPrecision
  if (precision !== 'float32' && precision !== 'float64') throw new Error('Invalid snapshot vector precision')
  const nodes = new Map<string, SerializedNode>()
  const edges = new Map<string, SerializedEdge>()
  const vectors = new Map<string, number[]>()
  if (snapshot.nodes) for (const [id, node] of snapshot.nodes) nodes.set(id, decodeNode(node))
  if (snapshot.edges) for (const [id, edge] of snapshot.edges) edges.set(id, edge)
  if (snapshot.vectors) for (const [id, vector] of snapshot.vectors) vectors.set(id, unpack(vector))
  return { nodes, edges, vectors, indexes: snapshot.indexes ?? [], schemaDefinitions: snapshot.schemaDefinitions ? structuredClone(snapshot.schemaDefinitions) : undefined, vectorPrecision: precision }
}
