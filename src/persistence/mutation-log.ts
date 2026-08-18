import type { MutationRecord, GraphOperation } from '../types.js'
import type { PersistenceChanges } from './adapter.js'

/** Build the durable logical representation of one adapter commit. */
export function mutationRecordFromChanges(changes: PersistenceChanges, sequence: bigint): MutationRecord | undefined {
  const operationId = changes.operationId ?? `mutation-${sequence}`
  const transactionId = changes.transactionId ?? operationId
  const operations: GraphOperation[] = []
  for (const node of changes.putNodes) operations.push({ type: 'putNode', payload: node as unknown as Record<string, unknown> })
  for (const id of changes.deleteNodeIds) operations.push({ type: 'deleteNode', payload: { id } })
  for (const edge of changes.putEdges) operations.push({ type: 'putEdge', payload: edge as unknown as Record<string, unknown> })
  for (const id of changes.deleteEdgeIds) operations.push({ type: 'deleteEdge', payload: { id } })
  for (const vector of changes.putVectors) operations.push({ type: 'putVector', payload: vector as unknown as Record<string, unknown> })
  for (const id of changes.deleteVectorIds) operations.push({ type: 'deleteVector', payload: { id } })
  if (changes.indexDefinitions) operations.push({ type: 'setIndexes', payload: { indexes: changes.indexDefinitions } })
  if (operations.length === 0) return undefined
  return { operationId, sequence, transactionId, timestamp: Date.now(), operations }
}
