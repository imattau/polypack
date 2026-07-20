import type { SerializedNode, SerializedEdge } from '../types.js'
import type { PersistenceAdapter, PersistenceChanges, PersistedNodeQuery } from './adapter.js'
import { applyPersistedCountPagination, applyPersistedNodeQuery, matchesPersistedNode } from './query.js'

/** IndexedDB database name and schema version. */
export interface IndexedDBConfig {
  name: string
  version: number
  /** Node data fields to index for ordered persisted queries. */
  nodeIndexes?: string[]
}

type ResolvedIndexedDBConfig = Required<IndexedDBConfig>

const DEFAULT_CONFIG: ResolvedIndexedDBConfig = { name: 'polypack', version: 2, nodeIndexes: [] }

function nodeDataIndexName(field: string): string {
  return `data:${field}`
}

function openDB(config: ResolvedIndexedDBConfig): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(config.name, config.version)
    req.onupgradeneeded = () => {
      const db = req.result
      const nodeStore = !db.objectStoreNames.contains('nodes')
        ? db.createObjectStore('nodes', { keyPath: 'id' })
        : req.transaction!.objectStore('nodes')
      if (!nodeStore.indexNames.contains('type')) {
        nodeStore.createIndex('type', 'type', { unique: false })
      }
      for (const field of new Set(config.nodeIndexes)) {
        const indexName = nodeDataIndexName(field)
        if (!nodeStore.indexNames.contains(indexName)) {
          nodeStore.createIndex(indexName, `data.${field}`, { unique: false })
        }
      }
      if (!db.objectStoreNames.contains('edges')) {
        const store = db.createObjectStore('edges', { keyPath: 'id' })
        store.createIndex('source', 'source', { unique: false })
        store.createIndex('target', 'target', { unique: false })
        store.createIndex('type', 'type', { unique: false })
      }
      if (!db.objectStoreNames.contains('vectors')) {
        db.createObjectStore('vectors', { keyPath: 'id' })
      }
    }
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

function idbRequest<T>(req: IDBRequest<T>): Promise<T> {
  return new Promise((resolve, reject) => {
    req.onsuccess = () => resolve(req.result)
    req.onerror = () => reject(req.error)
  })
}

function idbTransaction(tx: IDBTransaction): Promise<void> {
  return new Promise((resolve, reject) => {
    tx.oncomplete = () => resolve()
    tx.onerror = () => reject(tx.error)
    tx.onabort = () => reject(tx.error)
  })
}

function cursorAll<T>(store: IDBObjectStore): Promise<T[]> {
  return new Promise((resolve, reject) => {
    const results: T[] = []
    const req = store.openCursor()
    req.onsuccess = () => {
      const cursor = req.result
      if (cursor) {
        results.push(cursor.value)
        cursor.continue()
      } else {
        resolve(results)
      }
    }
    req.onerror = () => reject(req.error)
  })
}

function cursorQueryNodes(
  source: IDBObjectStore | IDBIndex,
  query: PersistedNodeQuery,
  range?: IDBKeyRange,
  direction: IDBCursorDirection = 'next',
): Promise<SerializedNode[]> {
  if (query.limit === 0) return Promise.resolve([])
  return new Promise((resolve, reject) => {
    const results: SerializedNode[] = []
    let matched = 0
    const offset = query.offset ?? 0
    const req = source.openCursor(range, direction)
    req.onsuccess = () => {
      const cursor = req.result
      if (!cursor) {
        resolve(results)
        return
      }
      const node = cursor.value as SerializedNode
      if (matchesPersistedNode(node, query)) {
        if (matched++ >= offset) results.push(node)
        if (query.limit !== undefined && results.length >= query.limit) {
          resolve(results)
          return
        }
      }
      cursor.continue()
    }
    req.onerror = () => reject(req.error)
  })
}

function queryIndexRange(query: PersistedNodeQuery, field: string): IDBKeyRange | undefined {
  const exact = query.attributes?.[field]
  if (exact !== undefined) {
    try {
      return IDBKeyRange.only(exact as IDBValidKey)
    } catch {
      return undefined
    }
  }
  const range = query.attributeRanges?.[field]
  if (!range) return undefined
  if (range.above !== undefined && range.below !== undefined) {
    return IDBKeyRange.bound(range.above, range.below, true, true)
  }
  if (range.above !== undefined) return IDBKeyRange.lowerBound(range.above, true)
  if (range.below !== undefined) return IDBKeyRange.upperBound(range.below, true)
  return undefined
}

/** Browser persistence adapter backed by three IndexedDB object stores. */
export class IndexedDBAdapter implements PersistenceAdapter {
  private dbPromise: Promise<IDBDatabase> | null = null
  private config: ResolvedIndexedDBConfig

  constructor(config?: Partial<IndexedDBConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config, nodeIndexes: config?.nodeIndexes ?? [] }
  }

  async applyChanges(changes: PersistenceChanges): Promise<void> {
    const database = await this.db()
    const tx = database.transaction(['nodes', 'edges', 'vectors'], 'readwrite')
    const nodes = tx.objectStore('nodes')
    const edges = tx.objectStore('edges')
    const vectors = tx.objectStore('vectors')
    try {
      for (const id of changes.deleteNodeIds) nodes.delete(id)
      for (const id of changes.deleteEdgeIds) edges.delete(id)
      for (const id of changes.deleteVectorIds) vectors.delete(id)
      for (const node of changes.putNodes) nodes.put(node)
      for (const edge of changes.putEdges) edges.put(edge)
      for (const entry of changes.putVectors) vectors.put(entry)
    } catch (error) {
      tx.abort()
      throw error
    }
    await idbTransaction(tx)
  }

  private async db(): Promise<IDBDatabase> {
    if (!this.dbPromise) {
      this.dbPromise = openDB(this.config)
    }
    return this.dbPromise
  }

  async putNode(node: SerializedNode): Promise<void> {
    const database = await this.db()
    const tx = database.transaction('nodes', 'readwrite')
    tx.objectStore('nodes').put(node)
    await idbTransaction(tx)
  }

  async bulkPutNodes(nodes: SerializedNode[]): Promise<void> {
    if (nodes.length === 0) return
    const database = await this.db()
    const tx = database.transaction('nodes', 'readwrite')
    const store = tx.objectStore('nodes')
    for (const node of nodes) store.put(node)
    await idbTransaction(tx)
  }

  async getNode(id: string): Promise<SerializedNode | undefined> {
    const database = await this.db()
    return idbRequest(database.transaction('nodes').objectStore('nodes').get(id))
  }

  async getNodes(ids: string[]): Promise<SerializedNode[]> {
    if (ids.length === 0) return []
    const database = await this.db()
    const tx = database.transaction('nodes')
    const store = tx.objectStore('nodes')
    const results = await Promise.all(ids.map(id => idbRequest<SerializedNode | undefined>(store.get(id))))
    return results.filter((node): node is SerializedNode => node !== undefined)
  }

  async deleteNode(id: string): Promise<void> {
    const database = await this.db()
    const tx = database.transaction('nodes', 'readwrite')
    tx.objectStore('nodes').delete(id)
    await idbTransaction(tx)
  }

  async bulkDeleteNodes(ids: string[]): Promise<void> {
    if (ids.length === 0) return
    const database = await this.db()
    const tx = database.transaction('nodes', 'readwrite')
    const store = tx.objectStore('nodes')
    for (const id of ids) store.delete(id)
    await idbTransaction(tx)
  }

  async allNodeIds(): Promise<string[]> {
    const database = await this.db()
    const tx = database.transaction('nodes')
    const ids: string[] = []
    return new Promise((resolve, reject) => {
      const req = tx.objectStore('nodes').openCursor()
      req.onsuccess = () => {
        const cursor = req.result
        if (cursor) {
          ids.push(cursor.primaryKey as string)
          cursor.continue()
        } else {
          resolve(ids)
        }
      }
      req.onerror = () => reject(req.error)
    })
  }

  async queryNodes(query: PersistedNodeQuery): Promise<SerializedNode[]> {
    const database = await this.db()
    const tx = database.transaction('nodes')
    const store = tx.objectStore('nodes')
    const cursorPage = query.orderBy === undefined &&
      Number.isInteger(query.offset ?? 0) && (query.offset ?? 0) >= 0 &&
      (query.limit === undefined || (Number.isInteger(query.limit) && query.limit >= 0))

    const orderedFieldConstrained = query.orderBy && (
      Object.prototype.hasOwnProperty.call(query.attributes ?? {}, query.orderBy.field) ||
      query.attributeRanges?.[query.orderBy.field] !== undefined
    )
    const indexedOrder = query.orderBy && orderedFieldConstrained &&
      store.indexNames.contains(nodeDataIndexName(query.orderBy.field)) &&
      Number.isInteger(query.offset ?? 0) && (query.offset ?? 0) >= 0 &&
      (query.limit === undefined || (Number.isInteger(query.limit) && query.limit >= 0))
    if (indexedOrder) {
      const { field, direction } = query.orderBy!
      return cursorQueryNodes(
        store.index(nodeDataIndexName(field)),
        query,
        queryIndexRange(query, field),
        direction === 'asc' ? 'next' : 'prev',
      )
    }

    if (cursorPage && (!query.nodeTypes || query.nodeTypes.length === 1)) {
      if (query.nodeTypes?.length === 1 && store.indexNames.contains('type')) {
        return cursorQueryNodes(store.index('type'), query, IDBKeyRange.only(query.nodeTypes[0]))
      }
      if (!query.nodeTypes) return cursorQueryNodes(store, query)
    }

    let candidates: SerializedNode[]

    if (query.nodeTypes && query.nodeTypes.length === 0) {
      candidates = []
    } else if (query.nodeTypes && store.indexNames.contains('type')) {
      const index = store.index('type')
      const groups = await Promise.all(
        query.nodeTypes.map(type => idbRequest<SerializedNode[]>(index.getAll(type))),
      )
      candidates = groups.flat()
    } else {
      candidates = await cursorAll<SerializedNode>(store)
    }

    return applyPersistedNodeQuery(candidates, query)
  }

  async countNodes(query: PersistedNodeQuery): Promise<number> {
    const onlySingleType = query.nodeTypes?.length === 1 &&
      !query.attributes && !query.attributeRanges
    if (onlySingleType) {
      const database = await this.db()
      const store = database.transaction('nodes').objectStore('nodes')
      if (store.indexNames.contains('type')) {
        const count = await idbRequest(store.index('type').count(query.nodeTypes![0]))
        return applyPersistedCountPagination(count, query)
      }
    }
    return (await this.queryNodes(query)).length
  }

  async putEdge(edge: SerializedEdge): Promise<void> {
    const database = await this.db()
    const tx = database.transaction('edges', 'readwrite')
    tx.objectStore('edges').put(edge)
    await idbTransaction(tx)
  }

  async bulkPutEdges(edges: SerializedEdge[]): Promise<void> {
    if (edges.length === 0) return
    const database = await this.db()
    const tx = database.transaction('edges', 'readwrite')
    const store = tx.objectStore('edges')
    for (const edge of edges) store.put(edge)
    await idbTransaction(tx)
  }

  async getAllEdges(): Promise<SerializedEdge[]> {
    const database = await this.db()
    return cursorAll<SerializedEdge>(database.transaction('edges').objectStore('edges'))
  }

  private async getEdgesByIndex(
    indexName: 'source' | 'target',
    values: string[],
    type?: string,
  ): Promise<SerializedEdge[]> {
    if (values.length === 0) return []
    const database = await this.db()
    const store = database.transaction('edges').objectStore('edges')
    const index = store.index(indexName)
    const groups = await Promise.all(
      [...new Set(values)].map(value => idbRequest<SerializedEdge[]>(index.getAll(value))),
    )
    const edges = groups.flat()
    return type === undefined ? edges : edges.filter(edge => edge.type === type)
  }

  async getEdgesBySources(sources: string[], type?: string): Promise<SerializedEdge[]> {
    return this.getEdgesByIndex('source', sources, type)
  }

  async getEdgesByTargets(targets: string[], type?: string): Promise<SerializedEdge[]> {
    return this.getEdgesByIndex('target', targets, type)
  }

  async deleteEdge(id: string): Promise<void> {
    const database = await this.db()
    const tx = database.transaction('edges', 'readwrite')
    tx.objectStore('edges').delete(id)
    await idbTransaction(tx)
  }

  async bulkDeleteEdges(ids: string[]): Promise<void> {
    if (ids.length === 0) return
    const database = await this.db()
    const tx = database.transaction('edges', 'readwrite')
    const store = tx.objectStore('edges')
    for (const id of ids) store.delete(id)
    await idbTransaction(tx)
  }

  async putVector(id: string, vector: number[]): Promise<void> {
    const database = await this.db()
    const tx = database.transaction('vectors', 'readwrite')
    tx.objectStore('vectors').put({ id, vector })
    await idbTransaction(tx)
  }

  async bulkPutVectors(entries: Array<{ id: string; vector: number[] }>): Promise<void> {
    if (entries.length === 0) return
    const database = await this.db()
    const tx = database.transaction('vectors', 'readwrite')
    const store = tx.objectStore('vectors')
    for (const entry of entries) store.put(entry)
    await idbTransaction(tx)
  }

  async deleteVector(id: string): Promise<void> {
    const database = await this.db()
    const tx = database.transaction('vectors', 'readwrite')
    tx.objectStore('vectors').delete(id)
    await idbTransaction(tx)
  }

  async getVectors(ids: string[]): Promise<Array<{ id: string; vector: number[] }>> {
    if (ids.length === 0) return []
    const database = await this.db()
    const store = database.transaction('vectors').objectStore('vectors')
    const results = await Promise.all(
      ids.map(id => idbRequest<{ id: string; vector: number[] } | undefined>(store.get(id))),
    )
    return results.filter((v): v is { id: string; vector: number[] } => v !== undefined)
  }

  async getAllVectors(): Promise<Array<{ id: string; vector: number[] }>> {
    const database = await this.db()
    return cursorAll<{ id: string; vector: number[] }>(database.transaction('vectors').objectStore('vectors'))
  }

  async clearAll(): Promise<void> {
    const database = await this.db()
    const stores = ['nodes', 'edges', 'vectors']
    const tx = database.transaction(stores, 'readwrite')
    for (const store of stores) tx.objectStore(store).clear()
    await idbTransaction(tx)
  }

  async close(): Promise<void> {
    if (!this.dbPromise) return
    try {
      const database = await this.dbPromise
      database.close()
    } catch {
      // Ignore close failures
    }
    this.dbPromise = null
  }
}
