import type { SerializedNode, SerializedEdge } from '../types'
import type { PersistenceAdapter } from './adapter'

export interface IndexedDBConfig {
  name: string
  version: number
}

const DEFAULT_CONFIG: IndexedDBConfig = { name: 'polypack', version: 1 }

function openDB(config: IndexedDBConfig): Promise<IDBDatabase> {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(config.name, config.version)
    req.onupgradeneeded = () => {
      const db = req.result
      if (!db.objectStoreNames.contains('nodes')) {
        db.createObjectStore('nodes', { keyPath: 'id' })
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

export class IndexedDBAdapter implements PersistenceAdapter {
  private dbPromise: Promise<IDBDatabase> | null = null
  private config: IndexedDBConfig

  constructor(config?: Partial<IndexedDBConfig>) {
    this.config = { ...DEFAULT_CONFIG, ...config }
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
