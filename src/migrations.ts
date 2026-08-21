import type { PolyNode, PolyEdge } from './types.js'
import type { PolyGraph } from './graph.js'

export interface MigrationDefinition {
  from: number
  to: number
  migrateNode(node: PolyNode): PolyNode | void
  migrateEdge?: (edge: PolyEdge) => PolyEdge | void
}

export interface MigrationProgress {
  from: number
  to: number
  processed: number
  total: number
  migrated: number
  lastProcessed?: { kind: 'node' | 'edge'; id: string }
}

export interface MigrationOptions {
  dryRun?: boolean
  batchSize?: number
  signal?: AbortSignal
  resumeAfter?: { nodeId?: string; edgeId?: string }
  onProgress?: (progress: MigrationProgress) => void | Promise<void>
}

export interface MigrationReport extends MigrationProgress {
  dryRun: boolean
}

export class MigrationError extends Error {
  readonly name = 'MigrationError'
  constructor(message: string) {
    super(message)
  }
}

/** Registry and runner for application-level record migrations. */
export class MigrationRegistry {
  private definitions = new Map<number, MigrationDefinition>()

  register(definition: MigrationDefinition): void {
    if (!Number.isInteger(definition.from) || !Number.isInteger(definition.to) || definition.to <= definition.from) {
      throw new RangeError('Migration versions must be integers with to greater than from')
    }
    if (this.definitions.has(definition.from)) throw new MigrationError(`Migration from ${definition.from} is already registered`)
    this.definitions.set(definition.from, definition)
  }

  get all(): MigrationDefinition[] {
    return [...this.definitions.values()].sort((a, b) => a.from - b.from)
  }

  private path(from: number, to: number): MigrationDefinition[] {
    if (from === to) return []
    const result: MigrationDefinition[] = []
    let version = from
    while (version < to) {
      const definition = this.definitions.get(version)
      if (!definition || definition.to > to) throw new MigrationError(`No contiguous migration path from ${version} to ${to}`)
      result.push(definition)
      version = definition.to
    }
    return result
  }

  async run(graph: PolyGraph, from: number, to: number, options: MigrationOptions = {}): Promise<MigrationReport> {
    if (!Number.isInteger(from) || !Number.isInteger(to) || to < from) throw new RangeError('Migration versions must be integers with to greater than or equal to from')
    const path = this.path(from, to)
    const batchSize = options.batchSize ?? Number.MAX_SAFE_INTEGER
    if (!Number.isInteger(batchSize) || batchSize < 1) throw new RangeError('Migration batchSize must be a positive integer')
    const checkAborted = (): void => {
      if (options.signal?.aborted) throw new MigrationError('Migration was aborted')
    }

    await graph.flush()
    const allNodes = (await graph.queryPersisted().toArray()).sort((a, b) => a.id.localeCompare(b.id))
    const allEdges = (await graph.persistence.getAllEdges()).sort((a, b) => a.id.localeCompare(b.id))
    let nodeStart = 0
    if (options.resumeAfter?.nodeId) {
      const index = allNodes.findIndex(node => node.id === options.resumeAfter!.nodeId)
      if (index === -1) throw new MigrationError(`resumeAfter.nodeId ${options.resumeAfter.nodeId} was not found; refusing to silently restart the migration from the beginning`)
      nodeStart = index + 1
    }
    let edgeStart = 0
    if (options.resumeAfter?.edgeId) {
      const index = allEdges.findIndex(edge => edge.id === options.resumeAfter!.edgeId)
      if (index === -1) throw new MigrationError(`resumeAfter.edgeId ${options.resumeAfter.edgeId} was not found; refusing to silently restart the migration from the beginning`)
      edgeStart = index + 1
    }
    const nodes = allNodes.slice(nodeStart)
    const edges = allEdges.slice(edgeStart)
    const report: MigrationReport = { from, to, processed: 0, total: nodes.length + edges.length, migrated: 0, dryRun: options.dryRun === true }
    const migrate = (node: PolyNode): PolyNode => {
      let current = node
      for (const definition of path) {
        const result = definition.migrateNode(current)
        if (result !== undefined) current = result
      }
      return current
    }
    const migrateEdge = (edge: PolyEdge): PolyEdge => {
      let current = edge
      for (const definition of path) {
        if (!definition.migrateEdge) continue
        const result = definition.migrateEdge(current)
        if (result !== undefined) current = result
      }
      return current
    }

    for (let offset = 0; offset < nodes.length; offset += batchSize) {
      checkAborted()
      const originals = nodes.slice(offset, offset + batchSize)
      const batch = originals.map(node => migrate(node))
      for (let i = 0; i < batch.length; i++) {
        if (batch[i].id !== originals[i].id || batch[i].type !== originals[i].type) {
          throw new MigrationError(`Node migration changed identity or type for ${originals[i].id}`)
        }
      }
      if (!options.dryRun && path.length > 0) {
        await graph.transaction(tx => {
          for (const node of batch) tx.addNode(node)
        })
      }
      report.processed += batch.length
      report.migrated += path.length > 0 ? batch.length : 0
      report.lastProcessed = { kind: 'node', id: batch[batch.length - 1].id }
      await options.onProgress?.({ ...report })
    }
    for (let offset = 0; offset < edges.length; offset += batchSize) {
      checkAborted()
      const originals = edges.slice(offset, offset + batchSize)
      const batch = originals.map(edge => migrateEdge({
        ...edge,
        data: edge.data ? structuredClone(edge.data) : undefined,
      }))
      for (let i = 0; i < batch.length; i++) {
        if (batch[i].id !== originals[i].id || batch[i].source !== originals[i].source || batch[i].target !== originals[i].target || batch[i].type !== originals[i].type) {
          throw new MigrationError(`Edge migration changed identity or endpoints for ${originals[i].id}`)
        }
      }
      const edgeMigrations = path.some(definition => definition.migrateEdge)
      const changed = batch.filter((edge, index) => JSON.stringify(edge.data) !== JSON.stringify(originals[index].data))
      if (!options.dryRun && edgeMigrations && changed.length > 0) {
        await graph.transaction(tx => {
          for (const edge of changed) tx.updateEdge(edge.id, edge.data ?? {})
        })
      }
      report.processed += batch.length
      report.migrated += edgeMigrations ? batch.length : 0
      report.lastProcessed = { kind: 'edge', id: batch[batch.length - 1].id }
      await options.onProgress?.({ ...report })
    }
    return report
  }
}
