import * as fs from 'node:fs/promises'
import * as path from 'node:path'
import { BinaryStoreAdapter } from './binary-store.js'
import type { VectorPrecision } from './binary-format.js'

export interface VectorPrecisionMigrationOptions {
  /** Retained copy of the original store. Defaults to a timestamped sibling directory. */
  backupDir?: string
}

export interface VectorPrecisionMigrationResult {
  storeDir: string
  backupDir: string
  from: VectorPrecision
  to: VectorPrecision
}

/**
 * Offline-only Node filesystem migration. Keep all processes using the store stopped
 * for the duration; the source is checkpointed, rebuilt, verified, then directory-renamed.
 */
export async function migrateVectorPrecision(
  storeDir: string,
  precision: VectorPrecision,
  options: VectorPrecisionMigrationOptions = {},
): Promise<VectorPrecisionMigrationResult> {
  if (precision !== 'float32' && precision !== 'float64') throw new TypeError(`Unsupported vector precision: ${precision}`)
  const absoluteStore = path.resolve(storeDir)
  const parent = path.dirname(absoluteStore)
  const backupDir = path.resolve(options.backupDir ?? `${absoluteStore}.backup-${Date.now()}`)
  if (backupDir === absoluteStore || backupDir.startsWith(`${absoluteStore}${path.sep}`) || absoluteStore.startsWith(`${backupDir}${path.sep}`)) {
    throw new Error('Backup directory must be separate from the store directory')
  }
  const tempDir = path.join(parent, `.${path.basename(absoluteStore)}.migrate-${process.pid}-${Date.now()}`)
  await fs.access(absoluteStore)
  try { await fs.access(backupDir); throw new Error(`Backup directory already exists: ${backupDir}`) }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }
  try { await fs.access(tempDir); throw new Error(`Temporary directory already exists: ${tempDir}`) }
  catch (error) { if ((error as NodeJS.ErrnoException).code !== 'ENOENT') throw error }

  const source = new BinaryStoreAdapter({ storeDir: absoluteStore })
  let destination: BinaryStoreAdapter | undefined
  let sourcePrecision: VectorPrecision = 'float64'
  let sourceClosed = false
  let sourceMoved = false
  try {
    await source.checkpoint()
    sourcePrecision = await source.getVectorPrecision()
    const [nodeIds, nodes, edges, vectors, indexes, schemas, mutations] = await Promise.all([
      source.allNodeIds(),
      source.allNodeIds().then(ids => source.getNodes(ids)),
      source.getAllEdges(),
      source.getAllVectors(),
      source.getIndexDefinitions(),
      source.getSchemaDefinitions(),
      source.getMutationsSince(0n),
    ])
    if (nodes.length !== nodeIds.length) throw new Error('Source store changed while being read; stop all writers and retry')
    await source.close()
    sourceClosed = true

    destination = new BinaryStoreAdapter({ storeDir: tempDir, vectorPrecision: precision, compactThreshold: Number.MAX_SAFE_INTEGER })
    await destination.applyChanges({
      indexDefinitions: indexes,
      schemaDefinitions: schemas,
      putNodes: nodes,
      deleteNodeIds: [],
      putEdges: edges,
      deleteEdgeIds: [],
      putVectors: vectors,
      deleteVectorIds: [],
    })
    await destination.importMutationHistory(mutations)
    await destination.checkpoint()
    const report = await destination.verify()
    const latest = await destination.latestMutationSequence()
    if (!report.ok || report.nodeCount !== nodes.length || report.edgeCount !== edges.length || report.vectorCount !== vectors.length || latest !== (mutations.at(-1)?.sequence ?? 0n)) {
      throw new Error(`Migrated store verification failed: ${report.errors.join('; ') || 'record counts or mutation sequence differ'}`)
    }
    if (await destination.getVectorPrecision() !== precision) throw new Error('Migrated store precision verification failed')
    await destination.close()
    destination = undefined

    await fs.rename(absoluteStore, backupDir)
    sourceMoved = true
    try {
      await fs.rename(tempDir, absoluteStore)
    } catch (error) {
      await fs.rename(backupDir, absoluteStore)
      sourceMoved = false
      throw error
    }
    return { storeDir: absoluteStore, backupDir, from: sourcePrecision, to: precision }
  } catch (error) {
    if (destination) await destination.close().catch(() => undefined)
    if (!sourceClosed) await source.close().catch(() => undefined)
    if (sourceMoved) {
      try { await fs.rename(backupDir, absoluteStore) } catch { /* Preserve the backup for manual recovery. */ }
    }
    await fs.rm(tempDir, { recursive: true, force: true }).catch(() => undefined)
    throw error
  }
}
