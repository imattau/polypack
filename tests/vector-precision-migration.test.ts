import { describe, expect, it } from 'vitest'
import { mkdtempSync, rmSync } from 'node:fs'
import { tmpdir } from 'node:os'
import { join } from 'node:path'
import { BinaryStoreAdapter } from '../src/persistence/binary-store'
import { migrateVectorPrecision } from '../src/persistence/migrate'

describe('offline vector precision migration', () => {
  it('converts precision, retains the original backup, and preserves mutation sequences', async () => {
    const root = mkdtempSync(join(tmpdir(), 'polypack-precision-migration-'))
    const storeDir = join(root, 'store')
    const backupDir = join(root, 'backup')
    try {
      const source = new BinaryStoreAdapter({ storeDir, vectorPrecision: 'float64' })
      await source.putNode({ id: 'n', type: 'doc', data: {}, vector: [1 / 3, 0.123456789], insertedAt: 1, updatedAt: 1 })
      await source.putVector('n', [1 / 3, 0.123456789])
      const beforeSequence = await source.latestMutationSequence()
      await source.close()

      const result = await migrateVectorPrecision(storeDir, 'float32', { backupDir })
      expect(result).toMatchObject({ from: 'float64', to: 'float32', storeDir, backupDir })

      const migrated = new BinaryStoreAdapter({ storeDir })
      expect(await migrated.getVectorPrecision()).toBe('float32')
      expect((await migrated.getNode('n'))!.vector).toEqual([Math.fround(1 / 3), Math.fround(0.123456789)])
      expect(await migrated.latestMutationSequence()).toBe(beforeSequence)
      expect((await migrated.verify()).ok).toBe(true)
      await migrated.close()

      const backup = new BinaryStoreAdapter({ storeDir: backupDir })
      expect(await backup.getVectorPrecision()).toBe('float64')
      expect((await backup.getNode('n'))!.vector).toEqual([1 / 3, 0.123456789])
      expect(await backup.latestMutationSequence()).toBe(beforeSequence)
      await backup.close()

      const widenedBackupDir = join(root, 'float32-backup')
      await migrateVectorPrecision(storeDir, 'float64', { backupDir: widenedBackupDir })
      const widened = new BinaryStoreAdapter({ storeDir })
      expect(await widened.getVectorPrecision()).toBe('float64')
      expect((await widened.getNode('n'))!.vector).toEqual([Math.fround(1 / 3), Math.fround(0.123456789)])
      await widened.close()
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })

  it('leaves the source untouched when a requested backup already exists', async () => {
    const root = mkdtempSync(join(tmpdir(), 'polypack-precision-migration-fail-'))
    const storeDir = join(root, 'store')
    const backupDir = join(root, 'backup')
    try {
      const source = new BinaryStoreAdapter({ storeDir, vectorPrecision: 'float64' })
      await source.putNode({ id: 'n', type: 'doc', data: {}, vector: null, insertedAt: 1, updatedAt: 1 })
      await source.close()
      const { mkdirSync } = await import('node:fs')
      mkdirSync(backupDir)
      await expect(migrateVectorPrecision(storeDir, 'float32', { backupDir })).rejects.toThrow('Backup directory already exists')
      const stillThere = new BinaryStoreAdapter({ storeDir })
      expect(await stillThere.getVectorPrecision()).toBe('float64')
      expect(await stillThere.getNode('n')).toBeDefined()
      await stillThere.close()
    } finally {
      rmSync(root, { recursive: true, force: true })
    }
  })
})
