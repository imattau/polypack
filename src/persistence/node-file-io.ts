import * as fs from 'node:fs'
import * as fsp from 'node:fs/promises'
import * as path from 'node:path'
import type { FileIO } from './file-io.js'

export class NodeFileIO implements FileIO {
  private dir: string
  private syncWrites: boolean

  constructor(dir: string, syncWrites = false) {
    this.dir = dir
    this.syncWrites = syncWrites
  }

  private resolve(name: string): string {
    return path.join(this.dir, name)
  }

  async readFile(name: string): Promise<Uint8Array | null> {
    try {
      return await fsp.readFile(this.resolve(name))
    } catch (err: unknown) {
      if ((err as { code?: string }).code === 'ENOENT') return null
      throw err
    }
  }

  async writeFile(name: string, data: Uint8Array): Promise<void> {
    await fsp.mkdir(this.dir, { recursive: true })
    const filePath = this.resolve(name)
    const tmpPath = filePath + '.tmp'
    const handle = await fsp.open(tmpPath, 'w')
    try {
      await handle.writeFile(data)
      if (this.syncWrites) await handle.sync()
    } finally {
      await handle.close()
    }
    await fsp.rename(tmpPath, filePath)
    if (this.syncWrites) await this.fsyncDir()
  }

  async appendFile(name: string, data: Uint8Array): Promise<void> {
    await fsp.mkdir(this.dir, { recursive: true })
    const filePath = this.resolve(name)
    const handle = await fsp.open(filePath, 'a')
    try {
      await handle.write(data)
      if (this.syncWrites) await handle.sync()
    } finally {
      await handle.close()
    }
  }

  async deleteFile(name: string): Promise<void> {
    try {
      await fsp.unlink(this.resolve(name))
    } catch (err: unknown) {
      if ((err as { code?: string }).code !== 'ENOENT') throw err
    }
  }

  async fileExists(name: string): Promise<boolean> {
    try {
      await fsp.access(this.resolve(name), fs.constants.F_OK)
      return true
    } catch {
      return false
    }
  }

  private async fsyncDir(): Promise<void> {
    try {
      const dirHandle = await fsp.open(this.dir, 'r')
      try {
        await dirHandle.sync()
      } finally {
        await dirHandle.close()
      }
    } catch {
      // Directory fsync is unsupported on some platforms — best effort only.
    }
  }
}

export function createNodeFileIO(storeDir: string, syncWrites = false): NodeFileIO {
  return new NodeFileIO(storeDir, syncWrites)
}
