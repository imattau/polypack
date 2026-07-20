import * as fs from 'node:fs'
import * as fsp from 'node:fs/promises'
import * as path from 'node:path'

export interface FileIO {
  readFile(name: string): Promise<Uint8Array | null>
  writeFile(name: string, data: Uint8Array): Promise<void>
  appendFile(name: string, data: Uint8Array): Promise<void>
  deleteFile(name: string): Promise<void>
  fileExists(name: string): Promise<boolean>
}

export function isNode(): boolean {
  return typeof process !== 'undefined' && process.versions != null && (process.versions as Record<string, string>).node != null
}

export async function createFileIO(storeDir: string): Promise<FileIO> {
  if (isNode()) {
    return new NodeFileIO(storeDir)
  }
  if (typeof navigator !== 'undefined' && typeof (navigator as unknown as Record<string, unknown>).storage !== 'undefined') {
    return OPFSFileIO.create(storeDir)
  }
  throw new Error('No filesystem available (not Node.js and no OPFS support)')
}

export class NodeFileIO implements FileIO {
  private dir: string

  constructor(dir: string) {
    this.dir = dir
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
    await fsp.writeFile(tmpPath, data)
    await fsp.rename(tmpPath, filePath)
  }

  async appendFile(name: string, data: Uint8Array): Promise<void> {
    await fsp.mkdir(this.dir, { recursive: true })
    const filePath = this.resolve(name)
    const handle = await fsp.open(filePath, 'a')
    try {
      await handle.write(data)
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
}

export class OPFSFileIO implements FileIO {
  private dirHandle: FileSystemDirectoryHandle

  private constructor(dirHandle: FileSystemDirectoryHandle) {
    this.dirHandle = dirHandle
  }

  static async create(storeDir: string): Promise<OPFSFileIO> {
    const root = await navigator.storage.getDirectory()
    const dirHandle = await root.getDirectoryHandle(storeDir, { create: true })
    return new OPFSFileIO(dirHandle)
  }

  async readFile(name: string): Promise<Uint8Array | null> {
    try {
      const handle = await this.dirHandle.getFileHandle(name)
      const file = await handle.getFile()
      return new Uint8Array(await file.arrayBuffer())
    } catch {
      return null
    }
  }

  async writeFile(name: string, data: Uint8Array): Promise<void> {
    const handle = await this.dirHandle.getFileHandle(name, { create: true })
    const writable = await handle.createWritable({ keepExistingData: false })
    await writable.write(data)
    await writable.close()
  }

  async appendFile(name: string, data: Uint8Array): Promise<void> {
    const handle = await this.dirHandle.getFileHandle(name, { create: true })
    const file = await handle.getFile()
    const existing = new Uint8Array(await file.arrayBuffer())
    const combined = new Uint8Array(existing.length + data.length)
    combined.set(existing)
    combined.set(data, existing.length)
    const writable = await handle.createWritable({ keepExistingData: false })
    await writable.write(combined)
    await writable.close()
  }

  async deleteFile(name: string): Promise<void> {
    try {
      await this.dirHandle.removeEntry(name)
    } catch {
      // Ignore if not found
    }
  }

  async fileExists(name: string): Promise<boolean> {
    try {
      await this.dirHandle.getFileHandle(name)
      return true
    } catch {
      return false
    }
  }
}

export class MemoryFileIO implements FileIO {
  private files = new Map<string, Uint8Array>()

  async readFile(name: string): Promise<Uint8Array | null> {
    return this.files.get(name) ?? null
  }

  async writeFile(name: string, data: Uint8Array): Promise<void> {
    this.files.set(name, new Uint8Array(data))
  }

  async appendFile(name: string, data: Uint8Array): Promise<void> {
    const existing = this.files.get(name) ?? new Uint8Array(0)
    const combined = new Uint8Array(existing.length + data.length)
    combined.set(existing)
    combined.set(data, existing.length)
    this.files.set(name, combined)
  }

  async deleteFile(name: string): Promise<void> {
    this.files.delete(name)
  }

  async fileExists(name: string): Promise<boolean> {
    return this.files.has(name)
  }
}
