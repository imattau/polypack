export class StoreLockError extends Error {
  readonly name = 'StoreLockError'
  readonly lockFile: string
  readonly metadata?: Record<string, unknown>

  constructor(lockFile: string, metadata?: Record<string, unknown>) {
    super(`Store is already locked by another writer: ${lockFile}`)
    this.lockFile = lockFile
    this.metadata = metadata
  }
}

export class ReadOnlyStoreError extends Error {
  readonly name = 'ReadOnlyStoreError'
  constructor() {
    super('The store was opened in read-only mode')
  }
}
