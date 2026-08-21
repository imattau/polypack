/** A unique secondary index would contain more than one record for a key. */
export class UniqueConstraintError extends Error {
  readonly name = 'UniqueConstraintError'
  readonly indexName: string
  readonly key: string
  readonly conflictingId: string

  constructor(indexName: string, key: string, conflictingId: string) {
    super(`Unique index ${indexName} is violated by key ${key} (record ${conflictingId})`)
    this.indexName = indexName
    this.key = key
    this.conflictingId = conflictingId
  }
}
