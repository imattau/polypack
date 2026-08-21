/** A write was based on a stale record revision. */
export class ConflictError extends Error {
  readonly name = 'ConflictError'
  readonly id: string
  readonly expectedRevision: number
  readonly actualRevision: number | undefined

  constructor(id: string, expectedRevision: number, actualRevision: number | undefined) {
    super(`Revision conflict for ${id}: expected ${expectedRevision}, actual ${actualRevision ?? 'missing'}`)
    this.id = id
    this.expectedRevision = expectedRevision
    this.actualRevision = actualRevision
  }
}
