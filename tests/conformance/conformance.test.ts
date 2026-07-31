import { describe, it, expect } from 'vitest'
import { loadFixtures, runFixture, ConformanceError } from './runner'

const fixtures = loadFixtures()

describe('conformance fixtures', () => {
  it('loads fixtures without TypeScript implementation details', () => {
    expect(fixtures.length).toBeGreaterThanOrEqual(10)
    for (const fixture of fixtures) {
      expect(fixture.schemaVersion).toBe(1)
      expect(fixture.name.length).toBeGreaterThan(0)
      expect(fixture.group.length).toBeGreaterThan(0)
    }
  })

  for (const fixture of fixtures) {
    it(`${fixture.group} / ${fixture.name}`, () => {
      try {
        runFixture(fixture)
      } catch (e) {
        if (e instanceof ConformanceError) {
          throw new Error(`conformance ${fixture.name}:\n${e.message}`)
        }
        throw e
      }
    })
  }
})
