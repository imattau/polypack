import { defineConfig } from 'vitest/config'

export default defineConfig({
  test: {
    environment: 'jsdom',
    setupFiles: ['./tests/setup.ts'],
    testTimeout: 300_000,
    pool: 'forks',
    poolOptions: {
      forks: { singleFork: true },
    },
  },
})
