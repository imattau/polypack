// Bundles the browser-facing source with esbuild (platform: browser) and
// verifies the entry points never resolve `node:` built-ins. The node subpath
// is expected to fail, confirming it is genuinely node-only.
import { build } from 'esbuild'

const browserEntries = [
  'src/index.ts',
  'src/persistence/index.ts',
  'src/persistence/opfs.ts',
  'src/react.ts',
  'src/activation.ts',
  'src/sync/index.ts',
]

const nodeBuiltIn = /node:(fs|path|os|url|util|assert|child_process)/

for (const entry of browserEntries) {
  const result = await build({
    entryPoints: [entry],
    bundle: true,
    platform: 'browser',
    format: 'esm',
    write: false,
    outdir: '/tmp/polypack-browser-check',
    // The node IO is only reachable through a dynamic import that esbuild
    // dead-strips on the browser platform; leaving node built-ins external
    // keeps the bundle buildable while the scan below still rejects any
    // bundle that actually references them.
    external: ['node:*'],
  })
  for (const file of result.outputFiles) {
    if (nodeBuiltIn.test(file.text)) {
      console.error(`Browser bundle for ${entry} references node built-ins`)
      process.exit(1)
    }
  }
}

try {
  await build({
    entryPoints: ['src/persistence/node.ts'],
    bundle: true,
    platform: 'browser',
    format: 'esm',
    write: false,
    outdir: '/tmp/polypack-browser-check',
  })
  console.error('Expected the node persistence subpath to fail a browser bundle; it did not')
  process.exit(1)
} catch {
  // Expected: the node subpath statically imports node built-ins.
}

console.log(`browser-bundle: ${browserEntries.length} browser entry points clean, node subpath rejected`)
