// Verifies that root-reachable entry points never statically import `node:`
// built-ins, so native browser imports and bundlers stay clean. The Node IO is
// only reachable through a dynamic import guarded by an isNode() check.
import { readFile } from 'node:fs/promises'
import { dirname, resolve, relative } from 'node:path'

const roots = [
  'dist/index.js',
  'dist/persistence/index.js',
  'dist/persistence/opfs.js',
  'dist/react.js',
  'dist/sync/index.js',
]

const fromRe = /\bfrom\s*['"]([^'"]+)['"]/g

async function exists(p) {
  try {
    await readFile(p)
    return true
  } catch {
    return false
  }
}

async function resolveFile(target) {
  if (await exists(target)) return target
  const withJs = target.endsWith('.js') ? target : `${target}.js`
  if (await exists(withJs)) return withJs
  const index = resolve(target, 'index.js')
  if (await exists(index)) return index
  return null
}

const visited = new Set()
const queue = [...roots]
const offending = []

while (queue.length > 0) {
  const rel = queue.shift()
  const abs = resolve(rel)
  if (visited.has(abs)) continue
  visited.add(abs)
  const src = await readFile(abs, 'utf8')
  for (const match of src.matchAll(fromRe)) {
    const spec = match[1]
    if (spec.startsWith('node:')) {
      offending.push(`${relative(process.cwd(), abs)} -> ${spec}`)
      continue
    }
    if (!spec.startsWith('.') && !spec.startsWith('/')) continue
    const target = await resolveFile(resolve(dirname(abs), spec))
    if (target) queue.push(relative(process.cwd(), target))
  }
}

if (offending.length > 0) {
  console.error('Browser-unsafe static imports found in root-reachable modules:')
  for (const o of offending) console.error(`  ${o}`)
  process.exit(1)
}

console.log(`browser-safe: ${visited.size} modules checked from ${roots.length} entry points`)
