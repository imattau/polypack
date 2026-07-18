import { act, cleanup, renderHook } from '@testing-library/react'
import { afterEach, beforeEach, describe, expect, it, vi } from 'vitest'
import { PolyGraph } from '../src/graph'
import { useGraphQuery } from '../src/react'

describe('useGraphQuery', () => {
  beforeEach(() => {
    vi.useFakeTimers()
  })

  afterEach(() => {
    cleanup()
    vi.useRealTimers()
    vi.restoreAllMocks()
  })

  it('uses updated node-type filters without recreating the subscription', () => {
    const graph = new PolyGraph()
    let runs = 0
    const { rerender } = renderHook(
      ({ nodeTypes }) => useGraphQuery(graph, () => ++runs, [], 10, nodeTypes),
      { initialProps: { nodeTypes: ['a'] } },
    )
    expect(runs).toBe(1)

    rerender({ nodeTypes: ['b'] })
    act(() => {
      graph.addNode({ id: 'a', type: 'a', data: {}, insertedAt: 1, updatedAt: 1 })
      vi.advanceTimersByTime(20)
    })
    expect(runs).toBe(1)

    act(() => {
      graph.addNode({ id: 'b', type: 'b', data: {}, insertedAt: 2, updatedAt: 2 })
      vi.advanceTimersByTime(10)
    })
    expect(runs).toBe(2)
  })

  it('uses the latest delay and coalesces rapid mutation bursts', () => {
    const graph = new PolyGraph()
    let runs = 0
    const { rerender } = renderHook(
      ({ delay }) => useGraphQuery(graph, () => ++runs, [], delay),
      { initialProps: { delay: 100 } },
    )
    rerender({ delay: 10 })

    act(() => {
      for (let i = 0; i < 5; i++) {
        graph.addNode({ id: `n${i}`, type: 't', data: {}, insertedAt: i, updatedAt: i })
      }
      vi.advanceTimersByTime(9)
    })
    expect(runs).toBe(1)
    act(() => vi.advanceTimersByTime(1))
    expect(runs).toBe(2)
  })

  it('discards stale async results after dependencies change', async () => {
    let resolveOld!: (value: string) => void
    let resolveNew!: (value: string) => void
    const oldResult = new Promise<string>(resolve => { resolveOld = resolve })
    const newResult = new Promise<string>(resolve => { resolveNew = resolve })
    const graph = new PolyGraph()
    const { result, rerender } = renderHook(
      ({ version }) => useGraphQuery(graph, () => version === 1 ? oldResult : newResult, [version], 0),
      { initialProps: { version: 1 } },
    )

    rerender({ version: 2 })
    await act(async () => {
      resolveOld('old')
      await Promise.resolve()
    })
    expect(result.current).toBeUndefined()

    await act(async () => {
      resolveNew('new')
      await Promise.resolve()
    })
    expect(result.current).toBe('new')
  })

  it('does not publish an async result after unmount', async () => {
    let resolve!: (value: string) => void
    const pending = new Promise<string>(done => { resolve = done })
    const graph = new PolyGraph()
    const { unmount } = renderHook(() => useGraphQuery(graph, () => pending, [], 0))
    unmount()

    await act(async () => {
      resolve('late')
      await Promise.resolve()
    })
  })

  it('reports query errors through the optional callback', async () => {
    vi.spyOn(console, 'error').mockImplementation(() => undefined)
    const error = new Error('query failed')
    const onError = vi.fn()
    const graph = new PolyGraph()
    renderHook(() => useGraphQuery(graph, () => Promise.reject(error), [], 0, undefined, onError))

    await act(async () => {
      await Promise.resolve()
      await Promise.resolve()
    })
    expect(onError).toHaveBeenCalledWith(error)
  })
})
