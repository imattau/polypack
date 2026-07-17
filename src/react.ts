import { useEffect, useRef, useState, useCallback } from 'react'
import type { PolyGraph } from './graph.js'

/** Run a reactive graph query and optionally return a value before its first result. */
export function useLiveQuery<T>(
  graph: PolyGraph,
  querier: () => Promise<T> | T,
  deps?: unknown[],
  defaultResult?: T
): T | undefined {
  const result = useGraphQuery(graph, querier, deps ?? [])
  return result ?? defaultResult
}

/**
 * Run a sync or async query immediately and again after graph mutations.
 * Results and change events are debounced by `delay`; `nodeTypes` can suppress
 * reruns for unrelated node changes.
 */
export function useGraphQuery<T>(
  graph: PolyGraph,
  queryFn: () => T | Promise<T>,
  deps: unknown[],
  delay = 200,
  nodeTypes?: string[],
): T | undefined {
  const [result, setResult] = useState<T | undefined>(undefined)
  const timerRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const changeDebounceRef = useRef<ReturnType<typeof setTimeout> | undefined>(undefined)
  const delayRef = useRef(delay)
  const mountedRef = useRef(true)
  const isFirstRef = useRef(true)
  const nodeTypesRef = useRef(nodeTypes)
  const queryInFlightRef = useRef(false)
  const rerunPendingRef = useRef(false)
  const queryEpochRef = useRef({})
  const runQueryRef = useRef<() => void>(() => {})

  useEffect(() => {
    nodeTypesRef.current = nodeTypes
  }, [nodeTypes])

  const runQuery = useCallback(() => {
    const epoch = queryEpochRef.current

    if (queryInFlightRef.current) {
      rerunPendingRef.current = true
      return
    }

    queryInFlightRef.current = true

    const finishQuery = () => {
      queryInFlightRef.current = false
      if (rerunPendingRef.current && mountedRef.current) {
        rerunPendingRef.current = false
        runQueryRef.current()
      }
    }

    try {
      const value = queryFn()
      const setValue = (resolved: T) => {
        if (epoch !== queryEpochRef.current) return
        if (isFirstRef.current) {
          isFirstRef.current = false
          if (mountedRef.current) setResult(resolved)
        } else {
          clearTimeout(timerRef.current)
          timerRef.current = setTimeout(() => {
            if (mountedRef.current) setResult(resolved)
          }, delayRef.current)
        }
      }
      if (value instanceof Promise) {
        value
          .then(setValue)
          .catch((err) => {
            if (epoch !== queryEpochRef.current) return
            console.error('[Graph] Query error:', err)
            if (mountedRef.current) setResult(undefined)
          })
          .finally(finishQuery)
      } else {
        setValue(value)
        finishQuery()
      }
    } catch (err) {
      console.error('[Graph] Query error:', err)
      if (mountedRef.current) setResult(undefined)
      finishQuery()
    }
  // This hook intentionally accepts a caller-provided dependency list.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, deps)

  useEffect(() => {
    runQueryRef.current = runQuery
  }, [runQuery])

  useEffect(() => {
    delayRef.current = delay
  }, [delay])

  useEffect(() => {
    mountedRef.current = true
    isFirstRef.current = true

    runQuery()

    const subscription = graph.changes.subscribe((event) => {
      const types = nodeTypesRef.current
      if (types && event.nodeType && !types.includes(event.nodeType)) {
        return
      }
      clearTimeout(changeDebounceRef.current)
      changeDebounceRef.current = setTimeout(() => {
        if (mountedRef.current) runQueryRef.current()
      }, delayRef.current)
    })

    return () => {
      mountedRef.current = false
      queryEpochRef.current = {}
      rerunPendingRef.current = false
      clearTimeout(timerRef.current)
      clearTimeout(changeDebounceRef.current)
      subscription.unsubscribe()
    }
  }, [runQuery, graph])

  return result
}
