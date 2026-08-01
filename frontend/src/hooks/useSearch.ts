import { useState, useCallback } from 'react'
import type { SearchResult, SearchMeta, SearchParams } from '../types'

export function useSearch() {
  const [results, setResults] = useState<SearchResult[]>([])
  const [meta,    setMeta]    = useState<SearchMeta | null>(null)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState<string | null>(null)

  const search = useCallback(async ({
    q,
    page      = 1,
    sort      = 'score',
    category  = '',
    dateFrom  = '',
    dateTo    = '',
    fuzzy     = false,
    highlight = true,
  }: SearchParams) => {
    if (!q?.trim()) return

    setLoading(true)
    setError(null)

    const params = new URLSearchParams({
      q,
      page:      String(page),
      sort,
      highlight: String(highlight),
    })
    if (category) params.set('category', category)
    if (dateFrom) params.set('date_from', dateFrom)
    if (dateTo)   params.set('date_to', dateTo)
    if (fuzzy)    params.set('fuzzy', 'true')

    try {
      const res = await fetch(`/search?${params}`)
      if (!res.ok) {
        const err = await res.json() as { detail?: string }
        throw new Error(err.detail ?? `HTTP ${res.status}`)
      }
      const data = await res.json()
      setResults(data.results as SearchResult[])
      setMeta({
        query:       data.query,
        total:       data.total,
        page:        data.page,
        total_pages: data.total_pages,
        has_next:    data.has_next,
        has_prev:    data.has_prev,
        start:       data.start,
        end:         data.end,
        sort_by:     data.sort_by,
        latency_ms:  data.latency_ms,
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
      setResults([])
      setMeta(null)
    } finally {
      setLoading(false)
    }
  }, [])

  const clear = useCallback(() => {
    setResults([])
    setMeta(null)
    setError(null)
  }, [])

  return { results, meta, loading, error, search, clear }
}
