import { useState, useEffect } from 'react'
import type { AnalyticsData } from '../types'

export function useAnalytics() {
  const [data,    setData]    = useState<AnalyticsData | null>(null)
  const [loading, setLoading] = useState(true)
  const [error,   setError]   = useState<string | null>(null)

  const load = async () => {
    setLoading(true)
    try {
      const [summary, top, zero, volume] = await Promise.all([
        fetch('/analytics/summary').then(r => r.json()),
        fetch('/analytics/top?n=10').then(r => r.json()),
        fetch('/analytics/zero?n=10').then(r => r.json()),
        fetch('/analytics/volume').then(r => r.json()),
      ])
      setData({
        summary,
        top:    top.top_queries,
        zero:   zero.zero_result_queries,
        volume: volume.volume,
      })
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Unknown error')
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => { load() }, [])

  return { data, loading, error, reload: load }
}
