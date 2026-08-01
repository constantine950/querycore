import { useState, useCallback } from 'react'
import { SearchBar }   from '../components/SearchBar.tsx'
import { ResultCard }  from '../components/ResultCard.tsx'
import { FilterPanel } from '../components/FilterPanel.tsx'
import { Pagination }  from '../components/Pagination.tsx'
import { useSearch }   from '../hooks/useSearch'
import type { Filters, SearchParams } from '../types'

const DEFAULT_FILTERS: Filters = {
  category: '', sort: 'score', dateFrom: '', dateTo: '', fuzzy: false,
}

export function SearchPage() {
  const { results, meta, loading, error, search } = useSearch()
  const [filters,      setFilters]      = useState<Filters>(DEFAULT_FILTERS)
  const [currentQuery, setCurrentQuery] = useState('')

  const run = useCallback((params: SearchParams) => {
    const q = params.q ?? currentQuery
    setCurrentQuery(q)
    search({
      q,
      sort:     filters.sort,
      category: filters.category,
      dateFrom: filters.dateFrom,
      dateTo:   filters.dateTo,
      fuzzy:    filters.fuzzy,
      page:     params.page ?? 1,
    })
  }, [filters, currentQuery, search])

  const handleFilterChange = (newFilters: Filters) => {
    setFilters(newFilters)
    if (currentQuery) {
      search({
        q:        currentQuery,
        sort:     newFilters.sort,
        category: newFilters.category,
        dateFrom: newFilters.dateFrom,
        dateTo:   newFilters.dateTo,
        fuzzy:    newFilters.fuzzy,
        page:     1,
      })
    }
  }

  const isEmpty    = !loading && !error && results.length === 0 && meta !== null
  const hasResults = results.length > 0

  return (
    <div style={{ maxWidth: '780px', margin: '0 auto', padding: '40px 20px' }}>

      {/* Header */}
      <div style={{ marginBottom: '40px' }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '12px', marginBottom: '6px' }}>
          <h1 style={{ fontFamily: 'var(--mono)', fontSize: '1.5rem', fontWeight: 500, color: 'var(--text)', letterSpacing: '-0.02em' }}>
            QueryCore
          </h1>
          <span style={{
            background: 'var(--accent-dim)', color: '#a5b4fc',
            borderRadius: '4px', fontSize: '0.65rem', fontWeight: 700,
            padding: '2px 7px', textTransform: 'uppercase', letterSpacing: '0.08em',
          }}>
            search engine
          </span>
        </div>
        <p style={{ color: 'var(--text-2)', fontSize: '0.875rem' }}>
          TF-IDF ranking · fuzzy search · phrase matching · autocomplete
        </p>
      </div>

      <div style={{ marginBottom: '16px' }}>
        <SearchBar onSearch={run} loading={loading} />
      </div>

      <div style={{ marginBottom: '28px' }}>
        <FilterPanel filters={filters} onChange={handleFilterChange} />
      </div>

      {/* Meta bar */}
      {meta && (
        <div style={{
          display: 'flex', justifyContent: 'space-between', alignItems: 'center',
          marginBottom: '16px', color: 'var(--text-2)', fontSize: '0.8rem', fontFamily: 'var(--mono)',
        }}>
          <span>
            {meta.total > 0
              ? `${meta.start}–${meta.end} of ${meta.total} results`
              : '0 results'}
            {' '}for <strong style={{ color: 'var(--text)' }}>"{meta.query}"</strong>
          </span>
          <span>{meta.latency_ms.toFixed(1)}ms</span>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{
          background: 'rgba(248,113,113,0.1)', border: '1px solid rgba(248,113,113,0.3)',
          borderRadius: 'var(--radius)', color: 'var(--red)',
          fontSize: '0.875rem', padding: '12px 16px', marginBottom: '16px',
        }}>
          {error}
        </div>
      )}

      {/* Loading */}
      {loading && (
        <div style={{ color: 'var(--text-2)', fontSize: '0.875rem', textAlign: 'center', padding: '40px 0' }}>
          Searching…
        </div>
      )}

      {/* Empty */}
      {isEmpty && (
        <div style={{ color: 'var(--text-2)', fontSize: '0.875rem', textAlign: 'center', padding: '40px 0' }}>
          No results for <strong>"{meta!.query}"</strong>.
          {!filters.fuzzy && (
            <span> Try enabling{' '}
              <button
                onClick={() => handleFilterChange({ ...filters, fuzzy: true })}
                style={{ background: 'none', border: 'none', color: 'var(--accent)', cursor: 'pointer', fontSize: '0.875rem' }}
              >fuzzy search</button>.
            </span>
          )}
        </div>
      )}

      {/* Results */}
      {!loading && hasResults && (
        <div style={{ display: 'flex', flexDirection: 'column', gap: '10px' }}>
          {results.map((r, i) => (
            <ResultCard key={r.doc_id} result={r} rank={(meta?.start ?? 1) + i} />
          ))}
        </div>
      )}

      {/* Pagination */}
      {!loading && meta && meta.total_pages > 1 && (
        <Pagination meta={meta} onPage={(p) => run({ q: currentQuery, page: p })} />
      )}

      {/* Initial empty state */}
      {!meta && !loading && !error && (
        <div style={{
          color: 'var(--text-2)', fontSize: '0.875rem',
          textAlign: 'center', padding: '60px 0', lineHeight: 2,
        }}>
          <div style={{ fontSize: '2rem', marginBottom: '12px' }}>⌕</div>
          Try{' '}<code style={{ color: 'var(--accent)', fontFamily: 'var(--mono)' }}>search engine</code>{' '}
          or{' '}<code style={{ color: 'var(--accent)', fontFamily: 'var(--mono)' }}>"inverted index"</code>{' '}
          or{' '}<code style={{ color: 'var(--accent)', fontFamily: 'var(--mono)' }}>quantum OR algorithm</code>
        </div>
      )}
    </div>
  )
}
