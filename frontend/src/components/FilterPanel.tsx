import type { CSSProperties } from 'react'
import type { Filters } from '../types'

interface Props {
  filters:  Filters
  onChange: (f: Filters) => void
}

const sel: CSSProperties = {
  background: 'var(--surface)', border: '1px solid var(--border)',
  borderRadius: 'var(--radius)', color: 'var(--text)',
  fontSize: '0.8rem', padding: '6px 10px', cursor: 'pointer', outline: 'none',
}
const inp: CSSProperties = { ...sel, width: '130px' }

export function FilterPanel({ filters, onChange }: Props) {
  const set = (key: keyof Filters, val: string | boolean) =>
    onChange({ ...filters, [key]: val })

  return (
    <div style={{ display: 'flex', gap: '10px', flexWrap: 'wrap', alignItems: 'center' }}>
      <span style={{ color: 'var(--text-2)', fontSize: '0.8rem', whiteSpace: 'nowrap' }}>Filter:</span>

      <select style={sel} value={filters.category} onChange={e => set('category', e.target.value)}>
        <option value="">All categories</option>
        <option value="computer_science">Computer Science</option>
        <option value="science">Science</option>
        <option value="history">History</option>
      </select>

      <select style={sel} value={filters.sort} onChange={e => set('sort', e.target.value)}>
        <option value="score">Sort: Relevance</option>
        <option value="date">Sort: Date</option>
        <option value="title">Sort: Title</option>
      </select>

      <input style={inp} type="date" value={filters.dateFrom}
        onChange={e => set('dateFrom', e.target.value)} title="From date" />
      <span style={{ color: 'var(--muted)', fontSize: '0.75rem' }}>→</span>
      <input style={inp} type="date" value={filters.dateTo}
        onChange={e => set('dateTo', e.target.value)} title="To date" />

      <button
        style={{
          background: filters.fuzzy ? 'var(--accent-dim)' : 'var(--surface)',
          border: `1px solid ${filters.fuzzy ? 'var(--accent)' : 'var(--border)'}`,
          borderRadius: 'var(--radius)',
          color: filters.fuzzy ? '#fff' : 'var(--text-2)',
          cursor: 'pointer', fontSize: '0.8rem',
          padding: '6px 12px', transition: 'all 0.15s',
        }}
        onClick={() => set('fuzzy', !filters.fuzzy)}
      >
        ≈ Fuzzy
      </button>
    </div>
  )
}
