import type { CSSProperties } from 'react'
import type { SearchMeta } from '../types'

interface Props {
  meta:   SearchMeta
  onPage: (page: number) => void
}

const btn = (disabled: boolean, active: boolean): CSSProperties => ({
  background: active ? 'var(--accent)' : 'var(--surface)',
  border: `1px solid ${active ? 'var(--accent)' : 'var(--border)'}`,
  borderRadius: 'var(--radius)',
  color: disabled ? 'var(--muted)' : active ? '#fff' : 'var(--text)',
  cursor: disabled ? 'not-allowed' : 'pointer',
  fontSize: '0.8rem', padding: '6px 12px',
  opacity: disabled ? 0.4 : 1, transition: 'all 0.15s',
})

export function Pagination({ meta, onPage }: Props) {
  if (meta.total_pages <= 1) return null

  const pages: (number | '…')[] = []
  for (let i = 1; i <= meta.total_pages; i++) {
    if (i === 1 || i === meta.total_pages || Math.abs(i - meta.page) <= 1) {
      pages.push(i)
    } else if (pages[pages.length - 1] !== '…') {
      pages.push('…')
    }
  }

  return (
    <div style={{ display: 'flex', gap: '6px', alignItems: 'center', justifyContent: 'center', marginTop: '24px' }}>
      <button style={btn(!meta.has_prev, false)}
        onClick={() => meta.has_prev && onPage(meta.page - 1)}
        disabled={!meta.has_prev}>← Prev</button>

      {pages.map((p, i) =>
        p === '…'
          ? <span key={`e${i}`} style={{ color: 'var(--muted)', padding: '0 4px' }}>…</span>
          : <button key={p} style={btn(false, p === meta.page)}
              onClick={() => p !== meta.page && onPage(p as number)}
            >{p}</button>
      )}

      <button style={btn(!meta.has_next, false)}
        onClick={() => meta.has_next && onPage(meta.page + 1)}
        disabled={!meta.has_next}>Next →</button>
    </div>
  )
}
