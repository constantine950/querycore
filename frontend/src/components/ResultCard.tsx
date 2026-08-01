import type { SearchResult } from '../types'

interface CategoryStyle { bg: string; text: string }

const CATEGORY_COLORS: Record<string, CategoryStyle> = {
  computer_science: { bg: 'rgba(99,102,241,0.15)',  text: '#a5b4fc' },
  science:          { bg: 'rgba(52,211,153,0.15)',  text: '#6ee7b7' },
  history:          { bg: 'rgba(251,191,36,0.15)',  text: '#fcd34d' },
  default:          { bg: 'rgba(139,143,168,0.15)', text: '#8b8fa8' },
}

interface Props {
  result: SearchResult
  rank:   number
}

export function ResultCard({ result, rank }: Props) {
  const cat = CATEGORY_COLORS[result.category] ?? CATEGORY_COLORS['default']

  return (
    <article
      style={{
        background: 'var(--surface)',
        border: '1px solid var(--border)',
        borderRadius: 'var(--radius-lg)',
        padding: '20px 24px',
        display: 'flex',
        gap: '20px',
        transition: 'border-color 0.15s',
      }}
      onMouseEnter={e => (e.currentTarget.style.borderColor = 'var(--muted)')}
      onMouseLeave={e => (e.currentTarget.style.borderColor = 'var(--border)')}
    >
      <div style={{
        color: 'var(--muted)',
        fontFamily: 'var(--mono)',
        fontSize: '0.75rem',
        minWidth: '24px',
        paddingTop: '3px',
        userSelect: 'none',
      }}>
        {String(rank).padStart(2, '0')}
      </div>

      <div style={{ flex: 1, minWidth: 0 }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '6px', flexWrap: 'wrap' }}>
          <h3 style={{ color: 'var(--text)', fontSize: '1rem', fontWeight: 600, lineHeight: 1.3 }}>
            {result.url
              ? <a href={result.url} target="_blank" rel="noreferrer">{result.title}</a>
              : result.title}
          </h3>
          <span style={{
            background: cat.bg, color: cat.text,
            borderRadius: '4px', fontSize: '0.7rem', fontWeight: 600,
            padding: '2px 7px', textTransform: 'uppercase',
            letterSpacing: '0.05em', whiteSpace: 'nowrap',
          }}>
            {result.category?.replace('_', ' ') || 'uncategorised'}
          </span>
        </div>

        <p
          style={{ color: 'var(--text-2)', fontSize: '0.875rem', lineHeight: 1.65, marginBottom: '10px' }}
          dangerouslySetInnerHTML={{ __html: result.snippet }}
        />

        <div style={{
          display: 'flex', gap: '16px',
          fontSize: '0.75rem', color: 'var(--muted)', fontFamily: 'var(--mono)',
        }}>
          <span>score {result.score.toFixed(4)}</span>
          {result.date && <span>{result.date}</span>}
          <span>{result.doc_id}</span>
        </div>
      </div>
    </article>
  )
}
