import { useState, useRef, useEffect, CSSProperties, KeyboardEvent } from 'react'
import { useAutocomplete } from '../hooks/useAutocomplete'
import type { SearchParams } from '../types'

interface Props {
  onSearch: (params: SearchParams) => void
  loading:  boolean
}

const styles: Record<string, CSSProperties> = {
  wrap:  { position: 'relative', width: '100%' },
  row:   { display: 'flex', gap: '8px', alignItems: 'center' },
  input: {
    flex: 1,
    background: 'var(--surface)',
    border: '1.5px solid var(--border)',
    borderRadius: 'var(--radius)',
    color: 'var(--text)',
    fontFamily: 'var(--mono)',
    fontSize: '1rem',
    padding: '12px 16px',
    outline: 'none',
    transition: 'border-color 0.15s, box-shadow 0.15s',
  },
  inputFocus: {
    borderColor: 'var(--accent)',
    boxShadow: '0 0 0 3px var(--accent-glow)',
  },
  btn: {
    background: 'var(--accent)',
    border: 'none',
    borderRadius: 'var(--radius)',
    color: '#fff',
    cursor: 'pointer',
    fontFamily: 'var(--sans)',
    fontSize: '0.9rem',
    fontWeight: 600,
    padding: '12px 20px',
    transition: 'background 0.15s',
    whiteSpace: 'nowrap',
  },
  dropdown: {
    position: 'absolute',
    top: 'calc(100% + 4px)',
    left: 0,
    right: 0,
    background: 'var(--surface)',
    border: '1.5px solid var(--border)',
    borderRadius: 'var(--radius)',
    overflow: 'hidden',
    zIndex: 100,
    boxShadow: '0 8px 24px rgba(0,0,0,0.4)',
  },
  suggestion: {
    padding: '10px 16px',
    cursor: 'pointer',
    fontFamily: 'var(--mono)',
    fontSize: '0.875rem',
    color: 'var(--text-2)',
    transition: 'background 0.1s, color 0.1s',
  },
}

export function SearchBar({ onSearch, loading }: Props) {
  const [query,   setQuery]   = useState('')
  const [focused, setFocused] = useState(false)

  const { suggestions, open, close } = useAutocomplete(query, focused)

  const submit = (q = query) => {
    if (!q.trim()) return
    close()
    onSearch({ q: q.trim() })
  }

  const handleKey = (e: KeyboardEvent<HTMLInputElement>) => {
    if (e.key === 'Enter')  submit()
    if (e.key === 'Escape') close()
  }

  const pickSuggestion = (s: string) => {
    setQuery(s)
    close()
    onSearch({ q: s })
  }

  useEffect(() => {
    const handler = (e: MouseEvent) => {
      if (!(e.target as Element).closest('[data-searchbar]')) close()
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  return (
    <div style={styles.wrap} data-searchbar="">
      <div style={styles.row}>
        <input
          style={{ ...styles.input, ...(focused ? styles.inputFocus : {}) }}
          value={query}
          onChange={e => setQuery(e.target.value)}
          onKeyDown={handleKey}
          onFocus={() => setFocused(true)}
          onBlur={() => setFocused(false)}
          placeholder='Search documents… or try "inverted index"'
          autoComplete="off"
          spellCheck={false}
        />
        <button
          style={styles.btn}
          onClick={() => submit()}
          disabled={loading}
        >
          {loading ? '…' : 'Search'}
        </button>
      </div>

      {open && suggestions.length > 0 && (
        <div style={styles.dropdown}>
          {suggestions.map((s, i) => (
            <div
              key={i}
              style={styles.suggestion}
              onMouseDown={() => pickSuggestion(s)}
              onMouseEnter={e => {
                (e.target as HTMLElement).style.background = 'var(--muted)'
                ;(e.target as HTMLElement).style.color = 'var(--text)'
              }}
              onMouseLeave={e => {
                (e.target as HTMLElement).style.background = 'transparent'
                ;(e.target as HTMLElement).style.color = 'var(--text-2)'
              }}
            >
              {s}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
