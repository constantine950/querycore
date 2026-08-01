import { useState } from 'react'
import { LandingPage } from './pages/LandingPage.tsx'
import { SearchPage }  from './pages/SearchPage.tsx'

type Page = 'landing' | 'search' | 'analytics'

function AnalyticsPage() {
  return (
    <div style={{ maxWidth: '780px', margin: '0 auto', padding: '40px 20px', color: 'var(--text-2)', textAlign: 'center' }}>
      Analytics dashboard — coming Day 24.
    </div>
  )
}

const NAV: { id: Page; label: string }[] = [
  { id: 'search',    label: 'Search'    },
  { id: 'analytics', label: 'Analytics' },
]

export default function App() {
  const [page, setPage] = useState<Page>('landing')

  if (page === 'landing') {
    return <LandingPage onEnter={() => setPage('search')} />
  }

  return (
    <>
      <nav style={{
        borderBottom: '1px solid var(--border)', padding: '0 20px',
        display: 'flex', gap: '4px', alignItems: 'center',
      }}>
        <button
          onClick={() => setPage('landing')}
          style={{
            background: 'none', border: 'none', color: 'var(--accent)',
            cursor: 'pointer', fontFamily: 'var(--mono)', fontSize: '0.85rem',
            fontWeight: 600, padding: '14px 12px 12px', marginRight: '8px',
          }}
        >
          QueryCore
        </button>

        {NAV.map(({ id, label }) => (
          <button
            key={id}
            onClick={() => setPage(id)}
            style={{
              background: 'none', border: 'none',
              borderBottom: `2px solid ${page === id ? 'var(--accent)' : 'transparent'}`,
              color: page === id ? 'var(--text)' : 'var(--text-2)',
              cursor: 'pointer', fontFamily: 'var(--sans)', fontSize: '0.875rem',
              fontWeight: page === id ? 600 : 400,
              padding: '14px 12px 12px', transition: 'color 0.15s',
            }}
          >
            {label}
          </button>
        ))}
      </nav>

      {page === 'search'    && <SearchPage />}
      {page === 'analytics' && <AnalyticsPage />}
    </>
  )
}
