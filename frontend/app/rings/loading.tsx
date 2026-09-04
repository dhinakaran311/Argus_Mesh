// Rings list loading state — shown by Next.js while the server component fetches data
export default function RingsLoading() {
  return (
    <>
      <div className="page-header">
        <h1>Abuse Rings</h1>
        <div style={{ display: 'flex', alignItems: 'center', gap: '0.75rem', marginTop: '0.3rem' }}>
          <div className="pulse-dots">
            <span /><span /><span />
          </div>
          <span style={{ fontSize: '0.82rem', color: 'var(--muted)' }}>Loading ring ledger…</span>
        </div>
      </div>

      <div className="panel" style={{ overflow: 'hidden' }}>
        {/* Table header skeleton */}
        <div style={{ display: 'grid', gridTemplateColumns: '2fr 1.2fr repeat(4, 1fr) 80px 80px', gap: '0 0.875rem', padding: '0.6rem 0.875rem', borderBottom: '1px solid var(--rule)' }}>
          {['Cluster ID', 'Ring type', 'Members', 'ML score', 'Graph score', 'Combined', '', ''].map((h, i) => (
            <div key={i} style={{ fontSize: '0.68rem', color: 'var(--muted)', fontWeight: 500, letterSpacing: '0.04em' }}>{h}</div>
          ))}
        </div>

        {/* Skeleton rows */}
        {Array.from({ length: 12 }).map((_, i) => (
          <div
            key={i}
            style={{
              display: 'grid',
              gridTemplateColumns: '2fr 1.2fr repeat(4, 1fr) 80px 80px',
              gap: '0 0.875rem',
              padding: '0.6rem 0.875rem',
              borderBottom: '1px solid var(--rule)',
              alignItems: 'center',
              opacity: 1 - i * 0.06,
            }}
          >
            <div className="skeleton skeleton-text" style={{ width: '80%' }} />
            <div className="skeleton skeleton-text" style={{ width: '60%' }} />
            <div className="skeleton skeleton-text" style={{ width: '50%' }} />
            <div className="skeleton skeleton-text" style={{ width: '55%' }} />
            <div className="skeleton skeleton-text" style={{ width: '55%' }} />
            <div className="skeleton skeleton-text" style={{ width: '60%' }} />
            <div className="skeleton" style={{ width: 52, height: 20 }} />
            <div className="skeleton" style={{ width: 68, height: 26 }} />
          </div>
        ))}
      </div>
    </>
  )
}
