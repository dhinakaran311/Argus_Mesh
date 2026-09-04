'use client'

import { useEffect, useState } from 'react'
import { useParams } from 'next/navigation'
import { api } from '@/lib/api'
import type { ReactFlowData } from '@/lib/api'
import RiskStamp from '@/components/RiskStamp'
import CorkboardGraph from '@/components/CorkboardGraph'
import Link from 'next/link'

// ---------------------------------------------------------------------------
// Skeleton helpers
// ---------------------------------------------------------------------------

function SkeletonRow() {
  return (
    <div style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #C8C2B266', paddingBottom: '0.45rem', alignItems: 'center' }}>
      <div className="skeleton skeleton-text" style={{ width: '45%' }} />
      <div className="skeleton skeleton-text" style={{ width: '30%' }} />
    </div>
  )
}

function LoadingSkeleton({ clusterId }: { clusterId: string }) {
  return (
    <div>
      {/* Header skeleton */}
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', flexWrap: 'wrap' }}>
          <div className="skeleton skeleton-title" style={{ width: 340 }} />
          <div className="skeleton" style={{ width: 72, height: 24 }} />
        </div>
        <p className="page-subtitle">
          <Link href="/rings" style={{ color: 'var(--muted)', textDecoration: 'none' }}>Rings</Link>
          {' / '}Case file
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: '1.5rem', alignItems: 'start' }}>
        {/* Card skeleton */}
        <div>
          <div className="panel-paper" style={{ padding: '1.5rem' }}>
            <div style={{ fontSize: '0.65rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#8A8070', marginBottom: '0.75rem' }}>
              Case file
            </div>
            <div style={{ fontFamily: 'JetBrains Mono, monospace', fontSize: '0.72rem', color: '#8A8070', marginBottom: '1.25rem', wordBreak: 'break-all' }}>
              {clusterId}
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem' }}>
              {[60, 45, 50, 35, 35, 40].map((w, i) => (
                <SkeletonRow key={i} />
              ))}
            </div>
            {/* Fetching indicator */}
            <div style={{ marginTop: '1.25rem', display: 'flex', alignItems: 'center', gap: '0.75rem', fontSize: '0.75rem', color: 'var(--muted)' }}>
              <div className="pulse-dots">
                <span /><span /><span />
              </div>
              Fetching case data…
            </div>
          </div>

          {/* Connection key placeholder */}
          <div className="panel" style={{ padding: '1rem', marginTop: '1rem' }}>
            <div className="section-label">Connection key</div>
            {[
              { color: '#C88A3B', label: 'Shared device', dash: false },
              { color: '#6B9080', label: 'Shared IP address', dash: true },
            ].map(({ color, label, dash }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.45rem', fontSize: '0.73rem', color: 'var(--muted)' }}>
                <svg width="24" height="2" style={{ flexShrink: 0 }}>
                  <line x1="0" y1="1" x2="24" y2="1" stroke={color} strokeWidth="1.5" strokeDasharray={dash ? '4 3' : undefined} />
                </svg>
                {label}
              </div>
            ))}
          </div>
        </div>

        {/* Graph skeleton */}
        <div style={{ height: 580, border: '1px solid var(--rule)', overflow: 'hidden' }}>
          <div className="corkboard" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1rem' }}>
            <div className="pulse-dots">
              <span /><span /><span />
            </div>
            <div style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>Loading entity graph…</div>
          </div>
        </div>
      </div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Graph loading state
// ---------------------------------------------------------------------------

function GraphLoading() {
  return (
    <div className="corkboard" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '1rem' }}>
      <div className="pulse-dots">
        <span /><span /><span />
      </div>
      <div style={{ color: 'var(--muted)', fontSize: '0.8rem' }}>Building connection graph…</div>
      <div style={{ color: 'var(--muted)', fontSize: '0.7rem', opacity: 0.6 }}>Querying Neo4j AuraDB</div>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main page
// ---------------------------------------------------------------------------

export default function RingDetailPage() {
  const params = useParams()
  const clusterId = decodeURIComponent(params.id as string)

  const [cluster, setCluster] = useState<any>(null)
  const [flowData, setFlowData] = useState<ReactFlowData | null | 'loading'>('loading')
  const [loading, setLoading] = useState(true)
  const [notFound, setNotFound] = useState(false)

  // Fetch cluster detail
  useEffect(() => {
    if (!clusterId) return
    api.cluster(clusterId)
      .then(c => { setCluster(c); setLoading(false) })
      .catch(() => { setNotFound(true); setLoading(false) })
  }, [clusterId])

  // Fetch graph lazily after cluster loads
  useEffect(() => {
    if (!cluster) return
    api.graph(clusterId).then(g => setFlowData(g))
  }, [cluster, clusterId])

  // --- Loading skeleton ---
  if (loading) return <LoadingSkeleton clusterId={clusterId} />

  // --- Not found ---
  if (notFound || !cluster) {
    return (
      <div style={{ padding: '3rem' }}>
        <div style={{ color: 'var(--alert)', marginBottom: '1rem', fontSize: '0.9rem', fontFamily: 'JetBrains Mono, monospace' }}>
          Cluster not found: {clusterId}
        </div>
        <Link href="/rings" className="btn">← Back to rings</Link>
      </div>
    )
  }

  const risk = cluster.combined_risk_score ?? 0

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem', flexWrap: 'wrap' }}>
          <h1 style={{
            fontFamily: 'Fraunces, Georgia, serif', fontSize: '1.75rem', fontWeight: 700,
            color: 'var(--text-strong)', letterSpacing: '-0.025em', margin: 0,
          }}>
            {clusterId}
          </h1>
          <RiskStamp level={cluster.risk_level} />
        </div>
        <p className="page-subtitle">
          <Link href="/rings" style={{ color: 'var(--muted)', textDecoration: 'none' }}>Rings</Link>
          {' / '}Case file
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '260px 1fr', gap: '1.5rem', alignItems: 'start' }}>

        {/* Paper case-file card */}
        <div>
          <div className="panel-paper" style={{ padding: '1.5rem' }}>
            <div style={{ fontSize: '0.65rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#8A8070', marginBottom: '0.75rem' }}>
              Case file
            </div>
            <h2 style={{ fontFamily: 'Fraunces, Georgia, serif', fontSize: '1.05rem', fontWeight: 700, color: 'var(--ink-paper)', margin: '0 0 1.25rem', letterSpacing: '-0.02em' }}>
              {clusterId}
            </h2>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.6rem', fontSize: '0.8rem' }}>
              {([
                ['Members',         cluster.cluster_size,                                  true],
                ['Ring type',       cluster.ring_type?.replace(/_/g, ' ') ?? 'Unknown',   false],
                ['Confirmed abuse', cluster.abuse_count ?? 0,                              true],
                ['ML score',        (cluster.avg_ml_score * 100).toFixed(1) + '%',         true],
                ['Graph score',     (cluster.graph_score * 100).toFixed(1) + '%',          true],
                ['Combined risk',   (risk * 100).toFixed(1) + '%',                         true],
              ] as [string, any, boolean][]).map(([label, value, mono]) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #C8C2B266', paddingBottom: '0.45rem' }}>
                  <span style={{ color: '#6A6050' }}>{label}</span>
                  <span style={{ fontFamily: mono ? 'JetBrains Mono, monospace' : 'Inter', fontWeight: 500, color: 'var(--ink-paper)', fontSize: '0.8rem' }}>
                    {String(value)}
                  </span>
                </div>
              ))}
            </div>

            <div style={{ marginTop: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <Link
                href={`/investigate?ring=${clusterId}`}
                className="btn btn-primary"
                style={{ width: '100%', justifyContent: 'center', fontSize: '0.8rem' }}
              >
                Start investigation
              </Link>
              <Link href="/rings" className="btn" style={{ width: '100%', justifyContent: 'center', fontSize: '0.8rem' }}>
                Back to rings
              </Link>
            </div>
          </div>

          {/* Connection key */}
          <div className="panel" style={{ padding: '1rem', marginTop: '1rem' }}>
            <div className="section-label">Connection key</div>
            {[
              { color: '#C88A3B', label: 'Shared device', dash: false },
              { color: '#6B9080', label: 'Shared IP address', dash: true },
            ].map(({ color, label, dash }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.45rem', fontSize: '0.73rem', color: 'var(--muted)' }}>
                <svg width="24" height="2" style={{ flexShrink: 0 }}>
                  <line x1="0" y1="1" x2="24" y2="1" stroke={color} strokeWidth="1.5" strokeDasharray={dash ? '4 3' : undefined} />
                </svg>
                {label}
              </div>
            ))}
          </div>
        </div>

        {/* Corkboard graph */}
        <div style={{ height: 580, border: '1px solid var(--rule)', overflow: 'hidden' }}>
          {flowData === 'loading' ? (
            <GraphLoading />
          ) : flowData && flowData.nodes.length > 0 ? (
            <CorkboardGraph flowData={flowData} />
          ) : (
            <div className="corkboard" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>Graph data unavailable</div>
              <div style={{ color: 'var(--muted)', fontSize: '0.72rem' }}>No entity connections found</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
