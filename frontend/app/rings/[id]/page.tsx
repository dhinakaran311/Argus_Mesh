import { api } from '@/lib/api'
import RiskStamp from '@/components/RiskStamp'
import CorkboardGraph from '@/components/CorkboardGraph'
import Link from 'next/link'
import { notFound } from 'next/navigation'

interface Props { params: { id: string } }

export default async function RingDetailPage({ params }: Props) {
  const id = decodeURIComponent(params.id)

  let cluster = null
  let flowData = null

  try { cluster = await api.cluster(id) } catch {}
  try { flowData = await api.graph(id) } catch {}

  if (!cluster) return notFound()

  const risk = cluster.combined_risk_score ?? 0

  return (
    <div>
      <div className="page-header">
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '1rem' }}>
          <h1 style={{ fontFamily: 'Fraunces, Georgia, serif', fontSize: '1.75rem', fontWeight: 700, color: 'var(--text-strong)', letterSpacing: '-0.025em', margin: 0 }}>
            {id}
          </h1>
          <RiskStamp level={cluster.risk_level} />
        </div>
        <p className="page-subtitle">
          <Link href="/rings" style={{ color: 'var(--muted)', textDecoration: 'none' }}>Rings</Link>
          {' '}/{' '}Case file
        </p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '280px 1fr', gap: '1.5rem', alignItems: 'start' }}>

        {/* Case file metadata card — paper surface */}
        <div>
          <div className="panel-paper" style={{ padding: '1.5rem' }}>
            <div style={{ fontSize: '0.65rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#8A8070', marginBottom: '0.75rem' }}>
              Case file
            </div>
            <h2 style={{ fontFamily: 'Fraunces, Georgia, serif', fontSize: '1.1rem', fontWeight: 700, color: 'var(--ink-paper)', margin: '0 0 1.25rem', letterSpacing: '-0.02em' }}>
              {id}
            </h2>

            <div style={{ display: 'flex', flexDirection: 'column', gap: '0.75rem', fontSize: '0.82rem' }}>
              {[
                { label: 'Members', value: cluster.cluster_size, mono: true },
                { label: 'Ring type', value: cluster.ring_type?.replace(/_/g, ' ') ?? 'Unknown' },
                { label: 'Confirmed abuse', value: cluster.abuse_count ?? 0, mono: true },
                { label: 'ML score', value: (cluster.avg_ml_score * 100).toFixed(1) + '%', mono: true },
                { label: 'Graph score', value: (cluster.graph_score * 100).toFixed(1) + '%', mono: true },
                { label: 'Combined risk', value: (risk * 100).toFixed(1) + '%', mono: true },
              ].map(({ label, value, mono }) => (
                <div key={label} style={{ display: 'flex', justifyContent: 'space-between', borderBottom: '1px solid #C8C2B266', paddingBottom: '0.5rem' }}>
                  <span style={{ color: '#6A6050' }}>{label}</span>
                  <span style={{ fontFamily: mono ? 'JetBrains Mono, monospace' : 'Inter', fontWeight: 500, color: 'var(--ink-paper)' }}>
                    {String(value)}
                  </span>
                </div>
              ))}
            </div>

            <div style={{ marginTop: '1.25rem', display: 'flex', flexDirection: 'column', gap: '0.5rem' }}>
              <Link href={`/investigate?id=${cluster.member_ids?.[0] ?? ''}`} className="btn btn-primary" style={{ width: '100%', justifyContent: 'center' }}>
                Start investigation
              </Link>
              <Link href="/rings" className="btn" style={{ width: '100%', justifyContent: 'center' }}>
                Back to rings
              </Link>
            </div>
          </div>

          {/* Legend — case key */}
          <div className="panel" style={{ padding: '1rem', marginTop: '1rem' }}>
            <div className="section-label">Connection key</div>
            {[
              { color: '#C88A3B', label: 'Shared device', dash: false },
              { color: '#6B9080', label: 'Shared IP address', dash: true },
              { color: '#5A5D6B', label: 'Shared payment card', dash: false },
            ].map(({ color, label, dash }) => (
              <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.6rem', marginBottom: '0.5rem', fontSize: '0.75rem', color: 'var(--muted)' }}>
                <svg width="24" height="2">
                  <line x1="0" y1="1" x2="24" y2="1" stroke={color} strokeWidth="1.5" strokeDasharray={dash ? '4 3' : undefined} />
                </svg>
                {label}
              </div>
            ))}
          </div>
        </div>

        {/* Corkboard graph */}
        <div style={{ height: 560, border: '1px solid var(--rule)', overflow: 'hidden' }}>
          {flowData ? (
            <CorkboardGraph flowData={flowData} />
          ) : (
            <div className="corkboard" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center', flexDirection: 'column', gap: '0.5rem' }}>
              <div style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>Graph data unavailable</div>
              <div style={{ color: 'var(--muted)', fontSize: '0.75rem' }}>Neo4j connection required</div>
            </div>
          )}
        </div>
      </div>
    </div>
  )
}
