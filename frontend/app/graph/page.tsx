'use client'
import { useEffect, useState } from 'react'
import { api, type ClusterSummary } from '@/lib/api'
import CorkboardGraph from '@/components/CorkboardGraph'
import RiskStamp from '@/components/RiskStamp'

export default function GraphPage() {
  const [clusters, setClusters] = useState<ClusterSummary[]>([])
  const [selected, setSelected] = useState<ClusterSummary | null>(null)
  const [flowData, setFlowData] = useState<any>(null)
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    api.clusters(20).then(setClusters).finally(() => setLoading(false))
  }, [])

  const loadGraph = async (c: ClusterSummary) => {
    setSelected(c)
    setFlowData(null)
    try {
      const data = await api.graph(c.cluster_id)
      setFlowData(data)
    } catch {
      setFlowData(null)
    }
  }

  return (
    <>
      <div className="page-header">
        <h1>Graph</h1>
        <p className="page-subtitle">Ring connection map — select a cluster to load</p>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '240px 1fr', gap: '1.5rem', height: 640 }}>

        {/* Cluster selector */}
        <div className="panel" style={{ overflow: 'auto', height: '100%' }}>
          <div style={{ padding: '0.75rem', borderBottom: '1px solid var(--rule)' }}>
            <div className="section-label" style={{ marginBottom: 0 }}>Select ring</div>
          </div>
          {loading ? (
            <div style={{ padding: '1rem', color: 'var(--muted)', fontSize: '0.8rem' }}>Loading…</div>
          ) : clusters.map((c) => (
            <button
              key={c.cluster_id}
              onClick={() => loadGraph(c)}
              style={{
                width: '100%', textAlign: 'left', padding: '0.65rem 0.875rem',
                background: selected?.cluster_id === c.cluster_id ? 'var(--brass-dim)' : 'transparent',
                border: 'none', borderBottom: '1px solid var(--rule)', cursor: 'pointer',
                borderLeft: selected?.cluster_id === c.cluster_id ? '2px solid var(--brass)' : '2px solid transparent',
              }}
            >
              <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.72rem', color: selected?.cluster_id === c.cluster_id ? 'var(--brass)' : 'var(--text-strong)', marginBottom: 3 }}>
                {c.cluster_id}
              </div>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <span style={{ fontSize: '0.68rem', color: 'var(--muted)' }}>{c.cluster_size} members</span>
                <RiskStamp level={c.risk_level} />
              </div>
            </button>
          ))}
        </div>

        {/* Canvas */}
        <div style={{ border: '1px solid var(--rule)', overflow: 'hidden', height: '100%' }}>
          {!selected ? (
            <div className="corkboard" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ textAlign: 'center' }}>
                <div style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>Select a ring from the list</div>
                <div style={{ color: 'var(--muted)', fontSize: '0.72rem', marginTop: '0.4rem' }}>Graph connections will load here</div>
              </div>
            </div>
          ) : !flowData ? (
            <div className="corkboard" style={{ height: '100%', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
              <div style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>Loading graph…</div>
            </div>
          ) : (
            <CorkboardGraph flowData={flowData} />
          )}
        </div>
      </div>

      {/* Legend */}
      <div style={{ marginTop: '1rem', display: 'flex', gap: '2rem', padding: '0.875rem 1rem', border: '1px solid var(--rule)', background: 'var(--surface)', alignItems: 'center' }}>
        <div className="section-label" style={{ marginBottom: 0 }}>Connection key</div>
        {[
          { color: '#C88A3B', label: 'Shared device', dash: false },
          { color: '#6B9080', label: 'Shared IP address', dash: true },
          { color: '#5A5D6B', label: 'Shared payment card', dash: false },
        ].map(({ color, label, dash }) => (
          <div key={label} style={{ display: 'flex', alignItems: 'center', gap: '0.5rem', fontSize: '0.75rem', color: 'var(--muted)' }}>
            <svg width="20" height="2"><line x1="0" y1="1" x2="20" y2="1" stroke={color} strokeWidth="1.5" strokeDasharray={dash ? '4 3' : undefined} /></svg>
            {label}
          </div>
        ))}
      </div>
    </>
  )
}
