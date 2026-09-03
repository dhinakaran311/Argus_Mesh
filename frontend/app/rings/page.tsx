import { api } from '@/lib/api'
import RiskStamp from '@/components/RiskStamp'
import Link from 'next/link'

async function getClusters() {
  try { return await api.clusters(100) } catch { return [] }
}

export const metadata = { title: 'Rings — AbuseRing Sentinel' }

export default async function RingsPage() {
  const clusters = await getClusters()

  return (
    <>
      <div className="page-header">
        <h1>Abuse Rings</h1>
        <p className="page-subtitle">{clusters.length} rings in the ledger</p>
      </div>

      <div className="panel">
        {clusters.length === 0 ? (
          <div style={{ padding: '2.5rem', color: 'var(--muted)', textAlign: 'center' }}>
            No rings flagged in the current dataset.
          </div>
        ) : (
          <table className="ledger">
            <thead>
              <tr>
                <th>Cluster ID</th>
                <th>Ring type</th>
                <th className="num">Members</th>
                <th className="num">Avg ML score</th>
                <th className="num">Graph score</th>
                <th className="num">Combined risk</th>
                <th>Risk level</th>
                <th></th>
              </tr>
            </thead>
            <tbody>
              {clusters.map((c) => (
                <tr key={c.cluster_id}>
                  <td className="id-cell">{c.cluster_id}</td>
                  <td style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>
                    {c.ring_type?.replace(/_/g, ' ') ?? '—'}
                  </td>
                  <td className="num font-mono-data">{c.cluster_size}</td>
                  <td className="num font-mono-data">{(c.avg_ml_score * 100).toFixed(1)}%</td>
                  <td className="num font-mono-data">{(c.graph_score * 100).toFixed(1)}%</td>
                  <td className="num font-mono-data" style={{ color: 'var(--text-strong)', fontWeight: 600 }}>
                    {(c.combined_risk_score * 100).toFixed(1)}%
                  </td>
                  <td><RiskStamp level={c.risk_level} /></td>
                  <td>
                    <Link href={`/rings/${c.cluster_id}`} className="btn" style={{ padding: '0.3rem 0.75rem', fontSize: '0.75rem' }}>
                      Open case
                    </Link>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </div>
    </>
  )
}
