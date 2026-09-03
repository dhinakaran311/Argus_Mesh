// Dashboard page — case log front page
import { api } from '@/lib/api'
import RiskStamp from '@/components/RiskStamp'
import Link from 'next/link'

function fmtCurrency(n: number) {
  return '₹' + new Intl.NumberFormat('en-IN').format(Math.round(n))
}

async function getDashboard() {
  try { return await api.dashboard() } catch { return null }
}

async function getRecentActivity() {
  try {
    const txns = await api.transactions(25, 0)
    return Array.isArray(txns) ? txns : []
  } catch { return [] }
}

async function getClusters() {
  try {
    const clusters = await api.clusters(8)
    return Array.isArray(clusters) ? clusters : []
  } catch { return [] }
}

export const metadata = { title: 'Case Log — AbuseRing Sentinel' }

export default async function DashboardPage() {
  const [stats, txns, clusters] = await Promise.all([
    getDashboard(),
    getRecentActivity(),
    getClusters(),
  ])

  const maxSize = clusters.length > 0 ? Math.max(...clusters.map(c => c.cluster_size)) : 1
  const now = new Date().toLocaleString('en-IN', {
    hour: '2-digit', minute: '2-digit',
    day: '2-digit', month: 'short', year: 'numeric'
  })

  return (
    <>
      <div className="page-header">
        <h1>Case Log</h1>
        <p className="page-subtitle">Fraud ring intelligence — updated in real time</p>
      </div>

      {/* Stamped summary line */}
      <div className="stamped-summary" style={{ marginBottom: '1.75rem' }}>
        <div className="stat-item">
          <span className="stat-label">rings identified</span>
          <span className="stat-value brass">{stats ? stats.total_rings.toLocaleString('en-IN') : '—'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">customers flagged</span>
          <span className="stat-value">{stats ? stats.abuse_customers.toLocaleString('en-IN') : '—'}</span>
        </div>
        <div className="stat-item">
          <span className="stat-label">critical</span>
          <span className="stat-value" style={{ color: 'var(--alert)' }}>
            {stats ? stats.critical_count.toLocaleString('en-IN') : '—'}
          </span>
        </div>
        <div className="stat-item">
          <span className="stat-label">avg risk score</span>
          <span className="stat-value">{stats ? (stats.avg_risk_score * 100).toFixed(0) + '%' : '—'}</span>
        </div>
        <span className="stamp-meta">Last updated {now}</span>
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 340px', gap: '1.75rem', alignItems: 'start' }}>

        {/* Recent high-risk activity ledger */}
        <div>
          <div className="section-label">Recent high-risk activity</div>
          <div className="panel">
            {txns.length === 0 ? (
              <div style={{ padding: '2rem', color: 'var(--muted)', fontSize: '0.85rem' }}>
                No transactions in the case log today.
              </div>
            ) : (
              <table className="ledger">
                <thead>
                  <tr>
                    <th>Customer</th>
                    <th>Merchant</th>
                    <th className="num">Amount</th>
                    <th>Status</th>
                    <th>Risk</th>
                  </tr>
                </thead>
                <tbody>
                  {txns.map((t, i) => (
                    <tr key={t.transaction_id ?? i}>
                      <td>
                        <Link
                          href={`/investigate?id=${t.customer_id}`}
                          style={{ color: 'var(--brass)', textDecoration: 'none', fontFamily: 'JetBrains Mono', fontSize: '0.75rem' }}
                        >
                          {t.customer_id?.slice(0, 12)}…
                        </Link>
                      </td>
                      <td className="id-cell" style={{ fontSize: '0.72rem', color: 'var(--muted)' }}>
                        {t.merchant_id?.slice(0, 10) ?? '—'}…
                      </td>
                      <td className="num font-mono-data">{fmtCurrency(t.amount ?? 0)}</td>
                      <td style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>{t.status}</td>
                      <td><RiskStamp level={t.risk_level ?? 'LOW'} /></td>
                    </tr>
                  ))}
                </tbody>
              </table>
            )}
          </div>
        </div>

        {/* Active rings by member count */}
        <div>
          <div className="section-label">Active rings by size</div>
          <div className="panel" style={{ padding: '1rem 1.25rem' }}>
            {clusters.length === 0 ? (
              <div style={{ color: 'var(--muted)', fontSize: '0.82rem', padding: '0.5rem 0' }}>
                No ring data — Neo4j graph may be empty.
                <br />
                <Link href="/rings" style={{ color: 'var(--brass)', fontSize: '0.78rem' }}>View rings ledger</Link>
              </div>
            ) : (
              clusters.map((c) => (
                <Link key={c.cluster_id} href={`/rings/${c.cluster_id}`} style={{ textDecoration: 'none' }}>
                  <div className="bar-row" style={{ cursor: 'pointer' }}>
                    <span className="bar-id">{c.cluster_id?.slice(0, 10) ?? '—'}</span>
                    <div className="bar-track">
                      <div className="bar-fill" style={{ width: `${(c.cluster_size / maxSize) * 100}%` }} />
                    </div>
                    <span className="bar-count">{c.cluster_size} members</span>
                  </div>
                </Link>
              ))
            )}

            <hr className="section-divider" style={{ margin: '1rem 0 0.75rem' }} />
            <Link href="/rings" className="btn" style={{ width: '100%', justifyContent: 'center' }}>
              View all rings
            </Link>
          </div>
        </div>

      </div>
    </>
  )
}
