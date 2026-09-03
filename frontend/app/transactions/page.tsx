import { api } from '@/lib/api'
import RiskStamp from '@/components/RiskStamp'
import Link from 'next/link'

export const metadata = { title: 'Transactions — AbuseRing Sentinel' }

async function getTransactions() {
  try { return await api.transactions(200, 0) } catch { return [] }
}

function fmtCurrency(n: number) {
  return '₹' + new Intl.NumberFormat('en-IN').format(Math.round(n))
}

export default async function TransactionsPage() {
  const txns = await getTransactions()

  return (
    <>
      <div className="page-header">
        <h1>Transactions</h1>
        <p className="page-subtitle">{txns.length} records in the ledger</p>
      </div>

      <div className="panel">
        <table className="ledger">
          <thead>
            <tr>
              <th>Transaction ID</th>
              <th>Customer</th>
              <th>Merchant</th>
              <th className="num">Amount</th>
              <th>Status</th>
              <th className="num">Risk score</th>
              <th>Risk level</th>
              <th></th>
            </tr>
          </thead>
          <tbody>
            {txns.length === 0 ? (
              <tr>
                <td colSpan={8} style={{ textAlign: 'center', color: 'var(--muted)', padding: '2.5rem' }}>
                  No transactions in the case log.
                </td>
              </tr>
            ) : txns.map((t, i) => (
              <tr key={t.transaction_id ?? i}>
                <td className="id-cell">{t.transaction_id}</td>
                <td>
                  <Link href={`/investigate?id=${t.customer_id}`} style={{ color: 'var(--brass)', textDecoration: 'none', fontFamily: 'JetBrains Mono', fontSize: '0.78rem' }}>
                    {t.customer_id}
                  </Link>
                </td>
                <td className="id-cell" style={{ fontSize: '0.75rem', color: 'var(--muted)' }}>
                  {t.merchant_id ?? '—'}
                </td>
                <td className="num font-mono-data">{fmtCurrency(t.amount ?? 0)}</td>
                <td style={{ fontSize: '0.78rem', color: 'var(--muted)' }}>{t.status}</td>
                <td className="num font-mono-data">
                  {t.risk_score != null ? (t.risk_score * 100).toFixed(1) + '%' : '—'}
                </td>
                <td><RiskStamp level={t.risk_level ?? 'LOW'} /></td>
                <td>
                  <Link href={`/investigate?id=${t.customer_id}`} className="btn" style={{ padding: '0.25rem 0.65rem', fontSize: '0.72rem' }}>
                    Investigate
                  </Link>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </>
  )
}
