'use client'
import { useEffect, useState } from 'react'
import { api } from '@/lib/api'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer,
} from 'recharts'

export default function ModelPage() {
  const [raw, setRaw] = useState<any>(null)
  const [features, setFeatures] = useState<Array<{ feature: string; importance: number }>>([])
  const [loading, setLoading] = useState(true)

  useEffect(() => {
    Promise.all([api.modelMetrics(), api.modelFeatures()])
      .then(([m, f]) => { setRaw(m); setFeatures(Array.isArray(f) ? f.slice(0, 15) : []) })
      .finally(() => setLoading(false))
  }, [])

  if (loading) return <div style={{ color: 'var(--muted)', padding: '2rem' }}>Loading model data…</div>

  // Unwrap nested primary_metrics
  const pm = raw?.primary_metrics ?? {}
  const metrics = {
    auc_roc:   raw?.roc_auc ?? pm.auc_roc ?? 0,
    precision: pm.precision ?? 0,
    recall:    pm.recall ?? 0,
    f1:        pm.f1 ?? 0,
    accuracy:  pm.accuracy ?? (pm.tp != null && pm.tn != null
      ? (pm.tp + pm.tn) / (pm.tp + pm.tn + (pm.fp ?? 0) + (pm.fn ?? 0)) : 0),
    tp: pm.tp ?? 0,
    tn: pm.tn ?? 0,
    fp: pm.fp ?? 0,
    fn: pm.fn ?? 0,
    threshold: pm.threshold ?? 0.1,
  }

  const confusionData = [
    { name: 'True Positive',  value: metrics.tp },
    { name: 'True Negative',  value: metrics.tn },
    { name: 'False Positive', value: metrics.fp },
    { name: 'False Negative', value: metrics.fn },
  ]

  const tooltipStyle = {
    background: 'var(--surface)',
    border: '1px solid var(--rule)',
    borderRadius: 0,
    color: 'var(--text)',
    fontSize: '0.78rem',
    fontFamily: 'JetBrains Mono',
  }

  const pct = (n: number) => (n * 100).toFixed(1) + '%'

  return (
    <>
      <div className="page-header">
        <h1>Model performance</h1>
        <p className="page-subtitle">XGBoost fraud classifier — {raw?.model_version ?? 'v1'}</p>
      </div>

      {/* Key metrics */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(5, 1fr)', gap: '1px', background: 'var(--rule)', border: '1px solid var(--rule)', marginBottom: '2rem' }}>
        {[
          { label: 'AUC-ROC',   value: pct(metrics.auc_roc) },
          { label: 'Precision', value: pct(metrics.precision) },
          { label: 'Recall',    value: pct(metrics.recall) },
          { label: 'F1 score',  value: pct(metrics.f1) },
          { label: 'Accuracy',  value: pct(metrics.accuracy) },
        ].map(({ label, value }) => (
          <div key={label} style={{ background: 'var(--surface)', padding: '1.25rem' }}>
            <div className="section-label" style={{ marginBottom: '0.4rem' }}>{label}</div>
            <div style={{ fontFamily: 'Fraunces, Georgia, serif', fontSize: '1.5rem', fontWeight: 700, color: 'var(--text-strong)', letterSpacing: '-0.025em' }}>
              {value}
            </div>
          </div>
        ))}
      </div>

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '1.75rem' }}>

        {/* SHAP feature importance */}
        <div>
          <div className="section-label">Feature importance (SHAP)</div>
          <div className="panel" style={{ padding: '1.25rem' }}>
            {features.length === 0 ? (
              <div style={{ color: 'var(--muted)', fontSize: '0.85rem' }}>No feature data available.</div>
            ) : (
              <ResponsiveContainer width="100%" height={380}>
                <BarChart data={features} layout="vertical" margin={{ left: 0, right: 20, top: 0, bottom: 0 }}>
                  <XAxis type="number" stroke="var(--muted)" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono', fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
                  <YAxis type="category" dataKey="feature" width={140} stroke="var(--muted)" tick={{ fontSize: 10, fontFamily: 'Inter', fill: 'var(--text)' }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'var(--brass-dim)' }} />
                  <Bar dataKey="importance" fill="var(--brass)" radius={0} maxBarSize={10} />
                </BarChart>
              </ResponsiveContainer>
            )}
          </div>
        </div>

        {/* Prediction breakdown */}
        <div>
          <div className="section-label">Prediction breakdown</div>
          <div className="panel" style={{ padding: '1.25rem' }}>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={confusionData} margin={{ left: 0, right: 0, top: 0, bottom: 0 }}>
                <XAxis dataKey="name" stroke="var(--muted)" tick={{ fontSize: 10, fontFamily: 'Inter', fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
                <YAxis stroke="var(--muted)" tick={{ fontSize: 11, fontFamily: 'JetBrains Mono', fill: 'var(--muted)' }} axisLine={false} tickLine={false} />
                <Tooltip contentStyle={tooltipStyle} cursor={{ fill: 'var(--brass-dim)' }} />
                <Bar dataKey="value" fill="var(--brass)" radius={0} maxBarSize={40} />
              </BarChart>
            </ResponsiveContainer>

            <hr style={{ border: 'none', borderTop: '1px solid var(--rule)', margin: '1.25rem 0 1rem' }} />

            <table className="ledger">
              <tbody>
                {[
                  ['Evaluation date', raw?.evaluation_date?.slice(0, 10) ?? '—'],
                  ['Test samples',    raw?.n_test_samples ?? '—'],
                  ['Risk threshold',  metrics.threshold.toFixed(2)],
                  ['Features',        raw?.model_meta?.num_features ?? 41],
                  ['Dataset split',   raw?.dataset_split ?? '—'],
                ].map(([label, value]) => (
                  <tr key={String(label)}>
                    <td style={{ color: 'var(--muted)', fontSize: '0.78rem' }}>{label}</td>
                    <td className="num font-mono-data">{String(value)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </>
  )
}
