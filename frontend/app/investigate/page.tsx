'use client'
import { useSearchParams } from 'next/navigation'
import { useState, useRef, Suspense } from 'react'
import RiskStamp from '@/components/RiskStamp'

// SSE client — uses relative path through Next.js proxy (no CORS)
function streamInvestigation(
  clusterId: string,
  onStep: (step: string, message: string) => void,
  onReport: (data: any) => void,
  onError: (err: string) => void,
  onDone: () => void,
): () => void {
  const ctrl = new AbortController()

  fetch('/api/investigate', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ cluster_id: clusterId }),
    signal: ctrl.signal,
  }).then(async (res) => {
    if (!res.ok) {
      const text = await res.text().catch(() => res.statusText)
      onError(`Server error ${res.status}: ${text}`)
      return
    }

    const reader = res.body?.getReader()
    if (!reader) { onError('No response body'); return }

    const decoder = new TextDecoder()
    let buffer = ''

    while (true) {
      const { done, value } = await reader.read()
      if (done) { onDone(); break }

      buffer += decoder.decode(value, { stream: true })
      const lines = buffer.split('\n')
      buffer = lines.pop() ?? ''

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const raw = line.slice(6).trim()
          if (!raw || raw === '[DONE]') { onDone(); return }
          try {
            const event = JSON.parse(raw)
            const step = event.step ?? event.type ?? ''
            const msg  = event.message ?? event.label ?? step
            if (step === 'complete' || step === 'done') {
              onReport(event.data ?? event)
              onDone()
            } else if (step === 'error') {
              onError(event.message ?? 'Investigation failed')
            } else {
              onStep(step, msg)
            }
          } catch { /* skip malformed */ }
        }
      }
    }
  }).catch((err) => {
    if (err.name !== 'AbortError') onError(String(err))
  })

  return () => ctrl.abort()
}

const STEP_ORDER = ['starting', 'facts', 'graph', 'ml', 'rag', 'reasoning']

type StepStatus = 'pending' | 'active' | 'done'
interface StepItem { key: string; label: string; status: StepStatus }

function labelFor(key: string) {
  const map: Record<string, string> = {
    starting:  'Initialising investigation',
    facts:     'Pulling customer facts',
    graph:     'Querying graph connections',
    ml:        'Scoring with ML model',
    rag:       'Searching case history',
    reasoning: 'Generating case report',
    complete:  'Report ready',
  }
  return map[key] ?? key
}

function InvestigateClient() {
  const params = useSearchParams()
  // Support pre-fill from ?ring=RING-001 (linked from ring detail)
  // or ?id=customer-uuid (linked from transactions)
  const prefill = params?.get('ring') ?? params?.get('id') ?? ''

  const [clusterId, setClusterId] = useState(prefill)
  const [steps, setSteps] = useState<StepItem[]>([])
  const [report, setReport] = useState<any>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const cancelRef = useRef<(() => void) | null>(null)

  const startInvestigation = () => {
    const id = clusterId.trim()
    if (!id) return

    cancelRef.current?.()
    setReport(null)
    setError(null)
    setSteps(STEP_ORDER.map(k => ({ key: k, label: labelFor(k), status: 'pending' })))
    setRunning(true)

    const cancel = streamInvestigation(
      id,
      (step, msg) => {
        setSteps(prev => {
          const idx = prev.findIndex(s => s.key === step)
          if (idx === -1) return prev
          return prev.map((s, i) => ({
            ...s,
            status: i < idx ? 'done' : i === idx ? 'active' : s.status,
          }))
        })
      },
      (data) => {
        setReport(data)
        setSteps(prev => prev.map(s => ({ ...s, status: 'done' })))
        setRunning(false)
      },
      (err) => { setError(err); setRunning(false) },
      () => setRunning(false),
    )
    cancelRef.current = cancel
  }

  const stepIcon = (status: StepStatus) => {
    if (status === 'done')   return <span style={{ color: 'var(--confirm)' }}>✓ done</span>
    if (status === 'active') return <span style={{ color: 'var(--brass)' }}>● working</span>
    return <span style={{ color: 'var(--rule)' }}>○ pending</span>
  }

  return (
    <>
      <div className="page-header">
        <h1>Investigation</h1>
        <p className="page-subtitle">AI case analysis — LangGraph + Groq llama-3.3-70b</p>
      </div>

      {/* Input — accepts RING-001 style IDs */}
      <div style={{ marginBottom: '1.75rem', maxWidth: 560 }}>
        <div className="section-label">Ring cluster ID</div>
        <div style={{ display: 'flex', gap: '0.75rem' }}>
          <input
            id="investigate-input"
            className="input-field"
            placeholder="e.g. RING-001"
            value={clusterId}
            onChange={(e) => setClusterId(e.target.value)}
            onKeyDown={(e) => e.key === 'Enter' && !running && startInvestigation()}
            disabled={running}
          />
          <button
            id="investigate-start-btn"
            className="btn btn-primary"
            onClick={startInvestigation}
            disabled={running || !clusterId.trim()}
            style={{ whiteSpace: 'nowrap', flexShrink: 0 }}
          >
            {running ? 'Running…' : 'Start investigation'}
          </button>
        </div>
        <div style={{ marginTop: '0.4rem', fontSize: '0.72rem', color: 'var(--muted)' }}>
          Use a ring ID from the <a href="/rings" style={{ color: 'var(--brass)' }}>Rings ledger</a> (e.g. RING-001, RING-007)
        </div>
      </div>

      {/* Case log stream */}
      {steps.length > 0 && (
        <div style={{ marginBottom: '1.75rem', maxWidth: 600 }}>
          <div className="section-label">Case log</div>
          <div className="case-log-stream">
            {steps.map((s, i) => (
              <div
                key={s.key}
                className={`log-step ${s.status === 'done' ? 'done' : s.status === 'active' ? 'active' : ''} animate-in`}
              >
                <span className="step-num">{i + 1}</span>
                <span className="step-label">{s.label}</span>
                <span className="step-status">{stepIcon(s.status)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ border: '1px solid var(--alert)', borderLeft: '3px solid var(--alert)', background: 'var(--alert-dim)', padding: '0.875rem 1rem', marginBottom: '1.5rem', fontSize: '0.85rem', color: 'var(--alert)', maxWidth: 600 }}>
          {error}
        </div>
      )}

      {/* Case report */}
      {report && (
        <div className="panel-paper" style={{ padding: '2rem', maxWidth: 720 }}>
          <div style={{ fontSize: '0.65rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#8A8070', marginBottom: '0.75rem' }}>
            Investigation report — {clusterId}
          </div>

          <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.25rem', flexWrap: 'wrap' }}>
            <h2 style={{ fontFamily: 'Fraunces, Georgia, serif', fontSize: '1.2rem', fontWeight: 700, color: 'var(--ink-paper)', margin: 0 }}>
              Verdict
            </h2>
            {report.risk_level && <RiskStamp level={report.risk_level} />}
            {report.confidence && (
              <span style={{ fontSize: '0.72rem', color: '#8A8070' }}>Confidence: {report.confidence}</span>
            )}
          </div>

          {report.summary && (
            <div style={{ fontSize: '0.875rem', lineHeight: 1.75, color: 'var(--ink-paper)', marginBottom: '1.25rem' }}>
              {report.summary}
            </div>
          )}

          {Array.isArray(report.key_evidence) && report.key_evidence.length > 0 && (
            <div style={{ marginBottom: '1.25rem' }}>
              <div style={{ fontSize: '0.72rem', letterSpacing: '0.06em', textTransform: 'uppercase', color: '#8A8070', marginBottom: '0.5rem' }}>
                Key evidence
              </div>
              <ul style={{ margin: 0, padding: '0 0 0 1.2rem', fontSize: '0.82rem', lineHeight: 1.7, color: 'var(--ink-paper)' }}>
                {report.key_evidence.map((e: string, i: number) => <li key={i}>{e}</li>)}
              </ul>
            </div>
          )}

          {report.recommended_action && (
            <div style={{ background: '#D4CFC820', borderLeft: '2px solid #C88A3B', padding: '0.75rem 1rem', fontSize: '0.82rem', color: '#3A3228' }}>
              <strong>Recommended action:</strong> {report.recommended_action}
            </div>
          )}

          <hr style={{ border: 'none', borderTop: '1px solid #C8C2B2', margin: '1.5rem 0 1rem' }} />

          <div style={{ display: 'flex', gap: '0.75rem', flexWrap: 'wrap' }}>
            <button
              className="btn"
              style={{ borderColor: '#C8C2B2', background: 'transparent', color: 'var(--ink-paper)' }}
              onClick={() => {
                const txt = `INVESTIGATION REPORT\nRing: ${clusterId}\nRisk: ${report.risk_level}\n\n${report.summary}\n\nKey Evidence:\n${(report.key_evidence ?? []).join('\n')}\n\nRecommended Action: ${report.recommended_action ?? '—'}`
                const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([txt], { type: 'text/plain' }))
                a.download = `case-${clusterId}.txt`; a.click()
              }}
            >
              Export case
            </button>
            <button className="btn" style={{ borderColor: 'var(--confirm)', color: 'var(--confirm)', background: 'transparent' }}
              onClick={() => { setReport(null); setSteps([]) }}>
              Mark cleared
            </button>
            <button className="btn" onClick={() => { setReport(null); setSteps([]); setClusterId('') }}>
              New investigation
            </button>
          </div>
        </div>
      )}

      {/* Empty state */}
      {steps.length === 0 && !error && !report && (
        <div style={{ color: 'var(--muted)', fontSize: '0.85rem', maxWidth: 480 }}>
          Enter a ring ID above to open an AI investigation. The agent will pull facts, query the graph, run the ML model, search similar cases, and produce a structured case report.
        </div>
      )}
    </>
  )
}

export default function InvestigatePage() {
  return (
    <Suspense>
      <InvestigateClient />
    </Suspense>
  )
}
