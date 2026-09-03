'use client'
import { useSearchParams } from 'next/navigation'
import { useState, useRef, Suspense } from 'react'
import { streamInvestigation, type SSEStep } from '@/lib/sse'
import RiskStamp from '@/components/RiskStamp'

const API_BASE = process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000'

const STEP_LABELS = [
  'Pulling customer record',
  'Querying graph connections',
  'Scoring with ML model',
  'Searching case history',
  'Generating case report',
]

type StepState = 'pending' | 'running' | 'done' | 'error'

interface StepItem {
  num: number
  label: string
  state: StepState
}

function InvestigateClient() {
  const params = useSearchParams()
  const [customerId, setCustomerId] = useState(params?.get('id') ?? '')
  const [steps, setSteps] = useState<StepItem[]>([])
  const [report, setReport] = useState<string | null>(null)
  const [verdict, setVerdict] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const cancelRef = useRef<(() => void) | null>(null)

  const startInvestigation = () => {
    if (!customerId.trim()) return
    setReport(null)
    setVerdict(null)
    setError(null)
    setSteps(STEP_LABELS.map((label, i) => ({ num: i + 1, label, state: 'pending' })))
    setRunning(true)

    const cancel = streamInvestigation(
      API_BASE,
      customerId.trim(),
      (event: SSEStep) => {
        if (event.type === 'step' && event.step !== undefined) {
          setSteps((prev) => prev.map((s) =>
            s.num === event.step ? { ...s, state: event.status === 'done' ? 'done' : 'running' }
            : s.num === event.step - 1 ? { ...s, state: 'done' } : s
          ))
        }
        if (event.type === 'report' || event.report) {
          setReport(event.report ?? null)
          setVerdict(event.verdict ?? null)
          setSteps((prev) => prev.map((s) => ({ ...s, state: 'done' })))
        }
        if (event.type === 'error') {
          setError(event.error ?? 'Investigation failed')
          setRunning(false)
        }
      },
      () => setRunning(false),
      (err) => { setError(err); setRunning(false) }
    )
    cancelRef.current = cancel
  }

  const stepIcon = (state: StepState) => {
    if (state === 'done')    return <span style={{ color: 'var(--confirm)' }}>✓ done</span>
    if (state === 'running') return <span style={{ color: 'var(--brass)' }}>● working</span>
    if (state === 'error')   return <span style={{ color: 'var(--alert)' }}>✗ error</span>
    return <span style={{ color: 'var(--rule)' }}>○ pending</span>
  }

  return (
    <>
      <div className="page-header">
        <h1>Investigation</h1>
        <p className="page-subtitle">AI-powered case analysis via LangGraph + Groq</p>
      </div>

      {/* Input */}
      <div style={{ display: 'flex', gap: '0.75rem', marginBottom: '1.75rem', maxWidth: 520 }}>
        <input
          className="input-field"
          placeholder="Customer ID — e.g. CUST_04821"
          value={customerId}
          onChange={(e) => setCustomerId(e.target.value)}
          onKeyDown={(e) => e.key === 'Enter' && !running && startInvestigation()}
          disabled={running}
        />
        <button
          className="btn btn-primary"
          onClick={startInvestigation}
          disabled={running || !customerId.trim()}
          style={{ whiteSpace: 'nowrap', flexShrink: 0 }}
        >
          {running ? 'Running…' : 'Start investigation'}
        </button>
      </div>

      {/* Case log stream — the ONE orchestrated motion moment */}
      {steps.length > 0 && (
        <div style={{ marginBottom: '1.75rem', maxWidth: 640 }}>
          <div className="section-label">Case log</div>
          <div className="case-log-stream">
            {steps.map((s) => (
              <div
                key={s.num}
                className={`log-step ${s.state === 'done' ? 'done' : s.state === 'running' ? 'active' : ''} ${s.state !== 'pending' ? 'animate-in' : ''}`}
              >
                <span className="step-num">{s.num}</span>
                <span className="step-label">{s.label}</span>
                <span className="step-status">{stepIcon(s.state)}</span>
              </div>
            ))}
          </div>
        </div>
      )}

      {/* Error */}
      {error && (
        <div style={{ border: '1px solid var(--alert)', borderLeft: '3px solid var(--alert)', background: 'var(--alert-dim)', padding: '0.875rem 1rem', marginBottom: '1.5rem', fontSize: '0.85rem', color: 'var(--alert)' }}>
          {error}
        </div>
      )}

      {/* Case report — paper surface */}
      {report && (
        <div className="panel-paper" style={{ padding: '2rem', maxWidth: 720 }}>
          <div style={{ fontSize: '0.65rem', letterSpacing: '0.1em', textTransform: 'uppercase', color: '#8A8070', marginBottom: '0.75rem' }}>
            Investigation report — {customerId}
          </div>

          {verdict && (
            <div style={{ display: 'flex', alignItems: 'center', gap: '1rem', marginBottom: '1.25rem' }}>
              <h2 style={{ fontFamily: 'Fraunces, Georgia, serif', fontSize: '1.25rem', fontWeight: 700, color: 'var(--ink-paper)', margin: 0 }}>
                Verdict
              </h2>
              <RiskStamp level={verdict} />
            </div>
          )}

          <div style={{ fontSize: '0.875rem', lineHeight: 1.75, color: 'var(--ink-paper)', whiteSpace: 'pre-wrap', fontFamily: 'Inter, sans-serif' }}>
            {report}
          </div>

          <hr style={{ border: 'none', borderTop: '1px solid #C8C2B2', margin: '1.5rem 0' }} />

          <div style={{ display: 'flex', gap: '0.75rem' }}>
            <button
              className="btn"
              style={{ borderColor: '#C8C2B2', background: 'transparent', color: 'var(--ink-paper)' }}
              onClick={() => {
                const blob = new Blob([`INVESTIGATION REPORT\n${customerId}\n\n${report}`], { type: 'text/plain' })
                const a = document.createElement('a'); a.href = URL.createObjectURL(blob); a.download = `case-${customerId}.txt`; a.click()
              }}
            >
              Export case
            </button>
            <button
              className="btn"
              style={{ borderColor: 'var(--confirm)', color: 'var(--confirm)', background: 'transparent' }}
              onClick={() => { setReport(null); setSteps([]); setCustomerId('') }}
            >
              Mark cleared
            </button>
          </div>
        </div>
      )}

      {/* Empty state */}
      {steps.length === 0 && !error && (
        <div style={{ color: 'var(--muted)', fontSize: '0.85rem', marginTop: '2rem' }}>
          Enter a customer ID above to open an investigation. The AI agent will gather evidence from the graph, ML model, and case history.
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
