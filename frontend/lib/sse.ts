// lib/sse.ts
// Canonical SSE client for AbuseRing Sentinel investigation stream.
// Exported from here; imported by investigate/page.tsx.

export type StepStatus = 'pending' | 'active' | 'done'

export interface InvestigationStep {
  step: string    // starting | facts | graph | ml | rag | reasoning | complete | error
  message: string
  data?: unknown
}

/**
 * Stream an investigation via SSE.
 *
 * @param clusterId  Ring cluster ID (e.g. RING-001 or a device-based ID like D0001)
 * @param onStep     Called for every intermediate step event
 * @param onReport   Called once with the final report data when step === 'complete'
 * @param onError    Called on stream error
 * @param onDone     Called when the stream closes cleanly
 * @returns          Cancellation function
 */
export function streamInvestigation(
  clusterId: string,
  onStep: (step: string, message: string) => void,
  onReport: (data: unknown) => void,
  onError: (err: string) => void,
  onDone: () => void,
): () => void {
  const ctrl = new AbortController()

  const headers: Record<string, string> = { 'Content-Type': 'application/json' }
  if (process.env.NEXT_PUBLIC_API_KEY) {
    headers['X-API-Key'] = process.env.NEXT_PUBLIC_API_KEY
  }

  fetch('/api/investigate', {
    method: 'POST',
    headers,
    // #10: use cluster_id (not customer_id — that was the wrong field in the old sse.ts)
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
            const event = JSON.parse(raw) as InvestigationStep
            const step  = event.step ?? ''
            const msg   = event.message ?? step
            if (step === 'complete' || step === 'done') {
              onReport(event.data ?? event)
              onDone()
            } else if (step === 'error') {
              onError(event.message ?? 'Investigation failed')
            } else {
              onStep(step, msg)
            }
          } catch { /* skip malformed lines */ }
        }
      }
    }
  }).catch((err) => {
    if (err.name !== 'AbortError') onError(String(err))
  })

  return () => ctrl.abort()
}
