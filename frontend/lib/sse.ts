// lib/sse.ts — EventSource wrapper for the investigation SSE stream
export interface SSEStep {
  type: 'step' | 'report' | 'error' | 'done'
  step?: number
  label?: string
  status?: 'running' | 'done' | 'error'
  report?: string
  verdict?: string
  risk_level?: string
  error?: string
}

type StepCallback = (step: SSEStep) => void

export function streamInvestigation(
  apiBase: string,
  customerId: string,
  onStep: StepCallback,
  onDone: () => void,
  onError: (err: string) => void
): () => void {
  // POST first to trigger the investigation, then open SSE
  const ctrl = new AbortController()

  fetch(`${apiBase}/api/investigate`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ customer_id: customerId }),
    signal: ctrl.signal,
  }).then(async (res) => {
    if (!res.ok) {
      onError(`Server error ${res.status}`)
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
            const event = JSON.parse(raw) as SSEStep
            onStep(event)
          } catch {
            // skip malformed lines
          }
        }
      }
    }
  }).catch((err) => {
    if (err.name !== 'AbortError') onError(String(err))
  })

  return () => ctrl.abort()
}
