// lib/api.ts — typed fetch wrappers for all backend endpoints
// ALWAYS use relative paths so Next.js proxy handles CORS for client-side fetches
const isServer = typeof window === 'undefined'
const BASE = isServer
  ? (process.env.NEXT_PUBLIC_API_URL ?? 'http://localhost:8000')
  : '' // client-side: use relative path through Next.js proxy rewrite

/** Fetch with a configurable timeout + optional ISR revalidation for server-side calls. */
async function fetchWithTimeout<T>(
  url: string,
  opts: RequestInit = {},
  timeoutMs = 30_000,
): Promise<T> {
  const controller = new AbortController()
  const tid = setTimeout(() => controller.abort(), timeoutMs)
  try {
    const res = await fetch(url, { ...opts, signal: controller.signal })
    clearTimeout(tid)
    if (!res.ok) throw new Error(`GET ${url} → ${res.status}`)
    return res.json() as Promise<T>
  } catch (err: any) {
    clearTimeout(tid)
    if (err?.name === 'AbortError') throw new Error(`GET ${url} timed out after ${timeoutMs / 1000}s`)
    throw err
  }
}

async function get<T>(path: string, cached = false): Promise<T> {
  const url = `${BASE}${path}`
  if (isServer && cached) {
    return fetchWithTimeout<T>(
      url,
      { next: { revalidate: 30 } } as RequestInit,
      10_000,
    )
  }
  return fetchWithTimeout<T>(url, { cache: 'no-store' })
}

async function post<T>(path: string, body: unknown): Promise<T> {
  const res = await fetch(`${BASE}${path}`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  })
  if (!res.ok) throw new Error(`POST ${path} → ${res.status}`)
  return res.json() as Promise<T>
}

/* ── Types ────────────────────────────────────────────────────────────── */
export type RiskLevel = 'LOW' | 'MEDIUM' | 'HIGH' | 'CRITICAL'

export interface HealthResponse {
  status: 'ok' | 'degraded'
  timestamp: string
  services: { neo4j: string; qdrant: string; ml_model: string; data: string }
}

export interface DashboardStats {
  total_customers: number
  abuse_customers: number
  abuse_rate_pct: number
  critical_count: number
  high_count: number
  medium_count: number
  low_count: number
  avg_risk_score: number
  total_rings: number
  total_devices: number
  total_ips: number
}

export interface ClusterSummary {
  cluster_id: string
  cluster_size: number
  avg_ml_score: number
  graph_score: number
  combined_risk_score: number
  risk_level: RiskLevel
  ring_type?: string
  member_ids: string[]
  abuse_count: number
}

export interface Transaction {
  transaction_id: string
  customer_id: string
  amount: number
  status: string
  risk_score?: number
  risk_level?: RiskLevel
  created_at?: string
  merchant_id?: string
}

export interface ReactFlowData {
  nodes: Array<{
    id: string
    type?: string
    position: { x: number; y: number }
    data: Record<string, unknown>
  }>
  edges: Array<{
    id: string
    source: string
    target: string
    type?: string
    data?: Record<string, unknown>
  }>
}

/* ── Response unwrappers (match actual backend shapes) ── */
async function getTransactions(limit: number, offset: number): Promise<Transaction[]> {
  const res = await get<{ transactions: Transaction[] } | Transaction[]>(
    `/api/transactions?limit=${limit}&offset=${offset}`
  )
  if (Array.isArray(res)) return res
  return (res as any).transactions ?? []
}

async function getClusters(limit: number): Promise<ClusterSummary[]> {
  try {
    const res = await get<ClusterSummary[] | { clusters: ClusterSummary[] }>(
      `/api/clusters?limit=${limit}`, true  // cached: 30 s revalidate + 10 s timeout
    )
    if (Array.isArray(res)) return res
    return (res as any).clusters ?? []
  } catch { return [] }
}

async function getGraph(clusterId: string): Promise<ReactFlowData | null> {
  try {
    const res = await get<any>(`/api/graph/${encodeURIComponent(clusterId)}`)
    if (res?.nodes && res?.edges) return res
    if (res?.react_flow_graph) return res.react_flow_graph
    return null
  } catch { return null }
}

export const api = {
  health:        () => get<HealthResponse>('/api/health'),
  dashboard:     () => get<DashboardStats>('/api/dashboard', true),  // cached, heavy Neo4j query
  clusters:      (limit = 50) => getClusters(limit),
  cluster:       (id: string) => get<ClusterSummary>(`/api/clusters/${encodeURIComponent(id)}`, true),
  graph:         (clusterId: string) => getGraph(clusterId),
  transactions:  (limit = 100, offset = 0) => getTransactions(limit, offset),
  modelMetrics:  () => get<any>('/api/model/metrics', true),
  modelFeatures: () =>
    get<{ features: Array<{ feature: string; importance: number }> }>('/api/model/features', true)
      .then(r => r.features ?? [])
      .catch(() => [] as any[]),
  // investigate takes cluster_id (RING-001 etc), not customer_id
  investigate:   (clusterId: string) =>
    post<unknown>('/api/investigate', { cluster_id: clusterId }),
}
