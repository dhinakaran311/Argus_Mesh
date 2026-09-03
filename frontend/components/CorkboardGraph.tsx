'use client'
import { useMemo } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
  type NodeProps,
  Handle,
  Position,
  type EdgeProps,
  getBezierPath,
  useNodesState,
  useEdgesState,
} from 'reactflow'
import 'reactflow/dist/style.css'

// ── Custom customer node ─────────────────────────────────────────────────────
function CustomerNode({ data }: NodeProps) {
  const risk = (data.risk_score as number) ?? 0
  const border =
    risk >= 0.8 ? 'var(--alert)' :
    risk >= 0.5 ? 'var(--caution)' :
    'var(--confirm)'

  const lvl = (data.risk_level as string) ?? 'LOW'
  const lvlColor =
    lvl === 'CRITICAL' || lvl === 'HIGH' ? 'var(--alert)' :
    lvl === 'MEDIUM' ? 'var(--caution)' : 'var(--confirm)'

  return (
    <div style={{
      background: 'var(--surface-2)',
      border: `1px solid ${border}`,
      borderTop: `3px solid ${border}`,
      padding: '0.65rem 0.9rem',
      width: 180,
      boxShadow: '0 4px 20px rgba(0,0,0,0.6)',
    }}>
      <Handle type="target" position={Position.Left} style={{ background: '#2A2D38', border: '1px solid #3A3D48', width: 7, height: 7 }} />
      <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.7rem', color: 'var(--text-strong)', marginBottom: 5, wordBreak: 'break-all', lineHeight: 1.3 }}>
        {String(data.id ?? '').replace('…', '')}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <span style={{
          fontSize: '0.6rem', fontWeight: 600, letterSpacing: '0.06em', textTransform: 'uppercase',
          padding: '0.1rem 0.4rem', borderLeft: `2px solid ${lvlColor}`, color: lvlColor,
          background: lvlColor + '20',
        }}>
          {lvl}
        </span>
        <span style={{ fontFamily: 'JetBrains Mono', fontSize: '0.65rem', color: 'var(--muted)' }}>
          {(risk * 100).toFixed(0)}%
        </span>
      </div>
      {data.ring_type && (
        <div style={{ marginTop: 4, fontSize: '0.6rem', color: 'var(--muted)', textTransform: 'uppercase', letterSpacing: '0.04em' }}>
          {String(data.ring_type).replace(/_/g, ' ')}
        </div>
      )}
      <Handle type="source" position={Position.Right} style={{ background: '#2A2D38', border: '1px solid #3A3D48', width: 7, height: 7 }} />
    </div>
  )
}

// ── String edge ───────────────────────────────────────────────────────────────
function StringEdge({ id, sourceX, sourceY, targetX, targetY, data }: EdgeProps) {
  const [path] = getBezierPath({ sourceX, sourceY, targetX, targetY, curvature: 0.15 })
  const edgeType = (data as any)?.type as string | undefined
  const color =
    edgeType === 'device' ? '#C88A3B' :
    edgeType === 'ip'     ? '#6B9080' : '#5A5D6B'
  const dash = edgeType === 'ip' ? '5 4' : undefined
  return (
    <path id={id} d={path} stroke={color} strokeWidth={1.5} fill="none" strokeDasharray={dash} opacity={0.7} />
  )
}

// ── Spread layout helper ──────────────────────────────────────────────────────
function spreadLayout(rawNodes: any[]): Array<{ id: string; position: { x: number; y: number } }> {
  const n = rawNodes.length
  if (n === 0) return []

  // Use a spiral/sunflower pattern for better spacing
  const result = []
  for (let i = 0; i < n; i++) {
    const angle  = i * 2.4  // golden angle ≈ 137.5°
    const radius = 60 + 70 * Math.sqrt(i)
    result.push({
      id:       rawNodes[i].id,
      position: {
        x: Math.round(600 + radius * Math.cos(angle)),
        y: Math.round(400 + radius * Math.sin(angle)),
      },
    })
  }
  return result
}

// ── Props ─────────────────────────────────────────────────────────────────────
interface Props {
  flowData: { nodes: any[]; edges: any[] } | null
}

export default function CorkboardGraph({ flowData }: Props) {
  // Memoize nodeTypes/edgeTypes to prevent React Flow warning #002
  const nodeTypes = useMemo(() => ({ customer: CustomerNode }), [])
  const edgeTypes = useMemo(() => ({ string: StringEdge }), [])

  const positions = useMemo(
    () => spreadLayout(flowData?.nodes ?? []),
    [flowData?.nodes]
  )

  const posMap = useMemo(() => {
    const m: Record<string, { x: number; y: number }> = {}
    positions.forEach(p => { m[p.id] = p.position })
    return m
  }, [positions])

  const initNodes: Node[] = useMemo(
    () => (flowData?.nodes ?? []).map((n) => ({
      id:       n.id,
      type:     'customer',
      position: posMap[n.id] ?? n.position ?? { x: 0, y: 0 },
      data:     n.data ?? n,
    })),
    [flowData?.nodes, posMap]
  )

  const initEdges: Edge[] = useMemo(
    () => (flowData?.edges ?? []).map((e) => ({
      id:     e.id,
      source: e.source,
      target: e.target,
      type:   'string',
      data:   e.data ?? {},
    })),
    [flowData?.edges]
  )

  const [nodes, , onNodesChange] = useNodesState(initNodes)
  const [edges, , onEdgesChange] = useEdgesState(initEdges)

  return (
    <ReactFlow
      nodes={nodes}
      edges={edges}
      onNodesChange={onNodesChange}
      onEdgesChange={onEdgesChange}
      nodeTypes={nodeTypes}
      edgeTypes={edgeTypes}
      fitView
      fitViewOptions={{ padding: 0.15, maxZoom: 0.9 }}
      minZoom={0.1}
      maxZoom={2}
      className="corkboard"
      proOptions={{ hideAttribution: true }}
    >
      <Background color="#2A2D38" gap={32} size={1} />
      <Controls
        style={{ background: 'var(--surface)', border: '1px solid var(--rule)', boxShadow: 'none' }}
        showInteractive={false}
      />
      <MiniMap
        style={{ background: 'var(--surface)', border: '1px solid var(--rule)' }}
        nodeColor={(n) => {
          const r = (n.data?.risk_score as number) ?? 0
          return r >= 0.8 ? '#B14A3D' : r >= 0.5 ? '#C4A24C' : '#6B9080'
        }}
        nodeStrokeWidth={0}
      />
    </ReactFlow>
  )
}
