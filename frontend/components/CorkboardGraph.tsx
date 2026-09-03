'use client'
import { useCallback, useEffect, useState } from 'react'
import ReactFlow, {
  Background,
  Controls,
  MiniMap,
  type Node,
  type Edge,
  type NodeProps,
  Handle,
  Position,
  EdgeProps,
  getBezierPath,
  useNodesState,
  useEdgesState,
} from 'reactflow'
import 'reactflow/dist/style.css'
import RiskStamp from '@/components/RiskStamp'

// ── Custom customer node (pinned card on corkboard) ──────────────────────
function CustomerNode({ data }: NodeProps) {
  const risk = data.risk_score ?? 0
  const border = risk >= 0.8 ? 'var(--alert)' : risk >= 0.5 ? 'var(--caution)' : 'var(--confirm)'
  return (
    <div style={{
      background: 'var(--surface)',
      border: `1px solid ${border}`,
      borderTop: `3px solid ${border}`,
      padding: '0.6rem 0.8rem',
      width: 160,
      fontSize: '0.72rem',
      boxShadow: '0 4px 16px rgba(0,0,0,0.5)',
    }}>
      <Handle type="target" position={Position.Left} style={{ background: 'var(--rule)', border: 'none', width: 6, height: 6 }} />
      <div style={{ fontFamily: 'JetBrains Mono', fontSize: '0.72rem', color: 'var(--text-strong)', marginBottom: 4, wordBreak: 'break-all' }}>
        {data.id}
      </div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
        <RiskStamp level={data.risk_level ?? 'LOW'} />
        <span style={{ fontFamily: 'JetBrains Mono', fontSize: '0.68rem', color: 'var(--muted)' }}>
          {(risk * 100).toFixed(0)}%
        </span>
      </div>
      {data.ring_type && (
        <div style={{ marginTop: 4, fontSize: '0.65rem', color: 'var(--muted)' }}>
          {String(data.ring_type).replace(/_/g, ' ')}
        </div>
      )}
      <Handle type="source" position={Position.Right} style={{ background: 'var(--rule)', border: 'none', width: 6, height: 6 }} />
    </div>
  )
}

// ── String edge (taut thread, not bezier SaaS curve) ─────────────────────
function StringEdge({ id, sourceX, sourceY, targetX, targetY, data }: EdgeProps) {
  const [path] = getBezierPath({ sourceX, sourceY, targetX, targetY, curvature: 0.1 })
  const color = data?.type === 'device' ? '#C88A3B' : data?.type === 'ip' ? '#6B9080' : '#5A5D6B'
  return (
    <>
      <path id={id} d={path} stroke={color} strokeWidth={1.5} fill="none" strokeDasharray={data?.type === 'ip' ? '4 3' : undefined} opacity={0.7} />
    </>
  )
}

const nodeTypes = { customer: CustomerNode }
const edgeTypes = { string: StringEdge }

interface Props {
  flowData: { nodes: any[]; edges: any[] } | null
}

export default function CorkboardGraph({ flowData }: Props) {
  const initNodes: Node[] = (flowData?.nodes ?? []).map((n) => ({
    id: n.id,
    type: 'customer',
    position: n.position ?? { x: Math.random() * 600, y: Math.random() * 400 },
    data: n.data ?? n,
  }))
  const initEdges: Edge[] = (flowData?.edges ?? []).map((e) => ({
    id: e.id,
    source: e.source,
    target: e.target,
    type: 'string',
    data: e.data ?? {},
  }))

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
      fitViewOptions={{ padding: 0.2 }}
      className="corkboard"
      proOptions={{ hideAttribution: true }}
    >
      <Background color="#2A2D38" gap={24} size={1} />
      <Controls style={{ background: 'var(--surface)', border: '1px solid var(--rule)', boxShadow: 'none' }} />
      <MiniMap
        style={{ background: 'var(--surface)', border: '1px solid var(--rule)' }}
        nodeColor={(n) => {
          const r = n.data?.risk_score ?? 0
          return r >= 0.8 ? '#B14A3D' : r >= 0.5 ? '#C4A24C' : '#6B9080'
        }}
      />
    </ReactFlow>
  )
}
