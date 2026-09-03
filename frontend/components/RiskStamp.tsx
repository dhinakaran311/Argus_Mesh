import type { RiskLevel } from '@/lib/api'

interface Props {
  level: RiskLevel | string
}

const MAP: Record<string, string> = {
  LOW:      'low',
  MEDIUM:   'medium',
  HIGH:     'high',
  CRITICAL: 'critical',
}

export default function RiskStamp({ level }: Props) {
  const cls = MAP[String(level).toUpperCase()] ?? 'low'
  return <span className={`risk-stamp ${cls}`}>{level}</span>
}
