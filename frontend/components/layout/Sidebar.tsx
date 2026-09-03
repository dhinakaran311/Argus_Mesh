'use client'
import Link from 'next/link'
import { usePathname } from 'next/navigation'

const NAV = [
  { href: '/',              label: 'Dashboard' },
  { href: '/rings',         label: 'Rings' },
  { href: '/graph',         label: 'Graph' },
  { href: '/investigate',   label: 'Investigate' },
  { href: '/model',         label: 'Model' },
  { href: '/transactions',  label: 'Transactions' },
]

export default function Sidebar() {
  const pathname = usePathname()

  return (
    <aside className="sidebar">
      <div className="sidebar-brand">
        <div className="sidebar-brand-title">AbuseRing<br />Sentinel</div>
        <div className="sidebar-brand-sub">Razorpay fraud intelligence</div>
      </div>

      <nav className="sidebar-nav" aria-label="Main navigation">
        {NAV.map(({ href, label }) => {
          const active = href === '/' ? pathname === '/' : pathname.startsWith(href)
          return (
            <Link
              key={href}
              href={href}
              className={`nav-link${active ? ' active' : ''}`}
              aria-current={active ? 'page' : undefined}
            >
              {label}
            </Link>
          )
        })}
      </nav>

      <div style={{ marginTop: 'auto', padding: '1rem 1.5rem' }}>
        <div className="section-label">Backend</div>
        <div style={{ fontSize: '0.7rem', color: 'var(--muted)' }}>
          localhost:8000
        </div>
      </div>
    </aside>
  )
}
