import type { Config } from 'tailwindcss'

const config: Config = {
  content: [
    './pages/**/*.{js,ts,jsx,tsx,mdx}',
    './components/**/*.{js,ts,jsx,tsx,mdx}',
    './app/**/*.{js,ts,jsx,tsx,mdx}',
  ],
  theme: {
    extend: {
      colors: {
        ink:        '#14161C',
        surface:    '#1D2029',
        'surface-2':'#252836',
        paper:      '#E7E2D6',
        brass:      '#C88A3B',
        alert:      '#B14A3D',
        confirm:    '#6B9080',
        caution:    '#C4A24C',
        rule:       '#2A2D38',
        muted:      '#5A5D6B',
        'text-base':'#C8C4BC',
        'text-strong':'#E2DDD5',
      },
      fontFamily: {
        serif:  ['Fraunces', 'Georgia', 'serif'],
        sans:   ['Inter', 'system-ui', 'sans-serif'],
        mono:   ['JetBrains Mono', 'Courier New', 'monospace'],
      },
    },
  },
  plugins: [],
}

export default config
