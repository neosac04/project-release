import type { Config } from 'tailwindcss'

export default {
  content: ['./index.html', './src/**/*.{js,ts,jsx,tsx}'],
  theme: {
    extend: {
      colors: {
        brand: {
          50:  '#f0f4ff',
          100: '#e0eaff',
          500: '#6366f1',
          600: '#4f46e5',
          700: '#4338ca',
          900: '#1e1b4b',
        },
        fake: { DEFAULT: '#ef4444', light: '#fee2e2' },
        real: { DEFAULT: '#22c55e', light: '#dcfce7' },
        uncertain: { DEFAULT: '#f59e0b', light: '#fef3c7' },
      },
      animation: {
        'spin-slow': 'spin 3s linear infinite',
        'pulse-soft': 'pulse 2s ease-in-out infinite',
      },
    },
  },
  plugins: [],
} satisfies Config
