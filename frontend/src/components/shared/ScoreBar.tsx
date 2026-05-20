import clsx from 'clsx'

interface ScoreBarProps {
  label: string
  value: number      // 0-1
  invert?: boolean   // if true, high value = bad (red)
  tooltip?: string
}

export function ScoreBar({ label, value, invert = false, tooltip }: ScoreBarProps) {
  const pct = Math.round(value * 100)
  const color = invert
    ? value > 0.6 ? 'bg-fake' : value > 0.35 ? 'bg-uncertain' : 'bg-real'
    : value > 0.6 ? 'bg-real' : value > 0.35 ? 'bg-uncertain' : 'bg-fake'

  return (
    <div className="mb-3" title={tooltip}>
      <div className="flex justify-between items-center mb-1">
        <span className="text-sm text-gray-400">{label}</span>
        <span className={clsx('text-sm font-semibold', invert
          ? (value > 0.6 ? 'text-fake' : value > 0.35 ? 'text-uncertain' : 'text-real')
          : (value > 0.6 ? 'text-real' : value > 0.35 ? 'text-uncertain' : 'text-fake')
        )}>
          {pct}%
        </span>
      </div>
      <div className="h-2 bg-gray-800 rounded-full overflow-hidden">
        <div
          className={clsx('h-full rounded-full transition-all duration-700', color)}
          style={{ width: `${pct}%` }}
        />
      </div>
    </div>
  )
}
