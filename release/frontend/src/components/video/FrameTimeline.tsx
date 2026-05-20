import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ResponsiveContainer,
} from 'recharts'
import type { FrameResult } from '@/types/detection'

interface FrameTimelineProps {
  frames: FrameResult[]
  temporalConsistency: number
}

interface TooltipPayload {
  value: number
  payload: FrameResult & { fake_pct: number }
}

function CustomTooltip({ active, payload }: { active?: boolean; payload?: TooltipPayload[] }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  return (
    <div className="bg-gray-900 border border-gray-700 rounded-lg px-3 py-2 text-xs shadow-xl">
      <p className="text-gray-400">Frame {d.frame_index} · {d.timestamp_sec.toFixed(2)}s</p>
      <p className={d.fake_pct >= 50 ? 'text-red-400 font-bold' : 'text-green-400 font-bold'}>
        {d.fake_pct}% fake probability
      </p>
      <p className="text-gray-500">{d.face_detected ? '👤 Face detected' : '🖼 No face'}</p>
    </div>
  )
}

export function FrameTimeline({ frames, temporalConsistency }: FrameTimelineProps) {
  const data = frames.map((f) => ({
    ...f,
    fake_pct: Math.round(f.final_score * 100),
  }))

  const consistencyLabel =
    temporalConsistency < 0.08 ? 'Consistent' :
    temporalConsistency < 0.20 ? 'Moderate' : 'Variable'

  const consistencyColor =
    temporalConsistency < 0.08 ? 'text-yellow-400' :
    temporalConsistency < 0.20 ? 'text-blue-400' : 'text-orange-400'

  return (
    <div className="card space-y-4">
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">Frame-by-Frame Analysis</h3>
        <span className={`text-xs font-medium ${consistencyColor}`}>
          {consistencyLabel} · σ={temporalConsistency.toFixed(3)}
        </span>
      </div>

      <ResponsiveContainer width="100%" height={200}>
        <LineChart data={data} margin={{ top: 5, right: 10, left: -20, bottom: 5 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="#1f2937" />
          <XAxis
            dataKey="timestamp_sec"
            tickFormatter={(v: number) => `${v.toFixed(1)}s`}
            tick={{ fill: '#6b7280', fontSize: 10 }}
            axisLine={{ stroke: '#374151' }}
            tickLine={false}
          />
          <YAxis
            domain={[0, 100]}
            tickFormatter={(v: number) => `${v}%`}
            tick={{ fill: '#6b7280', fontSize: 10 }}
            axisLine={{ stroke: '#374151' }}
            tickLine={false}
          />
          <Tooltip content={<CustomTooltip />} />
          <ReferenceLine y={50} stroke="#ef4444" strokeDasharray="4 4" strokeOpacity={0.5} />
          <Line
            type="monotone"
            dataKey="fake_pct"
            stroke="#f97316"
            strokeWidth={2}
            dot={(props) => {
              const { cx, cy, payload } = props as { cx: number; cy: number; payload: { face_detected: boolean; fake_pct: number } }
              return (
                <circle
                  key={`dot-${cx}-${cy}`}
                  cx={cx}
                  cy={cy}
                  r={payload.face_detected ? 4 : 3}
                  fill={payload.fake_pct >= 50 ? '#ef4444' : '#22c55e'}
                  stroke={payload.face_detected ? '#f97316' : 'transparent'}
                  strokeWidth={1.5}
                />
              )
            }}
            activeDot={{ r: 6, fill: '#f97316' }}
          />
        </LineChart>
      </ResponsiveContainer>

      <div className="flex items-center gap-4 text-xs text-gray-500">
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-red-500 inline-block" />
          Fake (≥50%)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full bg-green-500 inline-block" />
          Real (&lt;50%)
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-2 h-2 rounded-full border border-orange-400 inline-block" />
          Face detected
        </span>
        <span className="ml-auto">{frames.length} frames sampled</span>
      </div>
    </div>
  )
}
