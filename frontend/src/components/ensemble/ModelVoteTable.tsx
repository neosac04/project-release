import type { ModelVote } from '@/types/detection'
import clsx from 'clsx'
import { Card } from '@/components/shared/Card'
import {
  BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell,
} from 'recharts'

const MODEL_LABELS: Record<string, string> = {
  vit:          'ViT (Full image)',
  f3net:        'F3Net (Frequency)',
  efficientnet: 'EfficientNet (Face)',
  xceptionnet:  'XceptionNet',
  siglip:       'SigLIP (Vision-Language)',
}

// Shorter labels used on the bar chart where space is limited
const MODEL_CHART_LABELS: Record<string, string> = {
  vit:          'ViT',
  f3net:        'F3Net',
  efficientnet: 'EffNet',
  xceptionnet:  'Xcept',
  siglip:       'SigLIP',
}

const MODEL_DESCRIPTIONS: Record<string, string> = {
  vit:          'Vision Transformer — looks at the whole image for learned synthesis patterns',
  f3net:        'Decomposes the image into 4 frequency bands via DCT — catches GAN/diffusion artifacts',
  efficientnet: 'Convolutional net on the face crop — detects micro-texture inconsistencies',
  xceptionnet:  'Localised manipulation artifact detector',
  siglip:       'Fine-tuned SigLIP classifier (94.44% accuracy) — vision-language features on the face crop',
}

interface ModelVoteTableProps {
  votes: Record<string, ModelVote>
  weights: Record<string, number>
}

export function ModelVoteTable({ votes, weights }: ModelVoteTableProps) {
  // Sort: highest fusion weight first
  const entries = Object.entries(votes).sort(
    ([a], [b]) => (weights[b] ?? 0) - (weights[a] ?? 0),
  )

  const chartData = entries.map(([name, v]) => ({
    name: MODEL_CHART_LABELS[name] ?? name,
    fake: Math.round(v.fake_prob * 100),
  }))

  return (
    <Card title="Individual Model Predictions">
      <p className="text-xs text-gray-500 mb-4">
        Four specialists vote independently. Each one looks for different evidence.
        The verdict is a confidence-weighted blend (weights below).
      </p>

      {/* Bar chart */}
      <div className="h-48 mb-6">
        <ResponsiveContainer width="100%" height="100%">
          <BarChart data={chartData} margin={{ top: 4, right: 8, left: -20, bottom: 32 }}>
            <XAxis
              dataKey="name"
              tick={{ fill: '#9ca3af', fontSize: 11 }}
              interval={0}
              angle={-25}
              textAnchor="end"
              height={48}
            />
            <YAxis tick={{ fill: '#9ca3af', fontSize: 10 }} domain={[0, 100]} unit="%" />
            <Tooltip
              contentStyle={{ background: '#111827', border: '1px solid #374151', borderRadius: 8 }}
              labelStyle={{ color: '#e5e7eb' }}
              formatter={(v: number) => [`${v}% fake`]}
            />
            <Bar dataKey="fake" radius={[4, 4, 0, 0]}>
              {chartData.map((d) => (
                <Cell
                  key={d.name}
                  fill={d.fake >= 65 ? '#ef4444' : d.fake <= 35 ? '#22c55e' : '#f59e0b'}
                />
              ))}
            </Bar>
          </BarChart>
        </ResponsiveContainer>
      </div>

      {/* Detail rows */}
      <div className="space-y-3">
        {entries.map(([name, v]) => {
          const fake = Math.round(v.fake_prob * 100)
          const weight = Math.round((weights[name] ?? 0) * 100)
          const verdict = fake >= 65 ? 'fake' : fake <= 35 ? 'real' : 'uncertain'
          const verdictColor =
            verdict === 'fake' ? 'text-fake' : verdict === 'real' ? 'text-real' : 'text-uncertain'
          const verdictBg =
            verdict === 'fake' ? 'bg-fake' : verdict === 'real' ? 'bg-real' : 'bg-uncertain'

          return (
            <div
              key={name}
              className="rounded-xl border border-gray-800 bg-gray-900/40 p-3"
            >
              <div className="flex items-start justify-between gap-3 mb-2">
                <div className="flex-1 min-w-0">
                  <p className="font-semibold text-gray-200 text-sm">
                    {MODEL_LABELS[name] ?? name}
                  </p>
                  <p className="text-xs text-gray-500 mt-0.5">
                    {MODEL_DESCRIPTIONS[name] ?? ''}
                  </p>
                </div>
                <div className="text-right flex-shrink-0">
                  <p className={clsx('text-xl font-bold leading-none', verdictColor)}>
                    {fake}%
                  </p>
                  <p className="text-[10px] text-gray-500 uppercase mt-0.5">fake</p>
                </div>
              </div>

              {/* Probability bar */}
              <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden mb-2">
                <div
                  className={clsx('h-full transition-all', verdictBg)}
                  style={{ width: `${fake}%` }}
                />
              </div>

              <div className="flex items-center justify-between text-[11px] text-gray-500">
                <span>Vote weight in fusion: <span className="text-gray-300 font-medium">{weight}%</span></span>
                <span>{Math.round(v.inference_time_ms)} ms</span>
              </div>
            </div>
          )
        })}
      </div>
    </Card>
  )
}
