import { Sparkles, AlertTriangle } from 'lucide-react'
import { Card } from '@/components/shared/Card'

interface ExplanationsCardProps {
  explanations: string[]
}

export function ExplanationsCard({ explanations }: ExplanationsCardProps) {
  if (!explanations || explanations.length === 0) {
    return (
      <Card title="Explanation">
        <p className="text-sm text-gray-500">No explanation available.</p>
      </Card>
    )
  }

  const [headline, ...details] = explanations

  return (
    <Card title="Explanation">
      <p className="text-xs text-gray-500 mb-4">
        Plain-language summary of what each detector saw.
      </p>

      {/* Headline finding */}
      <div className="flex gap-3 mb-4 p-3 rounded-xl bg-brand-600/10 border border-brand-600/30">
        <Sparkles size={18} className="text-brand-400 flex-shrink-0 mt-0.5" />
        <p className="text-sm text-gray-100 leading-relaxed">{headline}</p>
      </div>

      {/* Detailed findings */}
      {details.length > 0 && (
        <ul className="space-y-2.5">
          {details.map((text, i) => {
            const isWarning = text.startsWith('⚠')
            const clean = isWarning ? text.replace(/^⚠\s*/, '') : text
            return (
              <li
                key={i}
                className="flex gap-2.5 text-sm text-gray-300 leading-relaxed"
              >
                {isWarning ? (
                  <AlertTriangle size={14} className="text-uncertain flex-shrink-0 mt-1" />
                ) : (
                  <span className="w-1.5 h-1.5 rounded-full bg-gray-500 flex-shrink-0 mt-2" />
                )}
                <span className={isWarning ? 'text-uncertain' : ''}>{clean}</span>
              </li>
            )
          })}
        </ul>
      )}
    </Card>
  )
}
