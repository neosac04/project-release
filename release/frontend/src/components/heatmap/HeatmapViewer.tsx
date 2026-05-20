import { useEffect, useRef, useState } from 'react'
import { getHeatmapUrl } from '@/api/detection'
import { Spinner } from '@/components/shared/Spinner'
import clsx from 'clsx'

interface HeatmapModel {
  key: string
  label: string
  caption: string
}

const MODELS: HeatmapModel[] = [
  { key: 'ensemble',     label: 'Ensemble',        caption: 'Weighted average of all model heatmaps' },
  { key: 'vit',          label: 'ViT',             caption: 'Attention map from the global image transformer' },
  { key: 'f3net',        label: 'F3Net',           caption: 'GradCAM on the frequency-domain features' },
  { key: 'efficientnet', label: 'EfficientNet',    caption: 'GradCAM++ on facial texture features' },
]

interface HeatmapViewerProps {
  resultId: string
  originalUrl: string
  availableModels: string[]   // model_votes keys + ensemble
}

export function HeatmapViewer({ resultId, originalUrl, availableModels }: HeatmapViewerProps) {
  const tabs = MODELS.filter(
    (m) => m.key === 'ensemble' || availableModels.includes(m.key),
  )
  const [activeModel, setActiveModel] = useState(tabs[0]?.key ?? 'ensemble')
  const [opacity, setOpacity] = useState(0.55)
  const [heatmapUrl, setHeatmapUrl] = useState<string | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const canvasRef = useRef<HTMLCanvasElement>(null)

  useEffect(() => {
    setLoading(true)
    setError(null)
    setHeatmapUrl(null)
    const url = getHeatmapUrl(resultId, activeModel)
    const img = new Image()
    img.onload = () => { setHeatmapUrl(url); setLoading(false) }
    img.onerror = () => {
      setError(`Heatmap not available for ${activeModel}`)
      setLoading(false)
    }
    img.src = url
  }, [resultId, activeModel])

  useEffect(() => {
    const canvas = canvasRef.current
    if (!canvas) return
    const ctx = canvas.getContext('2d')
    if (!ctx) return

    const orig = new Image()
    orig.crossOrigin = 'anonymous'
    orig.onload = () => {
      canvas.width = orig.naturalWidth
      canvas.height = orig.naturalHeight
      ctx.drawImage(orig, 0, 0)

      if (heatmapUrl) {
        const hm = new Image()
        hm.crossOrigin = 'anonymous'
        hm.onload = () => {
          ctx.globalAlpha = opacity
          ctx.drawImage(hm, 0, 0, canvas.width, canvas.height)
          ctx.globalAlpha = 1
        }
        hm.src = heatmapUrl
      }
    }
    orig.src = originalUrl
  }, [originalUrl, heatmapUrl, opacity])

  const activeMeta = tabs.find((m) => m.key === activeModel)

  return (
    <div className="card">
      <h3 className="section-title">Why It Was Classified That Way</h3>
      <p className="text-xs text-gray-500 mb-4">
        These heatmaps show the image regions each model focused on. Red = areas the model
        considers suspicious; blue = areas it considers normal.
      </p>

      {/* Model tabs */}
      <div className="flex gap-2 mb-3 flex-wrap">
        {tabs.map((m) => (
          <button
            key={m.key}
            onClick={() => setActiveModel(m.key)}
            className={clsx(
              'px-3 py-1.5 rounded-lg text-sm font-medium transition-colors',
              activeModel === m.key
                ? 'bg-brand-600 text-white'
                : 'bg-gray-800 text-gray-400 hover:bg-gray-700',
            )}
          >
            {m.label}
          </button>
        ))}
      </div>

      {activeMeta && (
        <p className="text-xs text-gray-500 mb-3 italic">{activeMeta.caption}</p>
      )}

      {/* Opacity slider */}
      <div className="flex items-center gap-3 mb-4">
        <span className="text-xs text-gray-500 w-16">Overlay</span>
        <input
          type="range" min={0} max={1} step={0.05}
          value={opacity}
          onChange={(e) => setOpacity(Number(e.target.value))}
          className="flex-1 accent-brand-500"
        />
        <span className="text-xs text-gray-400 w-10 text-right">{Math.round(opacity * 100)}%</span>
      </div>

      {/* Canvas */}
      <div className="relative rounded-xl overflow-hidden bg-gray-800 min-h-48 flex items-center justify-center">
        {loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/60 z-10">
            <Spinner className="w-8 h-8" />
          </div>
        )}
        {error && !loading && (
          <div className="absolute inset-0 flex items-center justify-center bg-gray-900/80 z-10">
            <p className="text-xs text-gray-500">{error}</p>
          </div>
        )}
        <canvas ref={canvasRef} className="w-full h-auto rounded-xl" />
      </div>

      {/* Legend */}
      <div className="flex items-center gap-4 mt-3 justify-center flex-wrap">
        {[
          { color: 'bg-blue-500',  label: 'Authentic' },
          { color: 'bg-cyan-400',  label: 'Neutral' },
          { color: 'bg-yellow-400', label: 'Suspicious' },
          { color: 'bg-red-500',   label: 'High suspicion' },
        ].map(({ color, label }) => (
          <div key={label} className="flex items-center gap-1.5">
            <div className={`w-3 h-3 rounded-sm ${color}`} />
            <span className="text-xs text-gray-500">{label}</span>
          </div>
        ))}
      </div>
    </div>
  )
}
