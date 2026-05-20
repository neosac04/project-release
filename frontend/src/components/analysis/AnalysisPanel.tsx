import type { AnalysisMetrics } from '@/types/detection'

interface AnalysisPanelProps {
  analysis: AnalysisMetrics
  finalScore: number
  faceDetected: boolean
}

// ── Helpers ───────────────────────────────────────────────────────────────────

function MetricBar({
  label, value, color = '#58a6ff', invert = false,
}: {
  label: string
  value: number
  color?: string
  invert?: boolean
}) {
  const pct = Math.round(value * 100)
  const displayColor = invert
    ? value < 0.3 ? '#ef4444' : value < 0.6 ? '#f59e0b' : '#22c55e'
    : value > 0.7 ? '#ef4444' : value > 0.4 ? '#f59e0b' : color

  return (
    <div>
      <div className="flex justify-between text-[11px] mb-1">
        <span className="text-gray-400">{label}</span>
        <span className="text-gray-300 font-medium tabular-nums">{pct}%</span>
      </div>
      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
        <div
          className="h-full rounded-full transition-all duration-500"
          style={{ width: `${pct}%`, background: displayColor }}
        />
      </div>
    </div>
  )
}

function StatCard({ label, value, sub }: { label: string; value: string | number; sub?: string }) {
  return (
    <div className="bg-gray-900/60 rounded-xl p-3 text-center border border-gray-800">
      <div className="text-lg font-bold text-gray-100 leading-none mb-1">{value}</div>
      <div className="text-[10px] text-gray-500 uppercase tracking-wide">{label}</div>
      {sub && <div className="text-[10px] text-gray-600 mt-0.5">{sub}</div>}
    </div>
  )
}

type IndicatorLevel = 'red' | 'amber' | 'green'

function buildIndicators(
  analysis: AnalysisMetrics,
  finalScore: number,
  faceDetected: boolean,
): { label: string; level: IndicatorLevel }[] {
  const out: { label: string; level: IndicatorLevel }[] = []

  if (finalScore > 0.75)       out.push({ label: 'HIGH FAKE PROBABILITY',  level: 'red'   })
  else if (finalScore > 0.55)  out.push({ label: 'MODERATE FAKE SIGNAL',   level: 'amber' })

  if (!faceDetected)           out.push({ label: 'NO FACE DETECTED',       level: 'amber' })

  if (analysis.symmetry_score !== null && analysis.symmetry_score !== undefined && analysis.symmetry_score < 0.6)
    out.push({ label: 'FACIAL ASYMMETRY DETECTED', level: 'amber' })

  if (analysis.fft.spectral_irregularity > 0.15)
    out.push({ label: 'FREQUENCY ANOMALY',         level: 'red'   })

  if (analysis.texture.noise_level > 0.5)
    out.push({ label: 'UNNATURAL NOISE PATTERN',   level: 'amber' })

  if (analysis.skin && analysis.skin.pore_detail < 0.15 && faceDetected)
    out.push({ label: 'LOW SKIN DETAIL (AI skin?)', level: 'red'  })

  if (out.length === 0)
    out.push({ label: 'NO STRONG ANOMALIES FOUND', level: 'green' })

  return out
}

const INDICATOR_STYLES: Record<IndicatorLevel, string> = {
  red:   'bg-red-500/10   border-red-500/40   text-red-400',
  amber: 'bg-yellow-500/10 border-yellow-500/40 text-yellow-400',
  green: 'bg-green-500/10 border-green-500/40 text-green-400',
}

// ── Main component ────────────────────────────────────────────────────────────

export function AnalysisPanel({ analysis, finalScore, faceDetected }: AnalysisPanelProps) {
  const indicators = buildIndicators(analysis, finalScore, faceDetected)

  const topRegions = analysis.top_attention_regions.slice(0, 5)
  const hasRegions = topRegions.length > 0 && Object.keys(analysis.region_scores).length > 0

  return (
    <div className="card space-y-5">
      <div>
        <h3 className="section-title mb-1">Forensic Analysis</h3>
        <p className="text-xs text-gray-500">
          Signal-level breakdown from frequency, texture, and facial structure analysis.
        </p>
      </div>

      {/* Indicator chips — mirrors v1 report header */}
      <div className="flex flex-wrap gap-2">
        {indicators.map((ind) => (
          <span
            key={ind.label}
            className={`text-[10px] font-semibold px-2.5 py-1 rounded-full border ${INDICATOR_STYLES[ind.level]}`}
          >
            {ind.label}
          </span>
        ))}
      </div>

      {/* Three-column grid */}
      <div className="grid grid-cols-1 md:grid-cols-3 gap-4">

        {/* ── FFT Frequency Analysis ──────────────────────────────────────── */}
        <div className="space-y-3">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Frequency Domain (FFT)
          </p>
          <MetricBar label="Low Frequency Energy"  value={analysis.fft.low_freq}              color="#58a6ff" />
          <MetricBar label="Mid Frequency Energy"  value={analysis.fft.mid_freq}              color="#58a6ff" />
          <MetricBar label="High Frequency Energy" value={analysis.fft.high_freq}             color="#f59e0b" />
          <MetricBar label="Spectral Irregularity" value={analysis.fft.spectral_irregularity} color="#ef4444" />
          <p className="text-[10px] text-gray-600 leading-relaxed">
            High spectral irregularity or flat high-freq energy indicates GAN/diffusion synthesis artefacts.
          </p>
        </div>

        {/* ── Texture & Artifacts ─────────────────────────────────────────── */}
        <div className="space-y-3">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Texture &amp; Artifacts
          </p>
          <MetricBar label="Sharpness"              value={analysis.texture.sharpness}             color="#22c55e" />
          <MetricBar label="Texture Uniformity"     value={analysis.texture.texture_uniformity}    color="#f59e0b" />
          <MetricBar label="Noise Level"            value={analysis.texture.noise_level}           color="#ef4444" />
          <MetricBar label="Compression Artifacts"  value={analysis.texture.compression_artifacts} color="#ef4444" />
          <p className="text-[10px] text-gray-600 leading-relaxed">
            Deepfakes often show over-smooth texture (low uniformity) or unnatural noise patterns.
          </p>
        </div>

        {/* ── Facial Structure ────────────────────────────────────────────── */}
        <div className="space-y-3">
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider">
            Facial Structure
          </p>

          {analysis.symmetry_score !== null && analysis.symmetry_score !== undefined ? (
            <div className="space-y-3">
              <MetricBar
                label="Facial Symmetry"
                value={analysis.symmetry_score}
                invert
              />
              <p className="text-[10px] text-gray-600">
                Real faces have slight natural asymmetry. Very high or very low symmetry can indicate deepfake compositing.
              </p>
            </div>
          ) : (
            <p className="text-[11px] text-gray-600">Symmetry unavailable — no face detected.</p>
          )}

          {analysis.skin ? (
            <div className="space-y-3 pt-1">
              <p className="text-xs text-gray-500">Skin Quality</p>
              <MetricBar label="Pore Detail"  value={analysis.skin.pore_detail} invert color="#22c55e" />
              <MetricBar label="Blotchiness"  value={analysis.skin.blotchiness} color="#f59e0b" />
              <MetricBar label="Edge Blending" value={analysis.skin.edge_blend} invert color="#22c55e" />
              <p className="text-[10px] text-gray-600">
                Low pore detail + perfect edge blending = common AI-skin signature.
              </p>
            </div>
          ) : (
            <p className="text-[11px] text-gray-600 pt-1">Skin analysis unavailable — no face detected.</p>
          )}
        </div>
      </div>

      {/* Region Attention */}
      {hasRegions && (
        <div>
          <p className="text-xs font-semibold text-gray-400 uppercase tracking-wider mb-3">
            Facial Region Attention
          </p>
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-5 gap-2">
            {topRegions.map((region) => {
              const scores = analysis.region_scores[region]
              const attention = scores?.attention ?? 0
              const texture = scores?.texture ?? 0
              const attnPct = Math.round(attention * 100)
              const label = region.replace(/_/g, ' ')
              const isHot = attention > 0.5
              return (
                <div
                  key={region}
                  className={`rounded-xl p-3 border text-center ${
                    isHot
                      ? 'bg-red-500/10 border-red-500/30'
                      : 'bg-gray-900/60 border-gray-800'
                  }`}
                >
                  <div className={`text-sm font-bold ${isHot ? 'text-red-400' : 'text-gray-300'}`}>
                    {attnPct}%
                  </div>
                  <div className="text-[10px] text-gray-500 capitalize mt-0.5">{label}</div>
                  <div className="text-[9px] text-gray-600 mt-0.5">
                    tex {Math.round(texture * 100)}%
                  </div>
                </div>
              )
            })}
          </div>
          <p className="text-[10px] text-gray-600 mt-2">
            Attention = how much each model focused on this region. High attention on mouth/eyes is a strong deepfake signal.
          </p>
        </div>
      )}
    </div>
  )
}
