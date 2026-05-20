import { ShieldCheck, ShieldX, ShieldAlert, Clock, Camera, CameraOff, Film } from 'lucide-react'
import clsx from 'clsx'
import type { Verdict } from '@/types/detection'

interface VerdictBannerProps {
  verdict: Verdict
  finalScore: number
  isUncertain: boolean
  faceDetected: boolean
  inferenceMs: number
  // Video-specific (optional)
  framesAnalyzed?: number
  facesDetected?: number
  temporalConsistency?: number
}

export function VerdictBanner({
  verdict, finalScore, isUncertain, faceDetected, inferenceMs,
  framesAnalyzed, facesDetected, temporalConsistency,
}: VerdictBannerProps) {
  const isVideo = framesAnalyzed !== undefined
  const fakePercent = Math.round(finalScore * 100)
  const realPercent = 100 - fakePercent
  const displayPercent = verdict === 'fake' ? fakePercent : realPercent

  const config = isUncertain
    ? {
        icon: ShieldAlert,
        label: 'UNCERTAIN',
        sub: isVideo
          ? `Low-confidence video verdict — borderline case (${fakePercent}% fake across ${framesAnalyzed} frames)`
          : 'Low-confidence verdict — borderline case',
        bg: 'bg-uncertain/10',
        border: 'border-uncertain/40',
        text: 'text-uncertain',
        ring: '#f59e0b',
      }
    : verdict === 'fake'
    ? {
        icon: ShieldX,
        label: 'LIKELY DEEPFAKE',
        sub: isVideo
          ? `This video is ${fakePercent}% likely to be a deepfake (${framesAnalyzed} frames analysed)`
          : `This image is ${fakePercent}% likely to be a deepfake`,
        bg: 'bg-fake/10',
        border: 'border-fake/40',
        text: 'text-fake',
        ring: '#ef4444',
      }
    : {
        icon: ShieldCheck,
        label: 'LIKELY AUTHENTIC',
        sub: isVideo
          ? `This video is ${realPercent}% likely to be real (${framesAnalyzed} frames analysed)`
          : `This image is ${realPercent}% likely to be real`,
        bg: 'bg-real/10',
        border: 'border-real/40',
        text: 'text-real',
        ring: '#22c55e',
      }

  const Icon = config.icon
  const circumference = 2 * Math.PI * 54
  const ringFraction = displayPercent / 100

  return (
    <div className={clsx('card border', config.border, config.bg)}>
      <div className="flex flex-col md:flex-row items-center gap-8">
        {/* Circular confidence gauge */}
        <div className="relative flex-shrink-0">
          <svg width={128} height={128} className="rotate-[-90deg]">
            <circle cx={64} cy={64} r={54} fill="none" stroke="#1f2937" strokeWidth={10} />
            <circle
              cx={64} cy={64} r={54} fill="none"
              stroke={config.ring} strokeWidth={10}
              strokeLinecap="round"
              strokeDasharray={circumference}
              strokeDashoffset={circumference * (1 - ringFraction)}
              style={{ transition: 'stroke-dashoffset 1s ease' }}
            />
          </svg>
          <div className="absolute inset-0 flex flex-col items-center justify-center">
            <span className={clsx('text-3xl font-bold', config.text)}>{displayPercent}%</span>
            <span className="text-xs text-gray-500 uppercase tracking-wide">
              {verdict === 'fake' ? 'fake' : 'real'}
            </span>
          </div>
        </div>

        {/* Verdict text */}
        <div className="flex-1 text-center md:text-left">
          <div className="flex items-center justify-center md:justify-start gap-3 mb-2">
            <Icon size={28} className={config.text} />
            <h2 className={clsx('text-2xl font-bold', config.text)}>{config.label}</h2>
          </div>
          <p className="text-gray-300 text-sm md:text-base mb-3">{config.sub}</p>

          <div className="flex flex-wrap gap-2 justify-center md:justify-start mb-3">
            <span className={clsx(
              'rounded-full px-3 py-1 text-xs font-medium border',
              isUncertain
                ? 'bg-uncertain/15 text-uncertain border-uncertain/40'
                : verdict === 'fake'
                ? 'bg-fake/15 text-fake border-fake/40'
                : 'bg-real/15 text-real border-real/40',
            )}>
              Final fused score: {fakePercent}% fake / {realPercent}% real
            </span>

            {isVideo ? (
              <>
                <span className="rounded-full px-3 py-1 text-xs font-medium border bg-gray-800 text-gray-300 border-gray-700 flex items-center gap-1">
                  <Film size={12} />
                  {framesAnalyzed} frames · {facesDetected} with face
                </span>
                {temporalConsistency !== undefined && (
                  <span className="rounded-full px-3 py-1 text-xs font-medium border bg-gray-800 text-gray-300 border-gray-700">
                    Temporal σ={temporalConsistency.toFixed(3)}
                  </span>
                )}
              </>
            ) : (
              <span className="rounded-full px-3 py-1 text-xs font-medium border bg-gray-800 text-gray-300 border-gray-700 flex items-center gap-1">
                {faceDetected ? <Camera size={12} /> : <CameraOff size={12} />}
                {faceDetected ? 'Face detected' : 'No face — full image used'}
              </span>
            )}
          </div>

          <div className="flex items-center gap-1 text-xs text-gray-500 justify-center md:justify-start">
            <Clock size={12} />
            Analysis completed in {Math.round(inferenceMs)} ms
          </div>
        </div>
      </div>
    </div>
  )
}
