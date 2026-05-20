import { useEffect } from 'react'
import { useNavigate } from 'react-router-dom'
import { ArrowLeft, Zap, Film, ImageIcon } from 'lucide-react'
import { useDetectionStore } from '@/store/detectionStore'
import { VerdictBanner } from '@/components/verdict/VerdictBanner'
import { ExplanationsCard } from '@/components/verdict/ExplanationsCard'
import { HeatmapViewer } from '@/components/heatmap/HeatmapViewer'
import { ModelVoteTable } from '@/components/ensemble/ModelVoteTable'
import { FrameTimeline } from '@/components/video/FrameTimeline'
import { isVideoResult } from '@/types/detection'

export function ResultsPage() {
  const navigate = useNavigate()
  const { result, videoResult, previewUrl, isVideo, reset } = useDetectionStore()

  // Redirect if neither result type is available
  const anyResult = result || videoResult
  useEffect(() => {
    if (!anyResult) navigate('/')
  }, [anyResult, navigate])

  if (!anyResult) return null

  const handleNew = () => {
    reset()
    navigate('/')
  }

  const availableModels = Object.keys(anyResult.model_votes)
  const isVid = isVideoResult(anyResult)

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4 sticky top-0 z-20 bg-gray-950/90 backdrop-blur">
        <div className="max-w-6xl mx-auto flex items-center gap-4">
          <button
            onClick={handleNew}
            className="flex items-center gap-2 text-gray-400 hover:text-gray-200 transition-colors text-sm"
          >
            <ArrowLeft size={16} />
            {isVid ? 'New Video' : 'New Image'}
          </button>
          <div className="flex items-center gap-2 ml-2">
            <div className="p-1.5 bg-brand-600 rounded-lg">
              <Zap size={14} className="text-white" />
            </div>
            <span className="font-semibold text-gray-200">Deepfake Detector</span>
          </div>
          <div className="ml-auto flex items-center gap-2 text-xs text-gray-600">
            {/* Media type badge */}
            <span className={`inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-xs font-medium border ${
              isVid
                ? 'bg-blue-900/30 text-blue-400 border-blue-800'
                : 'bg-brand-900/30 text-brand-400 border-brand-800'
            }`}>
              {isVid ? <Film size={10} /> : <ImageIcon size={10} />}
              {isVid ? 'VIDEO' : 'IMAGE'}
            </span>
            <span>{Math.round(anyResult.total_inference_time_ms)} ms · {availableModels.length} models</span>
          </div>
        </div>
      </header>

      <main className="flex-1 max-w-6xl mx-auto w-full px-4 py-8 space-y-6">
        {/* Verdict banner — full width */}
        <VerdictBanner
          verdict={anyResult.verdict}
          finalScore={anyResult.final_score}
          isUncertain={anyResult.is_uncertain}
          faceDetected={isVid ? anyResult.faces_detected > 0 : (result?.face_detected ?? false)}
          inferenceMs={anyResult.total_inference_time_ms}
          framesAnalyzed={isVid ? anyResult.frames_analyzed : undefined}
          facesDetected={isVid ? anyResult.faces_detected : undefined}
          temporalConsistency={isVid ? anyResult.temporal_consistency : undefined}
        />

        {/* Video-specific: Frame timeline */}
        {isVid && anyResult.frame_results.length > 0 && (
          <FrameTimeline
            frames={anyResult.frame_results}
            temporalConsistency={anyResult.temporal_consistency}
          />
        )}

        {/* Two-column layout */}
        <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
          {/* Left: heatmap (image only) or placeholder for video */}
          <div className="space-y-6">
            {!isVid && previewUrl && result && (
              <HeatmapViewer
                resultId={result.result_id}
                originalUrl={previewUrl}
                availableModels={availableModels}
              />
            )}
            {isVid && (
              <div className="card flex flex-col items-center justify-center py-10 text-center space-y-3">
                <Film size={36} className="text-gray-600" />
                <p className="text-sm text-gray-500">
                  Heatmaps are generated per-frame during analysis.
                </p>
                <p className="text-xs text-gray-600">
                  Check the frame timeline above for the per-frame score breakdown.
                </p>
              </div>
            )}
          </div>

          {/* Right: model votes + text explanations */}
          <div className="space-y-6">
            <ModelVoteTable
              votes={anyResult.model_votes}
              weights={anyResult.fusion_weights}
            />
            <ExplanationsCard explanations={anyResult.explanations} />
          </div>
        </div>
      </main>

      <footer className="border-t border-gray-800 px-6 py-4 text-center text-xs text-gray-700">
        Deepfake Detector · Multi-model ensemble: ViT · SigLIP · F3Net · EfficientNet-B4 · Hive AI
      </footer>
    </div>
  )
}
