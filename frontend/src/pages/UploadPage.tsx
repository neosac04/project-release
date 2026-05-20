import { useNavigate } from 'react-router-dom'
import { X, Zap, Film, ImageIcon } from 'lucide-react'
import { DropZone } from '@/components/upload/DropZone'
import { UploadProgress } from '@/components/upload/UploadProgress'
import { useDetectionStore } from '@/store/detectionStore'
import { useEffect } from 'react'

export function UploadPage() {
  const navigate = useNavigate()
  const { status, uploadProgress, selectedFile, previewUrl, isVideo, error, setFile, submit, reset } =
    useDetectionStore()

  useEffect(() => {
    if (status === 'complete') navigate('/results')
  }, [status, navigate])

  return (
    <div className="min-h-screen flex flex-col">
      {/* Header */}
      <header className="border-b border-gray-800 px-6 py-4">
        <div className="max-w-4xl mx-auto flex items-center gap-3">
          <div className="p-2 bg-brand-600 rounded-xl">
            <Zap size={20} className="text-white" />
          </div>
          <div>
            <h1 className="font-bold text-gray-100">Deepfake Detector</h1>
            <p className="text-xs text-gray-500">Multi-model AI Image & Video Forensics</p>
          </div>
        </div>
      </header>

      <main className="flex-1 flex flex-col items-center justify-center px-4 py-12">
        <div className="w-full max-w-2xl space-y-6">
          {/* Title */}
          <div className="text-center">
            <h2 className="text-3xl font-bold text-gray-100 mb-2">
              Detect Deepfakes & AI-Generated Media
            </h2>
            <p className="text-gray-500 max-w-lg mx-auto">
              Four deep-learning specialists — ViT, SigLIP, F3Net and EfficientNet —
              vote on your image or video and explain their reasoning.
            </p>
          </div>

          {/* Model pills */}
          <div className="flex flex-wrap gap-2 justify-center">
            {[
              { label: 'ViT',          desc: 'Full image · global patterns' },
              { label: 'SigLIP',       desc: 'Vision-language · 94% acc' },
              { label: 'F3Net',        desc: 'DCT frequency artifacts' },
              { label: 'EfficientNet', desc: 'Face texture forensics' },
            ].map((m) => (
              <div key={m.label} className="bg-gray-900 border border-gray-800 rounded-full px-3 py-1.5 text-xs">
                <span className="text-gray-300 font-medium">{m.label}</span>
                <span className="text-gray-600 ml-1">· {m.desc}</span>
              </div>
            ))}
          </div>

          {/* Upload area */}
          {(status === 'idle' || status === 'error') && !selectedFile && (
            <DropZone onFile={setFile} />
          )}

          {/* Preview + submit */}
          {selectedFile && (status === 'idle' || status === 'error') && (
            <div className="card space-y-4">
              <div className="flex items-center gap-4">
                {/* Thumbnail or video icon */}
                {previewUrl && !isVideo ? (
                  <img
                    src={previewUrl}
                    alt="preview"
                    className="w-20 h-20 object-cover rounded-xl border border-gray-700"
                  />
                ) : (
                  <div className="w-20 h-20 flex items-center justify-center rounded-xl border border-gray-700 bg-gray-800">
                    <Film size={28} className="text-gray-500" />
                  </div>
                )}
                <div className="flex-1 min-w-0">
                  <div className="flex items-center gap-2 mb-1">
                    {/* Media type badge */}
                    <span className={`inline-flex items-center gap-1 text-xs font-semibold px-2 py-0.5 rounded-full ${
                      isVideo
                        ? 'bg-blue-900/40 text-blue-400 border border-blue-800'
                        : 'bg-brand-900/40 text-brand-400 border border-brand-800'
                    }`}>
                      {isVideo ? <Film size={10} /> : <ImageIcon size={10} />}
                      {isVideo ? 'VIDEO' : 'IMAGE'}
                    </span>
                  </div>
                  <p className="font-medium text-gray-200 truncate">{selectedFile.name}</p>
                  <p className="text-sm text-gray-500">
                    {selectedFile.size > 1024 * 1024
                      ? `${(selectedFile.size / 1024 / 1024).toFixed(1)} MB`
                      : `${(selectedFile.size / 1024).toFixed(1)} KB`}
                  </p>
                </div>
                <button onClick={reset} className="text-gray-600 hover:text-gray-400 p-2">
                  <X size={18} />
                </button>
              </div>

              {isVideo && (
                <div className="bg-blue-950/30 border border-blue-900/40 rounded-lg px-4 py-3 text-xs text-blue-400">
                  Video analysis samples up to 32 frames. Processing may take 1–3 minutes.
                </div>
              )}

              {error && (
                <div className="bg-fake/10 border border-fake/30 rounded-lg px-4 py-3 text-sm text-fake">
                  {error}
                </div>
              )}

              <button className="btn-primary w-full" onClick={submit}>
                {isVideo ? 'Analyse Video' : 'Analyse Image'}
              </button>
            </div>
          )}

          {/* Progress */}
          {(status === 'uploading' || status === 'analyzing') && (
            <UploadProgress status={status} progress={uploadProgress} isVideo={isVideo} />
          )}

          {/* Feature grid */}
          {status === 'idle' && !selectedFile && (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 mt-4">
              {[
                { icon: '🎯', title: 'Final fused verdict', desc: 'Confidence-weighted ensemble of 4 specialists' },
                { icon: '🗳️', title: 'Per-model breakdown', desc: 'See exactly how each model voted' },
                { icon: '🔥', title: 'GradCAM heatmaps', desc: 'Visual evidence — see which regions flagged it' },
                { icon: '🎬', title: 'Video frame timeline', desc: 'Frame-by-frame score chart for video deepfakes' },
              ].map((f) => (
                <div key={f.title} className="card-sm">
                  <div className="text-2xl mb-2">{f.icon}</div>
                  <p className="text-sm font-medium text-gray-300">{f.title}</p>
                  <p className="text-xs text-gray-600 mt-1">{f.desc}</p>
                </div>
              ))}
            </div>
          )}
        </div>
      </main>
    </div>
  )
}
