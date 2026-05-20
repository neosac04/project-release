import { Spinner } from '@/components/shared/Spinner'

interface UploadProgressProps {
  status: 'uploading' | 'analyzing'
  progress: number
  isVideo?: boolean
}

export function UploadProgress({ status, progress, isVideo = false }: UploadProgressProps) {
  const mediaLabel = isVideo ? 'video' : 'image'
  const models = isVideo
    ? ['ViT', 'SigLIP', 'F3Net', 'EfficientNet', 'Frame analysis']
    : ['ViT', 'SigLIP', 'F3Net', 'EfficientNet']

  return (
    <div className="card flex flex-col items-center gap-6 py-10">
      <Spinner className="w-12 h-12" />
      <div className="text-center">
        <p className="text-lg font-semibold text-gray-200">
          {status === 'uploading' ? `Uploading ${mediaLabel}…` : 'Running detection pipeline…'}
        </p>
        <p className="text-sm text-gray-500 mt-1">
          {status === 'uploading'
            ? `${progress}% uploaded`
            : isVideo
              ? 'Sampling frames and running all models — this may take a minute or two'
              : 'Running 5 models + forensic analysis in parallel'}
        </p>
      </div>
      {status === 'uploading' && (
        <div className="w-64 h-2 bg-gray-800 rounded-full overflow-hidden">
          <div
            className="h-full bg-brand-500 rounded-full transition-all duration-300"
            style={{ width: `${progress}%` }}
          />
        </div>
      )}
      {status === 'analyzing' && (
        <div className="flex flex-wrap justify-center gap-3 text-xs text-gray-500">
          {models.map((m) => (
            <div key={m} className="flex flex-col items-center gap-1">
              <div className="w-2 h-2 rounded-full bg-brand-500 animate-pulse" />
              {m}
            </div>
          ))}
        </div>
      )}
    </div>
  )
}
