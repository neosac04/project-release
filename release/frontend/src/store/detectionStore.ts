import { create } from 'zustand'
import { immer } from 'zustand/middleware/immer'
import { submitDetection, submitVideoDetection, isVideoFile } from '@/api/detection'
import type { DetectionResponse, VideoDetectionResponse } from '@/types/detection'

type Status = 'idle' | 'uploading' | 'analyzing' | 'complete' | 'error'

interface DetectionStore {
  status: Status
  uploadProgress: number
  selectedFile: File | null
  previewUrl: string | null
  isVideo: boolean
  result: DetectionResponse | null
  videoResult: VideoDetectionResponse | null
  error: string | null
  selectedHeatmapModel: string

  setFile: (file: File) => void
  submit: () => Promise<void>
  setHeatmapModel: (model: string) => void
  reset: () => void
}

export const useDetectionStore = create<DetectionStore>()(
  immer((set, get) => ({
    status: 'idle',
    uploadProgress: 0,
    selectedFile: null,
    previewUrl: null,
    isVideo: false,
    result: null,
    videoResult: null,
    error: null,
    selectedHeatmapModel: 'ensemble',

    setFile: (file) => {
      const prev = get().previewUrl
      if (prev) URL.revokeObjectURL(prev)
      const video = isVideoFile(file)
      set((s) => {
        s.selectedFile = file
        s.previewUrl = video ? null : URL.createObjectURL(file)
        s.isVideo = video
        s.result = null
        s.videoResult = null
        s.error = null
        s.status = 'idle'
      })
    },

    submit: async () => {
      const file = get().selectedFile
      if (!file) return
      const video = get().isVideo
      set((s) => { s.status = 'uploading'; s.uploadProgress = 0; s.error = null })
      try {
        if (video) {
          const videoResult = await submitVideoDetection(file, (pct) => {
            set((s) => {
              s.uploadProgress = pct
              if (pct === 100) s.status = 'analyzing'
            })
          })
          set((s) => { s.videoResult = videoResult; s.status = 'complete' })
        } else {
          const result = await submitDetection(file, (pct) => {
            set((s) => {
              s.uploadProgress = pct
              if (pct === 100) s.status = 'analyzing'
            })
          })
          set((s) => { s.result = result; s.status = 'complete' })
        }
      } catch (err: unknown) {
        const msg = err instanceof Error ? err.message : 'Detection failed. Please try again.'
        set((s) => { s.error = msg; s.status = 'error' })
      }
    },

    setHeatmapModel: (model) => set((s) => { s.selectedHeatmapModel = model }),

    reset: () => {
      const prev = get().previewUrl
      if (prev) URL.revokeObjectURL(prev)
      set((s) => {
        s.status = 'idle'
        s.uploadProgress = 0
        s.selectedFile = null
        s.previewUrl = null
        s.isVideo = false
        s.result = null
        s.videoResult = null
        s.error = null
      })
    },
  })),
)
