import { apiClient } from './client'
import type { DetectionResponse, VideoDetectionResponse } from '@/types/detection'

export async function submitDetection(
  file: File,
  onProgress?: (pct: number) => void,
): Promise<DetectionResponse> {
  const form = new FormData()
  form.append('file', file)

  const res = await apiClient.post<DetectionResponse>('/detect', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    },
  })
  return res.data
}

export async function submitVideoDetection(
  file: File,
  onProgress?: (pct: number) => void,
): Promise<VideoDetectionResponse> {
  const form = new FormData()
  form.append('file', file)

  const res = await apiClient.post<VideoDetectionResponse>('/detect/video', form, {
    headers: { 'Content-Type': 'multipart/form-data' },
    onUploadProgress: (e) => {
      if (onProgress && e.total) {
        onProgress(Math.round((e.loaded / e.total) * 100))
      }
    },
  })
  return res.data
}

export function isVideoFile(file: File): boolean {
  return file.type.startsWith('video/') || /\.(mp4|mov|avi|mkv|webm|m4v)$/i.test(file.name)
}

export function getHeatmapUrl(resultId: string, modelName: string): string {
  const base = (import.meta.env.VITE_API_URL as string) || '/api/v1'
  return `${base}/heatmap/${resultId}/${modelName}`
}
