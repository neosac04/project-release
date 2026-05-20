// Matches backend/app/schemas/response.py and video_response.py

export type Verdict = 'real' | 'fake'

export interface ModelVote {
  fake_prob: number
  real_prob: number
  inference_time_ms: number
}

// ── Analysis metrics ──────────────────────────────────────────────────────────

export interface FFTMetrics {
  low_freq: number
  mid_freq: number
  high_freq: number
  spectral_irregularity: number
  profile: number[]
}

export interface TextureMetrics {
  sharpness: number
  texture_uniformity: number
  noise_level: number
  compression_artifacts: number
}

export interface SkinMetrics {
  pore_detail: number
  blotchiness: number
  edge_blend: number
}

export interface AnalysisMetrics {
  fft: FFTMetrics
  texture: TextureMetrics
  symmetry_score: number | null
  skin: SkinMetrics | null
  top_attention_regions: string[]
  region_scores: Record<string, Record<string, number>>
}

// ── Image detection ───────────────────────────────────────────────────────────

export interface DetectionResponse {
  result_id: string
  media_type: 'image'
  final_score: number
  verdict: Verdict
  face_detected: boolean
  is_uncertain: boolean
  model_votes: Record<string, ModelVote>
  fusion_weights: Record<string, number>
  explanations: string[]
  total_inference_time_ms: number
  analysis: AnalysisMetrics | null
}

// ── Video detection ───────────────────────────────────────────────────────────

export interface FrameResult {
  frame_index: number
  timestamp_sec: number
  final_score: number
  face_detected: boolean
}

export interface VideoDetectionResponse {
  result_id: string
  media_type: 'video'
  final_score: number
  verdict: Verdict
  is_uncertain: boolean
  frames_analyzed: number
  faces_detected: number
  frame_results: FrameResult[]
  temporal_consistency: number
  aggregation_strategy: string
  model_votes: Record<string, ModelVote>
  fusion_weights: Record<string, number>
  explanations: string[]
  total_inference_time_ms: number
  analysis: AnalysisMetrics | null
}

// ── Union helper ──────────────────────────────────────────────────────────────

export type AnyDetectionResult = DetectionResponse | VideoDetectionResponse

export function isVideoResult(r: AnyDetectionResult): r is VideoDetectionResponse {
  return r.media_type === 'video'
}
