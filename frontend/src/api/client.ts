import axios from 'axios'

const BASE_URL = (import.meta.env.VITE_API_URL as string) || '/api/v1'

export const apiClient = axios.create({
  baseURL: BASE_URL,
  timeout: 300_000, // 5 min — video analysis can be slow
})
