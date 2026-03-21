/**
 * Shared TypeScript types for Mobile Insights.
 */

export interface Highlight {
  timestamp: number;
  score: number;
}

export interface UploadItem {
  filename: string;
  status: 'pending' | 'processing' | 'done' | 'error' | 'unknown';
}

export interface UploadResponse {
  message: string;
  filename: string;
}

export interface ResultsResponse {
  id: string;
  status: string;
  highlights: Highlight[];
  error?: string;
}

export interface UploadsListResponse {
  uploads: UploadItem[];
}
