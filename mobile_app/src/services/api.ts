/**
 * API service for communicating with the Mobile Insights server.
 */
import {API_BASE_URL} from '../config';
import type {
  UploadResponse,
  ResultsResponse,
  UploadsListResponse,
} from '../types';

/**
 * Upload a video file to the server.
 */
export async function uploadVideo(
  fileUri: string,
  fileName: string,
  fileType: string,
): Promise<UploadResponse> {
  const formData = new FormData();
  formData.append('video', {
    uri: fileUri,
    name: fileName,
    type: fileType,
  } as unknown as Blob);

  const res = await fetch(`${API_BASE_URL}/upload`, {
    method: 'POST',
    body: formData,
    headers: {
      'Content-Type': 'multipart/form-data',
    },
  });

  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `Upload failed (${res.status})`);
  }

  return res.json();
}

/**
 * Fetch the list of uploaded videos with their processing status.
 */
export async function fetchUploads(): Promise<UploadsListResponse> {
  const res = await fetch(`${API_BASE_URL}/uploads`);
  if (!res.ok) {
    throw new Error(`Failed to fetch uploads (${res.status})`);
  }
  return res.json();
}

/**
 * Fetch highlight results for a specific video.
 */
export async function fetchResults(id: string): Promise<ResultsResponse> {
  const res = await fetch(`${API_BASE_URL}/results/${encodeURIComponent(id)}`);
  if (!res.ok) {
    const body = await res.json().catch(() => ({}));
    throw new Error(body.error ?? `Failed to fetch results (${res.status})`);
  }
  return res.json();
}
