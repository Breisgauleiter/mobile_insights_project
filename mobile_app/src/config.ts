/**
 * API configuration for the Mobile Insights app.
 * Set MOBILE_INSIGHTS_API_URL env var or edit the default below.
 */

// In development, use your local machine's IP (not localhost)
// because the RN emulator runs in its own network namespace.
export const API_BASE_URL =
  process.env.MOBILE_INSIGHTS_API_URL ?? 'http://10.0.2.2:3000';
