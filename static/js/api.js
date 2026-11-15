/**
 * JSON API client. Requests that share a `key` cancel each other, so only the latest response
 * for a view is ever rendered (no race between slow and fast requests).
 */
export class ApiError extends Error {
  constructor(message, status, detail = null) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
    this.detail = detail;
  }
}

const inFlight = new Map();

export const isAbortError = (err) => err?.name === 'AbortError';

function describeDetail(body) {
  const detail = body?.detail;
  if (!detail) return '';
  if (typeof detail === 'string') return detail;
  if (Array.isArray(detail)) return detail.map((d) => d.msg || JSON.stringify(d)).join('; ');
  if (detail.message) return detail.message;
  return JSON.stringify(detail);
}

function buildUrl(path, params) {
  const query = new URLSearchParams(
    Object.entries(params)
      .filter(([, v]) => v !== undefined && v !== null)
      .map(([k, v]) => [k, String(v)]),
  ).toString();
  return query ? `${path}?${query}` : path;
}

export async function getJson(path, params = {}, { key } = {}) {
  const controller = new AbortController();
  if (key) {
    inFlight.get(key)?.abort();
    inFlight.set(key, controller);
  }
  try {
    const response = await fetch(buildUrl(path, params), { signal: controller.signal });
    const body = await response.json().catch(() => null);
    if (controller.signal.aborted) throw new DOMException('Superseded by a newer request', 'AbortError');
    if (!response.ok) throw new ApiError(describeDetail(body) || response.statusText, response.status, body?.detail);
    return body;
  } catch (err) {
    if (isAbortError(err) || err instanceof ApiError) throw err;
    throw new ApiError(err.message || 'Network error', 0);
  } finally {
    if (key && inFlight.get(key) === controller) inFlight.delete(key);
  }
}
