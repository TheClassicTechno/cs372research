import { fetchJSON } from './client.js';

export function fetchLiveEvents() {
  return fetchJSON('/api/live_debate');
}

export function clearLiveEvents() {
  return fetch('/api/live_debate/clear', { method: 'POST' });
}
