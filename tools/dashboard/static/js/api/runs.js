import { fetchJSON } from './client.js';

export function fetchExperiments() {
  return fetchJSON('/runs/');
}

export function fetchRuns(experiment) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment));
}

export function fetchRunDetail(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId));
}

export function fetchPerformance(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/performance');
}

/** Fetch per-agent portfolio performance metrics. */
export function fetchAgentPerformance(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/performance/by-agent');
}

export function fetchPID(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/pid');
}

export function fetchCRIT(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/crit');
}

export function fetchDivergence(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/divergence');
}

export function fetchPortfolio(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/portfolio');
}

/** Fetch per-round, per-phase, per-agent portfolio performance. */
export function fetchRoundPerformance(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/performance/by-round');
}

/** Fetch debate impact: per-agent deltas and mean portfolio comparison. */
export function fetchDebateImpact(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/performance/debate-impact');
}

export function fetchRound(experiment, runId, roundNum) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/round/' + roundNum);
}

/** Fetch the file tree for a run directory. */
export function fetchTree(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/tree');
}

/** Fetch a single file from a run directory by relative path. */
export function fetchFile(experiment, runId, path) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/file?path=' + encodeURIComponent(path));
}

/** Fetch aggregate debate impact across experiments. */
export function fetchAblationDebateImpact() {
  return fetchJSON('/api/ablation/debate-impact');
}

/** Fetch the ablation summary JSON. */
export function fetchAblation() {
  return fetchJSON('/api/ablation');
}

/** Trigger regeneration of the ablation summary. */
export function regenerateAblation() {
  return fetch('/api/ablation/regenerate', { method: 'POST' }).then(function (r) { return r.json(); });
}
