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

export function fetchRound(experiment, runId, roundNum) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/round/' + roundNum);
}

export function fetchTree(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/tree');
}

export function fetchFile(experiment, runId, path) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/file?path=' + encodeURIComponent(path));
}
