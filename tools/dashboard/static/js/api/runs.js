/**
 * api/runs.js
 *
 * API functions for fetching experiment, run, and round-level data from the server.
 */

import { fetchJSON } from './client.js';

/** Fetch the list of available experiments. */
export function fetchExperiments() {
  return fetchJSON('/runs/');
}

/** Fetch the list of runs for a given experiment. */
export function fetchRuns(experiment) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment));
}

/** Fetch detailed metadata for a specific run. */
export function fetchRunDetail(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId));
}

/** Fetch portfolio performance metrics for a run. */
export function fetchPerformance(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/performance');
}

/** Fetch per-agent portfolio performance metrics. */
export function fetchAgentPerformance(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/performance/by-agent');
}

/** Fetch PID controller diagnostics for a run. */
export function fetchPID(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/pid');
}

/** Fetch CRIT (critique) analysis data for a run. */
export function fetchCRIT(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/crit');
}

/** Fetch agent opinion divergence data for a run. */
export function fetchDivergence(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) + '/' + encodeURIComponent(runId) + '/divergence');
}

/** Fetch portfolio allocation snapshots for a run. */
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

/** Fetch per-round agent collapse diagnostics. */
export function fetchCollapse(experiment, runId) {
  return fetchJSON('/runs/' + encodeURIComponent(experiment) +
                   '/' + encodeURIComponent(runId) + '/collapse');
}

/** Fetch detailed data for a specific debate round. */
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

/** Fetch aggregate CRIT reasoning diagnostics for an experiment. */
export function fetchCritDiagnostics(experiment) {
  return fetchJSON('/api/ablation/crit-diagnostics/' + encodeURIComponent(experiment));
}

/** Fetch paired t-test results for an experiment's collapse ratios. */
export function fetchPairedTests(experiment) {
  return fetchJSON('/api/ablation/paired-tests/' + encodeURIComponent(experiment));
}

/** Fetch paired t-tests on financial metrics (judge portfolio) for an experiment. */
export function fetchFinancialTests(experiment) {
  return fetchJSON('/api/ablation/financial-tests/' + encodeURIComponent(experiment));
}

/** Fetch paired t-tests on financial metrics (mean agent revisions) for an experiment. */
export function fetchFinancialTestsMeanRev(experiment) {
  return fetchJSON('/api/ablation/financial-tests/' + encodeURIComponent(experiment) + '/mean-revisions');
}

/** Fetch cross-ablation financial significance summary. */
export function fetchFinancialSignificance() {
  return fetchJSON('/api/ablation/financial-significance');
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
