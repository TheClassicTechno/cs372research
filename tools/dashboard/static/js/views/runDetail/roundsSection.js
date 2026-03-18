/**
 * roundsSection.js
 *
 * Handles lazy-loading of agent detail cards when the user clicks
 * "Load Agent Details" on a round card.
 */
import { fetchRound } from '../../api/runs.js';
import { buildAgentCards } from '../../components/card.js';
import { makeAgentLabel } from '../../utils/agentLabel.js';
import { appState } from '../../state.js';

/**
 * Fetch and render agent detail cards for a specific round.
 * Resolves agent display names via the manifest stored in appState.
 */
export function loadRoundAgents(experiment, runId, roundNum) {
  let buttons = document.querySelectorAll('[data-action="load-agents"][data-round="' + roundNum + '"]');
  let match = Array.from(buttons).find(function (btn) {
    return btn.dataset.experiment === experiment && btn.dataset.runId === runId;
  });
  if (!match) return;
  let container = match.parentElement;

  let agentLabel = makeAgentLabel(appState.manifest);
  container.innerHTML = '<span class="loading">Loading agent details...</span>';
  fetchRound(experiment, runId, roundNum)
    .then(function (detail) {
      container.innerHTML = '<div data-testid="rounds-content">' + buildAgentCards(detail, agentLabel) + '</div>';
    })
    .catch(function () {
      container.innerHTML = '<span style="color:#900;">Failed to load agent details.</span>';
    });
}
