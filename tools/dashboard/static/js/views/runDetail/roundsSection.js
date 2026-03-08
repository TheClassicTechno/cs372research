import { fetchRound } from '../../api/runs.js';
import { buildAgentCards } from '../../components/card.js';
import { appState } from '../../state.js';

export function loadRoundAgents(experiment, runId, roundNum) {
  // Find the button that was clicked and use its parent as the container
  var buttons = document.querySelectorAll('[data-action="load-agents"][data-round="' + roundNum + '"]');
  var container = null;
  for (var i = 0; i < buttons.length; i++) {
    if (buttons[i].dataset.experiment === experiment && buttons[i].dataset.runId === runId) {
      container = buttons[i].parentElement;
      break;
    }
  }
  if (!container) return;

  container.innerHTML = '<span class="loading">Loading agent details...</span>';
  fetchRound(experiment, runId, roundNum)
    .then(function (detail) {
      container.innerHTML = buildAgentCards(detail);
    })
    .catch(function () {
      container.innerHTML = '<span style="color:#900;">Failed to load agent details.</span>';
    });
}
