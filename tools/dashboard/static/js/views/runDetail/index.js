import { fetchRunDetail } from '../../api/runs.js';
import { buildCard } from '../../components/card.js';
import { buildRoundCard } from '../../components/card.js';
import { buildOverviewPanel, buildConfigCards } from './overviewSection.js';
import { loadDivergenceSection } from './divergenceSection.js';
import { loadPIDStatsSection } from './pidStatsSection.js';
import { loadPIDSection } from './pidSection.js';
import { loadCRITSection } from './critSection.js';
import { loadPortfolioSection } from './portfolioSection.js';
import { loadFileExplorer } from './fileExplorerSection.js';
import { loadFileContent } from './fileExplorerSection.js';
import { loadJudgePortfolio } from './judgePortfolioSection.js';
import { loadRoundAgents } from './roundsSection.js';
import { esc } from '../../utils/dom.js';
import { fmt } from '../../utils/format.js';
import { appState, setManifest } from '../../state.js';
import { T } from '../../utils/labels.js';

export function renderRunDetailView(experiment, runId, token) {
  let appDiv = document.getElementById('app');
  appDiv.innerHTML = '<a class="back-link" href="#runs">&larr; Back to runs</a>' +
    '<div class="loading">Loading run detail...</div>';

  fetchRunDetail(experiment, runId)
    .then(function (detail) {
      if (appState.viewToken !== token) return;
      renderRunDetail(detail, experiment, runId, token);
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      appDiv.innerHTML = '<a class="back-link" href="#runs">&larr; Back to runs</a>' +
        '<p style="color:#900;">Failed to load run detail.</p>';
    });
}

function renderRunDetail(detail, experiment, runId, token) {
  let m = detail.manifest || {};
  setManifest(m);
  let appDiv = document.getElementById('app');

  let html = '<a class="back-link" href="#runs">&larr; Back to runs</a>';

  // Divergence + PID stats at the very top
  html += '<div id="divergence-section"></div>';
  html += '<div id="pid-stats-section"></div>';

  html += buildOverviewPanel(detail, experiment, runId);

  // Container for remaining async sections
  html += '<div id="detail-sections"></div>';
  appDiv.innerHTML = html;

  let sectionsDiv = document.getElementById('detail-sections');
  let sectionsHtml = '';

  // Config cards (debate config, scenario config, ticker perf, macro)
  sectionsHtml += buildConfigCards(detail);

  // Raw manifest JSON
  let cards = T('cards');
  sectionsHtml += buildCard(cards.config_raw, '<pre class="content">' + esc(JSON.stringify(m, null, 2)) + '</pre>');

  sectionsHtml += '<div id="pid-section"></div>';
  sectionsHtml += '<div id="crit-section"></div>';
  sectionsHtml += '<div id="portfolio-section"></div>';

  // Rounds (debate replay)
  if (detail.round_summaries && detail.round_summaries.length > 0) {
    let roundsInner = '';
    for (let i = 0; i < detail.round_summaries.length; i++) {
      let rs = detail.round_summaries[i];
      roundsInner += buildRoundCard(rs, experiment, runId);
    }
    sectionsHtml += buildCard(cards.rounds_replay, roundsInner, true);
  }

  // Final Portfolio
  if (detail.final_portfolio) {
    let sorted = Object.entries(detail.final_portfolio).sort(function (a, b) { return b[1] - a[1]; });
    let fpCfg = T('final_portfolio');
    let ptHtml = '<table class="data-table"><tr><th>' + esc(fpCfg.columns[0]) + '</th><th>' + esc(fpCfg.columns[1]) + '</th></tr>';
    for (let i = 0; i < sorted.length; i++) {
      ptHtml += '<tr><td>' + esc(sorted[i][0]) + '</td><td>' + fmt(sorted[i][1]) + '</td></tr>';
    }
    ptHtml += '</table>';
    sectionsHtml += buildCard(cards.final_portfolio, ptHtml);
  }

  // File Explorer
  sectionsHtml += '<div id="file-explorer-section"></div>';

  sectionsDiv.innerHTML = sectionsHtml;

  // Load async data
  loadJudgePortfolio(experiment, runId, detail.final_portfolio, detail.manifest, token);
  loadDivergenceSection(experiment, runId, token);
  loadPIDStatsSection(experiment, runId, token);
  loadPIDSection(experiment, runId, token);
  loadCRITSection(experiment, runId, token);
  loadPortfolioSection(experiment, runId, token);
  loadFileExplorer(experiment, runId, token);
}

export function handleAction(action, el) {
  if (action === 'load-agents') {
    const experiment = el.dataset.experiment;
    const runId = el.dataset.runId;
    const round = parseInt(el.dataset.round);
    loadRoundAgents(experiment, runId, round);
  } else if (action === 'load-file') {
    const experiment = el.dataset.experiment;
    const runId = el.dataset.runId;
    const path = el.dataset.path;
    loadFileContent(experiment, runId, path);
  }
}
