/**
 * judgePortfolioSection.js
 *
 * Renders the FINAL ALLOCATIONS section on the run detail page:
 * combined allocation table, consensus performance, and per-agent performance.
 */
import { fetchPortfolio, fetchPerformance, fetchAgentPerformance, fetchRoundPerformance, fetchDebateImpact } from '../../api/runs.js';
import { buildSimpleAllocTable, buildAgentPerfTable, buildRoundAllocTable, buildDebateImpactTable, buildMeanPortfolioTable } from '../../components/table.js';
import { esc } from '../../utils/dom.js';
import { fmtPct, numFmt } from '../../utils/format.js';
import { makeAgentLabel } from '../../utils/agentLabel.js';
import { appState } from '../../state.js';

/**
 * Load and render the combined allocation table into #judge-alloc-wrap.
 * Falls back to buildSimpleAllocTable when trajectory data is unavailable.
 */
function loadAllocTable(experiment, runId, finalPortfolio, agentLabel, token) {
  fetchPortfolio(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      var wrap = document.getElementById('judge-alloc-wrap');
      if (!wrap) return;
      if (!data || data.length === 0) {
        wrap.innerHTML = buildSimpleAllocTable(finalPortfolio);
        return;
      }
      var last = data[data.length - 1];
      var agents = last.allocations || {};
      var agentNames = Object.keys(agents).sort();

      var tickers = Object.keys(finalPortfolio).sort(function (a, b) {
        return (finalPortfolio[b] || 0) - (finalPortfolio[a] || 0);
      });

      var th = '<table class="data-table" id="judge-alloc-table">';
      th += '<tr><th></th>';
      for (var a = 0; a < agentNames.length; a++) {
        th += '<th>' + esc(agentLabel(agentNames[a]).toUpperCase()) + '</th>';
      }
      th += '<th>JUDGE</th></tr>';

      for (var t = 0; t < tickers.length; t++) {
        var ticker = tickers[t];
        th += '<tr><td style="font-weight:600;">' + esc(ticker) + '</td>';
        for (var a = 0; a < agentNames.length; a++) {
          var w = agents[agentNames[a]] ? agents[agentNames[a]][ticker] : null;
          th += '<td style="font-weight:600;text-align:right;">' + fmtPct(w) + '</td>';
        }
        th += '<td style="font-weight:600;text-align:right;border-left:2px solid #d6c4a1;">' + fmtPct(finalPortfolio[ticker]) + '</td>';
        th += '</tr>';
      }
      th += '</table>';
      wrap.innerHTML = th;
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      var wrap = document.getElementById('judge-alloc-wrap');
      if (wrap) wrap.innerHTML = buildSimpleAllocTable(finalPortfolio);
    });
}

/**
 * Load and render consensus performance metrics into #perf-metrics.
 */
function loadConsensusPerf(experiment, runId, token) {
  fetchPerformance(experiment, runId)
    .then(function (perf) {
      if (appState.viewToken !== token) return;
      var perfDiv = document.getElementById('perf-metrics');
      if (!perfDiv) return;

      if (perf.error) {
        perfDiv.innerHTML = '<span style="color:#666;font-size:0.85em;">' + esc(perf.error) + '</span>';
        return;
      }

      var profitCls = perf.profit >= 0 ? 'perf-profit' : 'perf-loss';
      var profitSign = perf.profit >= 0 ? '+' : '';
      var returnPct = perf.return_pct;

      var ph = '<table class="data-table" id="perf-table">';
      ph += '<tr><th>JUDGE</th><th>Value</th></tr>';
      ph += '<tr><td>Initial Capital</td><td>$' + numFmt(perf.initial_capital) + '</td></tr>';
      ph += '<tr><td>Final Value</td><td class="' + profitCls + '">$' + numFmt(perf.final_value) + '</td></tr>';
      ph += '<tr><td>Profit/Loss</td><td class="' + profitCls + '">' + profitSign + '$' + numFmt(Math.abs(perf.profit)) + '</td></tr>';
      ph += '<tr><td>Return</td><td class="' + profitCls + '">' + profitSign + returnPct.toFixed(2) + '%</td></tr>';
      ph += '</table>';
      perfDiv.innerHTML = ph;
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      var perfDiv = document.getElementById('perf-metrics');
      if (perfDiv) perfDiv.innerHTML = '<span style="color:#666;font-size:0.85em;">Performance data unavailable</span>';
    });
}

/**
 * Load and render per-agent performance tables into #agent-perf-wrap.
 */
function loadAgentPerf(experiment, runId, agentLabel, token) {
  fetchAgentPerformance(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      var wrap = document.getElementById('agent-perf-wrap');
      if (!wrap) return;
      if (data.error || !data.agents) {
        wrap.innerHTML = '';
        return;
      }
      var roles = Object.keys(data.agents).sort();
      var html = '';
      for (var i = 0; i < roles.length; i++) {
        var label = agentLabel(roles[i]).toUpperCase();
        html += '<div>' + buildAgentPerfTable(data.agents[roles[i]], label) + '</div>';
      }
      wrap.innerHTML = html;
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      var wrap = document.getElementById('agent-perf-wrap');
      if (wrap) wrap.innerHTML = '';
    });
}

/**
 * Build HTML for all rounds from trajectory and performance data.
 * Returns concatenated HTML string for each round's phases.
 */
function buildAllRoundsHtml(trajectory, roundPerf, agentLabel) {
  var perfByRound = {};
  if (Array.isArray(roundPerf)) {
    for (var p = 0; p < roundPerf.length; p++) {
      perfByRound[roundPerf[p].round] = roundPerf[p];
    }
  }
  var html = '';
  for (var i = 0; i < trajectory.length; i++) {
    var entry = trajectory[i];
    var perf = perfByRound[entry.round];
    html += buildRoundSection(entry, perf !== undefined ? perf : {}, entry.round, agentLabel);
  }
  return html;
}

/**
 * Render per-round allocation and performance tables into #per-round-sections.
 * Fetches portfolio trajectory and round performance data in parallel.
 */
function loadPerRoundSections(experiment, runId, agentLabel, token) {
  Promise.all([
    fetchPortfolio(experiment, runId),
    fetchRoundPerformance(experiment, runId),
  ])
    .then(function (results) {
      if (appState.viewToken !== token) return;
      var wrap = document.getElementById('per-round-sections');
      if (!wrap) return;
      var trajectory = results[0];
      if (!Array.isArray(trajectory) || trajectory.length === 0) {
        wrap.innerHTML = '';
        return;
      }
      wrap.innerHTML = buildAllRoundsHtml(trajectory, results[1], agentLabel);
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      var wrap = document.getElementById('per-round-sections');
      if (wrap) wrap.innerHTML = '';
    });
}

/**
 * Build HTML for a single round's proposal and revision sections.
 * Returns HTML string with allocation tables and performance tables.
 */
function buildRoundSection(entry, perf, roundNum, agentLabel) {
  var h = '';
  var phases = [
    { key: 'proposals', label: 'PROPOSALS' },
    { key: 'revisions', label: 'REVISIONS' },
  ];
  for (var p = 0; p < phases.length; p++) {
    var phase = phases[p];
    var agents = entry[phase.key] || {};
    var agentNames = Object.keys(agents).sort();
    if (agentNames.length === 0) continue;

    var testId = 'round-' + roundNum + '-' + phase.key;
    h += '<div class="ov-title" style="margin-top:16px;" data-testid="' + esc(testId) + '-title">';
    h += 'ROUND ' + roundNum + ' — ' + phase.label + '</div>';
    h += '<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;">';
    h += buildRoundAllocTable(agents, agentNames, agentLabel, testId + '-alloc');
    h += buildRoundPerfTables(perf, phase.key, agentNames, agentLabel);
    h += '</div>';
  }
  return h;
}

/**
 * Build per-agent performance tables for a round phase.
 * Returns HTML string with agent perf tables in a flex row.
 */
function buildRoundPerfTables(perf, phaseKey, agentNames, agentLabel) {
  var phasePerf = perf[phaseKey];
  if (!phasePerf) return '';
  var h = '';
  for (var i = 0; i < agentNames.length; i++) {
    var role = agentNames[i];
    var agentPerf = phasePerf[role];
    if (!agentPerf) continue;
    var label = agentLabel(role).toUpperCase();
    h += '<div>' + buildAgentPerfTable(agentPerf, label) + '</div>';
  }
  return h;
}

/**
 * Load and render the debate impact section into #debate-impact-section.
 * Shows per-agent R1→final deltas and mean portfolio comparison.
 */
function loadDebateImpact(experiment, runId, agentLabel, token) {
  fetchDebateImpact(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      var wrap = document.getElementById('debate-impact-section');
      if (!wrap) return;
      if (data.error) { wrap.innerHTML = ''; return; }
      var mp = data.mean_portfolios;
      var html = '<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;">';
      html += buildDebateImpactTable(data.agent_deltas, agentLabel);
      html += buildMeanPortfolioTable(mp.r1_proposals, mp.r1_revisions, 'R1', 'debate-impact-mean');
      html += buildMeanPortfolioTable(mp.r2_proposals, mp.r2_revisions, 'R2', 'debate-impact-mean-r2');
      html += '</div>';
      wrap.innerHTML = html;
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      var wrap = document.getElementById('debate-impact-section');
      if (wrap) wrap.innerHTML = '';
    });
}

/**
 * Orchestrate the FINAL ALLOCATIONS section: layout scaffolding + async fetches.
 * Renders per-round allocation tables above the final allocations.
 */
export function loadJudgePortfolio(experiment, runId, finalPortfolio, manifest, token) {
  var div = document.getElementById('judge-portfolio-section');
  if (!div) return;
  if (!finalPortfolio || Object.keys(finalPortfolio).length === 0) {
    div.innerHTML = '';
    return;
  }

  var h = '<div class="ov-title" style="margin-top:16px;">DEBATE IMPACT</div>';
  h += '<div id="debate-impact-section"></div>';
  h += '<div id="per-round-sections"></div>';
  h += '<div class="ov-title" style="margin-top:16px;">FINAL ALLOCATIONS</div>';
  h += '<div id="judge-portfolio-layout" style="display:flex;gap:24px;align-items:flex-start;">';
  h += '<div id="judge-alloc-wrap"><span style="color:#666;font-size:0.85em;">Loading allocations...</span></div>';
  h += '<div id="perf-wrap" style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;">';
  h += '<div id="perf-metrics"><span style="color:#666;font-size:0.85em;">Loading performance...</span></div>';
  h += '<div id="agent-perf-wrap" style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;"></div>';
  h += '</div></div>';
  div.innerHTML = h;

  var agentLabel = makeAgentLabel(manifest);
  loadPerRoundSections(experiment, runId, agentLabel, token);
  loadDebateImpact(experiment, runId, agentLabel, token);
  loadAllocTable(experiment, runId, finalPortfolio, agentLabel, token);
  loadConsensusPerf(experiment, runId, token);
  loadAgentPerf(experiment, runId, agentLabel, token);
}
