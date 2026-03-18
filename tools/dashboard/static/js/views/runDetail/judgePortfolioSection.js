/**
 * judgePortfolioSection.js
 *
 * Renders the FINAL ALLOCATIONS section on the run detail page:
 * combined allocation table, consensus performance, and per-agent performance.
 */
import { fetchPortfolio, fetchPerformance, fetchAgentPerformance, fetchRoundPerformance, fetchDebateImpact } from '../../api/runs.js';
import { buildSimpleAllocTable, buildAgentPerfTable, buildRoundAllocTable, buildDebateImpactTable, buildMeanPortfolioTable, buildSharpeTable, buildCollapseTable, buildDebateSummaryPanel } from '../../components/table.js';
import { esc } from '../../utils/dom.js';
import { fmtPct, numFmt } from '../../utils/format.js';
import { makeAgentLabel } from '../../utils/agentLabel.js';
import { appState } from '../../state.js';
import { T } from '../../utils/labels.js';

/**
 * Load and render the combined allocation table into #judge-alloc-wrap.
 * Falls back to buildSimpleAllocTable when trajectory data is unavailable.
 */
function loadAllocTable(experiment, runId, finalPortfolio, agentLabel, token) {
  fetchPortfolio(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      let wrap = document.getElementById('judge-alloc-wrap');
      if (!wrap) return;
      if (!data || data.length === 0) {
        wrap.innerHTML = buildSimpleAllocTable(finalPortfolio);
        return;
      }
      let last = data[data.length - 1];
      let agents = last.allocations || {};
      let agentNames = Object.keys(agents).sort();

      let tickers = Object.keys(finalPortfolio).sort(function (a, b) {
        return (finalPortfolio[b] || 0) - (finalPortfolio[a] || 0);
      });

      let th = '<table class="data-table" id="judge-alloc-table">';
      th += '<tr><th></th>';
      th += agentNames.map(function (name) {
        return '<th>' + esc(agentLabel(name).toUpperCase()) + '</th>';
      }).join('');
      th += '<th>' + esc(T('simple_alloc').columns[1]) + '</th></tr>';

      tickers.forEach(function (ticker) {
        th += '<tr><td style="font-weight:600;">' + esc(ticker) + '</td>';
        for (const name of agentNames) {
          let w = agents[name] ? agents[name][ticker] : null;
          th += '<td style="font-weight:600;text-align:right;">' + fmtPct(w) + '</td>';
        }
        th += '<td style="font-weight:600;text-align:right;border-left:2px solid #d6c4a1;">' + fmtPct(finalPortfolio[ticker]) + '</td>';
        th += '</tr>';
      });
      th += '</table>';
      wrap.innerHTML = th;
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      let wrap = document.getElementById('judge-alloc-wrap');
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
      let perfDiv = document.getElementById('perf-metrics');
      if (!perfDiv) return;

      if (perf.error) {
        perfDiv.innerHTML = '<span style="color:#666;font-size:0.85em;">' + esc(perf.error) + '</span>';
        return;
      }

      let profitCls = perf.profit >= 0 ? 'perf-profit' : 'perf-loss';
      let profitSign = perf.profit >= 0 ? '+' : '';
      let returnPct = perf.return_pct;

      let jpCfg = T('judge_perf');
      let ph = '<table class="data-table" id="perf-table">';
      ph += '<tr><th>' + esc(jpCfg.columns[0]) + '</th><th>' + esc(jpCfg.columns[1]) + '</th></tr>';
      ph += '<tr><td>' + esc(jpCfg.rows.initial_capital) + '</td><td>$' + numFmt(perf.initial_capital) + '</td></tr>';
      ph += '<tr><td>' + esc(jpCfg.rows.final_value) + '</td><td class="' + profitCls + '">$' + numFmt(perf.final_value) + '</td></tr>';
      ph += '<tr><td>' + esc(jpCfg.rows.profit_loss) + '</td><td class="' + profitCls + '">' + profitSign + '$' + numFmt(Math.abs(perf.profit)) + '</td></tr>';
      ph += '<tr><td>' + esc(jpCfg.rows.return) + '</td><td class="' + profitCls + '">' + profitSign + returnPct.toFixed(2) + '%</td></tr>';
      ph += '</table>';
      perfDiv.innerHTML = ph;
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      let perfDiv = document.getElementById('perf-metrics');
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
      let wrap = document.getElementById('agent-perf-wrap');
      if (!wrap) return;
      if (data.error || !data.agents) {
        wrap.innerHTML = '';
        return;
      }
      let roles = Object.keys(data.agents).sort();
      wrap.innerHTML = roles.map(function (role) {
        let label = agentLabel(role).toUpperCase();
        return '<div>' + buildAgentPerfTable(data.agents[role], label) + '</div>';
      }).join('');
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      let wrap = document.getElementById('agent-perf-wrap');
      if (wrap) wrap.innerHTML = '';
    });
}

/**
 * Build HTML for all rounds from trajectory and performance data.
 * Returns concatenated HTML string for each round's phases.
 */
function buildAllRoundsHtml(trajectory, roundPerf, agentLabel) {
  let perfByRound = {};
  if (Array.isArray(roundPerf)) {
    roundPerf.forEach(function (p) { perfByRound[p.round] = p; });
  }
  return trajectory.map(function (entry) {
    let perf = perfByRound[entry.round];
    return buildRoundSection(entry, perf !== undefined ? perf : {}, entry.round, agentLabel);
  }).join('');
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
      let wrap = document.getElementById('per-round-sections');
      if (!wrap) return;
      let trajectory = results[0];
      if (!Array.isArray(trajectory) || trajectory.length === 0) {
        wrap.innerHTML = '';
        return;
      }
      wrap.innerHTML = buildAllRoundsHtml(trajectory, results[1], agentLabel);
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      let wrap = document.getElementById('per-round-sections');
      if (wrap) wrap.innerHTML = '';
    });
}

/**
 * Build HTML for a single round's proposal and revision sections.
 * Returns HTML string with allocation tables and performance tables.
 */
function buildRoundSection(entry, perf, roundNum, agentLabel) {
  let h = '';
  let phases = [
    { key: 'proposals', label: 'PROPOSALS' },
    { key: 'revisions', label: 'REVISIONS' },
  ];
  for (const phase of phases) {
    let agents = entry[phase.key] || {};
    let agentNames = Object.keys(agents).sort();
    if (agentNames.length === 0) continue;

    let testId = 'round-' + roundNum + '-' + phase.key;
    h += '<div class="ov-title" style="margin-top:16px;" data-testid="' + esc(testId) + '-title">';
    h += 'ROUND ' + roundNum + ' \u2014 ' + phase.label + '</div>';
    h += '<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;">';
    h += buildRoundAllocTable(agents, agentNames, agentLabel, testId + '-alloc');
    h += buildRoundPerfTables(perf, phase.key, agentNames, agentLabel);
    h += '</div>';
  }

  // Intervention retry phases
  let retries = entry.retries || [];
  let retryPerfs = (perf && perf.retries) || [];
  h += retries.reduce(function (acc, retryEntry, r) {
    let retryAgents = retryEntry || {};
    let retryNames = Object.keys(retryAgents).sort();
    if (retryNames.length === 0) return acc;

    let retryNum = r + 1;
    let retryTestId = 'round-' + roundNum + '-retry-' + retryNum;
    acc += '<div class="ov-title" style="margin-top:16px;" data-testid="' + esc(retryTestId) + '-title">';
    acc += 'ROUND ' + roundNum + ' \u2014 RETRY ' + retryNum + ' (INTERVENTION)</div>';
    acc += '<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;">';
    acc += buildRoundAllocTable(retryAgents, retryNames, agentLabel, retryTestId + '-alloc');
    let retryPerf = retryPerfs[r];
    if (retryPerf) {
      acc += retryNames.filter(function (role) { return !!retryPerf[role]; }).map(function (role) {
        let label = agentLabel(role).toUpperCase();
        return '<div>' + buildAgentPerfTable(retryPerf[role], label) + '</div>';
      }).join('');
    }
    acc += '</div>';
    return acc;
  }, '');
  return h;
}

/**
 * Build per-agent performance tables for a round phase.
 * Returns HTML string with agent perf tables in a flex row.
 */
function buildRoundPerfTables(perf, phaseKey, agentNames, agentLabel) {
  let phasePerf = perf[phaseKey];
  if (!phasePerf) return '';
  return agentNames.filter(function (role) { return !!phasePerf[role]; }).map(function (role) {
    let label = agentLabel(role).toUpperCase();
    return '<div>' + buildAgentPerfTable(phasePerf[role], label) + '</div>';
  }).join('');
}

/**
 * Build HTML for all debate impact tables from API data.
 *
 * @param {object} data - Debate impact API response
 * @param {function} agentLabel - Maps role to display label
 * @returns {string} HTML string
 */
function buildDebateImpactHtml(data, agentLabel) {
  let mp = data.mean_portfolios;
  let html = buildDebateSummaryPanel(data.summary);
  html += '<div style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;">';
  html += buildDebateImpactTable(data.agent_deltas, agentLabel);
  html += buildMeanPortfolioTable(mp.r1_proposals, mp.r1_revisions, 'R1', 'debate-impact-mean', mp.r1_js);
  if (mp.r2_proposals && mp.r2_revisions) {
    html += buildMeanPortfolioTable(mp.r2_proposals, mp.r2_revisions, 'R2', 'debate-impact-mean-r2');
  }
  if (mp.r2_js) {
    html += buildMeanPortfolioTable(mp.r2_revisions, mp.r2_js, 'R2 JS', 'debate-impact-mean-js');
  }
  if (data.sharpe) {
    html += buildSharpeTable(data.sharpe);
  }
  html += '</div>';
  return html;
}

/**
 * Render collapse diagnostics into #collapse-section from debate impact data.
 *
 * @param {Array} collapse - Collapse diagnostics array
 * @param {function} agentLabel - Maps role to display label
 */
function renderCollapseDiagnostics(collapse, agentLabel) {
  if (!collapse || !Array.isArray(collapse) || collapse.length === 0) return;
  let collapseWrap = document.getElementById('collapse-section');
  if (!collapseWrap) return;
  let ch = '<div class="ov-title">' + esc(T('sections').collapse_diagnostics) + '</div>';
  ch += '<div class="collapse-definitions">';
  ch += '<div class="collapse-def-term">Movement</div>';
  ch += '<div class="collapse-def-desc">L\u2081 distance between proposal and revision vectors. How much the agent changed its portfolio.</div>';
  ch += '<div class="collapse-def-term">Toward Consensus</div>';
  ch += '<div class="collapse-def-desc">dist(proposal, consensus) \u2212 dist(revision, consensus), where consensus is the equal-weight mean of all proposals and dist is L\u2081. ';
  ch += 'Positive = agent moved toward the group mean; negative = moved away. Measures sycophantic drift vs. independent conviction.</div>';
  ch += '<div class="collapse-def-term">Collapse Share</div>';
  ch += '<div class="collapse-def-desc">This agent\u2019s share of total consensus-seeking movement across all agents. ';
  ch += 'Identifies who is capitulating the most. Only agents with positive Toward Consensus contribute.</div>';
  ch += '<div class="collapse-def-term">Dissent</div>';
  ch += '<div class="collapse-def-desc">L\u2081 distance between the agent\u2019s final revision and consensus. How differentiated the agent remains after revision.</div>';
  ch += '</div>';
  collapse.forEach(function (c) {
    ch += buildCollapseTable(c, agentLabel);
  });
  collapseWrap.innerHTML = ch;
}

/**
 * Load and render the debate impact section into #debate-impact-section.
 * Shows summary panel, per-agent deltas, mean portfolios, Sharpe, and collapse.
 */
function loadDebateImpact(experiment, runId, agentLabel, token) {
  fetchDebateImpact(experiment, runId)
    .then(function (data) {
      if (appState.viewToken !== token) return;
      let wrap = document.getElementById('debate-impact-section');
      if (!wrap) return;
      if (data.error) { wrap.innerHTML = ''; return; }
      wrap.innerHTML = buildDebateImpactHtml(data, agentLabel);
      renderCollapseDiagnostics(data.collapse, agentLabel);
    })
    .catch(function () {
      if (appState.viewToken !== token) return;
      let wrap = document.getElementById('debate-impact-section');
      if (wrap) wrap.innerHTML = '';
    });
}

/**
 * Orchestrate the FINAL ALLOCATIONS section: layout scaffolding + async fetches.
 * Renders per-round allocation tables above the final allocations.
 */
export function loadJudgePortfolio(experiment, runId, finalPortfolio, manifest, token) {
  let div = document.getElementById('judge-portfolio-section');
  if (!div) return;
  if (!finalPortfolio || Object.keys(finalPortfolio).length === 0) {
    div.innerHTML = '';
    return;
  }

  let sec = T('sections');
  let h = '<div class="ov-title" style="margin-top:16px;">' + esc(sec.debate_impact) + '</div>';
  h += '<div id="debate-impact-section"></div>';
  h += '<div id="collapse-section"></div>';
  h += '<div id="per-round-sections"></div>';
  h += '<div class="ov-title" style="margin-top:16px;">' + esc(sec.final_allocations) + '</div>';
  h += '<div id="judge-portfolio-layout" style="display:flex;gap:24px;align-items:flex-start;">';
  h += '<div id="judge-alloc-wrap"><span style="color:#666;font-size:0.85em;">Loading allocations...</span></div>';
  h += '<div id="perf-wrap" style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;">';
  h += '<div id="perf-metrics"><span style="color:#666;font-size:0.85em;">Loading performance...</span></div>';
  h += '<div id="agent-perf-wrap" style="display:flex;gap:24px;align-items:flex-start;flex-wrap:wrap;"></div>';
  h += '</div></div>';
  div.innerHTML = h;

  let agentLabel = makeAgentLabel(manifest);
  loadPerRoundSections(experiment, runId, agentLabel, token);
  loadDebateImpact(experiment, runId, agentLabel, token);
  loadAllocTable(experiment, runId, finalPortfolio, agentLabel, token);
  loadConsensusPerf(experiment, runId, token);
  loadAgentPerf(experiment, runId, agentLabel, token);
}
