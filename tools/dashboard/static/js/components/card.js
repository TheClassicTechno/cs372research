/**
 * components/card.js
 *
 * Pure HTML builders for collapsible card UI elements used across views.
 */

import { esc } from '../utils/dom.js';
import { fmt } from '../utils/format.js';
import { renderTemplate } from '../utils/templates.js';
import '../templates/card.js';

/** Build a generic collapsible card with a title header and HTML body. */
export function buildCard(title, body, startOpen) {
  return renderTemplate('card', {
    openClass: startOpen ? ' open' : '',
    title: title,
    body: body,
  });
}

/** Build a collapsible card summarizing a single debate round's PID and CRIT metrics. */
export function buildRoundCard(rs, experiment, runId) {
  let roundNum = rs.round || 0;
  let metrics = rs.metrics || {};
  let pid = rs.pid || {};
  let crit = rs.crit || {};

  let summary = 'Round ' + roundNum + ' | ';
  summary += 'Beta: ' + fmt(pid.beta_in, 2) + ' \u2192 ' + fmt(pid.beta_new || metrics.beta_new, 2);
  summary += ' | <span style="text-decoration:overline">\u03c1</span>: ' + fmt(metrics.rho_bar || crit.rho_bar);
  summary += ' | ' + esc(pid.quadrant || metrics.quadrant || '\u2014');
  summary += ' | ' + esc(pid.tone_bucket || '\u2014');

  let body = '<pre class="content">' + esc(JSON.stringify(rs, null, 2)) + '</pre>';
  body += '<div style="margin-top:8px">';
  body += '<button data-action="load-agents" data-experiment="' + esc(experiment) + '" data-run-id="' + esc(runId) + '" data-round="' + roundNum + '">Load Agent Details</button>';
  body += '</div>';

  return '<div class="card" data-testid="round-card">' +
    '<div class="card-header"><span>' + summary + '</span><span class="arrow">&#9654;</span></div>' +
    '<div class="card-body">' + body + '</div></div>';
}

/** Build a collapsible card for a single agent event (proposal, critique, etc.). */
export function buildEventCard(ev) {
  let label = '[ROUND ' + ev.round + '] ' + ev.agent + ' \u2014 ' + ev.phase;
  let h = '<div class="card" data-testid="event-card" data-eid="' + esc(ev.id) + '">';
  h += '<div class="card-header">';
  h += '<span>' + esc(label) + '</span>';
  h += '<span class="arrow">&#9654;</span>';
  h += '</div>';
  h += '<div class="card-body">';
  h += '<pre class="content">' + esc(ev.content) + '</pre>';
  if (ev.portfolio) {
    h += '<div class="section-label">PORTFOLIO</div>';
    h += '<pre class="content">' + esc(JSON.stringify(ev.portfolio, null, 2)) + '</pre>';
  }
  h += '</div></div>';
  return h;
}

/** Build a styled header bar displaying experiment name, run ID, and status. */
export function buildRunHeader(data) {
  return '<div style="font-weight:600;font-size:0.85em;padding:6px 8px;border-top:1px solid #ccc;border-bottom:1px solid #ccc;margin-top:12px;margin-bottom:6px;background:#f0f0f0;">' +
    esc(data.experiment + ' / ' + data.run_id) + ' &mdash; ' + esc(data.status) + '</div>';
}

/**
 * Build expandable agent detail cards for a round.
 * Accepts an agentLabel resolver to map role keys to display names.
 */
export function buildAgentCards(detail, agentLabel) {
  let agents = detail.agents || {};
  let h = '';
  Object.keys(agents).sort().forEach(function (role) {
    let a = agents[role];
    let agentBody = '';
    if (a.proposal != null) {
      agentBody += '<div class="section-label">PROPOSAL</div>';
      agentBody += '<pre class="content">' + esc(typeof a.proposal === 'string' ? a.proposal : JSON.stringify(a.proposal, null, 2)) + '</pre>';
    }
    if (a.proposal_portfolio) {
      agentBody += '<div class="section-label">PROPOSAL PORTFOLIO</div>';
      agentBody += '<pre class="content">' + esc(JSON.stringify(a.proposal_portfolio, null, 2)) + '</pre>';
    }
    if (a.critique != null) {
      agentBody += '<div class="section-label">CRITIQUE</div>';
      agentBody += '<pre class="content">' + esc(typeof a.critique === 'string' ? a.critique : JSON.stringify(a.critique, null, 2)) + '</pre>';
    }
    if (a.revision != null) {
      agentBody += '<div class="section-label">REVISION</div>';
      agentBody += '<pre class="content">' + esc(typeof a.revision === 'string' ? a.revision : JSON.stringify(a.revision, null, 2)) + '</pre>';
    }
    if (a.revision_portfolio) {
      agentBody += '<div class="section-label">REVISION PORTFOLIO</div>';
      agentBody += '<pre class="content">' + esc(JSON.stringify(a.revision_portfolio, null, 2)) + '</pre>';
    }

    let cs = detail.crit_scores;
    if (cs && cs.agent_scores && cs.agent_scores[role]) {
      let ags = cs.agent_scores[role];
      agentBody += '<div class="section-label">CRIT SCORES</div>';
      agentBody += '<pre class="content">rho_i: ' + fmt(ags.rho_i) + '\n';
      if (ags.pillar_scores) {
        Object.keys(ags.pillar_scores).forEach(function (p) {
          agentBody += p + ': ' + fmt(ags.pillar_scores[p]) + '\n';
        });
      }
      agentBody += '</pre>';
    }

    if (a.crit_request != null) {
      agentBody += '<div class="section-label">CRIT REQUEST</div>';
      agentBody += '<pre class="content">' + esc(a.crit_request) + '</pre>';
    }
    if (a.crit_response != null) {
      agentBody += '<div class="section-label">CRIT RESPONSE</div>';
      agentBody += '<pre class="content">' + esc(a.crit_response) + '</pre>';
    }

    let displayName = (typeof agentLabel === 'function') ? agentLabel(role) : role;
    h += '<div class="card">' +
      '<div class="card-header"><span>' + esc(displayName.toUpperCase()) + '</span><span class="arrow">&#9654;</span></div>' +
      '<div class="card-body">' + agentBody + '</div></div>';
  });
  return h;
}
