/**
 * components/charts.js
 *
 * SVG chart builders for PID beta/rho and CRIT per-agent rho time-series.
 */

import { fmt } from '../utils/format.js';

/**
 * Render horizontal grid lines and Y-axis labels plus X-axis round labels.
 * @param {number} W - SVG width.
 * @param {number} H - SVG height.
 * @param {object} pad - Padding object with top, right, bottom, left.
 * @param {number[]} rounds - Array of round numbers.
 * @param {number} yMin - Minimum Y-axis value.
 * @param {number} yMax - Maximum Y-axis value.
 * @param {function} xPos - Maps a round number to its x pixel coordinate.
 * @param {function} yPos - Maps a metric value to its y pixel coordinate.
 * @returns {string} SVG markup for axes and grid.
 */
function buildChartAxes(W, H, pad, rounds, yMin, yMax, xPos, yPos) {
  let svg = '';
  for (let g = 0; g <= 4; g++) {
    let gv = yMin + (yMax - yMin) * g / 4;
    let gy = yPos(gv);
    svg += '<line x1="' + pad.left + '" y1="' + gy + '" x2="' + (W - pad.right) + '" y2="' + gy + '" stroke="#eee" />';
    svg += '<text x="' + (pad.left - 5) + '" y="' + (gy + 4) + '" text-anchor="end" fill="#666">' + gv.toFixed(2) + '</text>';
  }
  rounds.forEach(function (r) {
    svg += '<text x="' + xPos(r) + '" y="' + (H - pad.bottom + 20) + '" text-anchor="middle" fill="#666">R' + r + '</text>';
  });
  return svg;
}

/**
 * Render the beta polyline and data-point circles for the PID chart.
 * @param {object[]} data - Array of round data objects.
 * @param {number[]} rounds - Array of round numbers.
 * @param {(number|null)[]} betas - Array of beta values (may contain null).
 * @param {function} xPos - Maps a round number to its x pixel coordinate.
 * @param {function} yPos - Maps a metric value to its y pixel coordinate.
 * @returns {string} SVG markup for the beta series.
 */
function buildBetaPath(data, rounds, betas, xPos, yPos) {
  let svg = '';
  let betaPoints = data.reduce(function (pts, d, i) {
    if (betas[i] != null) pts.push(xPos(rounds[i]) + ',' + yPos(betas[i]));
    return pts;
  }, []);
  if (betaPoints.length > 1) {
    svg += '<polyline points="' + betaPoints.join(' ') + '" fill="none" stroke="#000" stroke-width="2" />';
  }
  data.forEach(function (d, i) {
    if (betas[i] != null) {
      svg += '<circle cx="' + xPos(rounds[i]) + '" cy="' + yPos(betas[i]) + '" r="3" fill="#000"><title>Beta: ' + fmt(betas[i]) + '</title></circle>';
    }
  });
  return svg;
}

/**
 * Render the rho_bar polyline and data-point circles (dashed, hollow markers).
 * @param {object[]} data - Array of round data objects.
 * @param {number[]} rounds - Array of round numbers.
 * @param {(number|null)[]} rhos - Array of rho_bar values (may contain null).
 * @param {function} xPos - Maps a round number to its x pixel coordinate.
 * @param {function} yPos - Maps a metric value to its y pixel coordinate.
 * @returns {string} SVG markup for the rho series.
 */
function buildRhoPath(data, rounds, rhos, xPos, yPos) {
  let svg = '';
  let rhoPoints = data.reduce(function (pts, d, i) {
    if (rhos[i] != null) pts.push(xPos(rounds[i]) + ',' + yPos(rhos[i]));
    return pts;
  }, []);
  if (rhoPoints.length > 1) {
    svg += '<polyline points="' + rhoPoints.join(' ') + '" fill="none" stroke="#000" stroke-width="2" stroke-dasharray="8,4" />';
  }
  data.forEach(function (d, i) {
    if (rhos[i] != null) {
      svg += '<circle cx="' + xPos(rounds[i]) + '" cy="' + yPos(rhos[i]) + '" r="3" fill="none" stroke="#000" stroke-width="1.5"><title>\u03c1: ' + fmt(rhos[i]) + '</title></circle>';
    }
  });
  return svg;
}

/**
 * Render horizontal grid lines, Y-axis labels, and X-axis round labels for the CRIT chart.
 * Uses pad.top + plotH + 20 for X-axis label placement (differs from PID chart).
 * @param {number} W - SVG width.
 * @param {object} pad - Padding object with top, right, bottom, left.
 * @param {number} plotH - Plot area height.
 * @param {number[]} rounds - Array of round numbers.
 * @param {number} yMin - Minimum Y-axis value.
 * @param {number} yMax - Maximum Y-axis value.
 * @param {function} xPos - Maps a round number to its x pixel coordinate.
 * @param {function} yPos - Maps a metric value to its y pixel coordinate.
 * @returns {string} SVG markup for axes and grid.
 */
function buildCRITAxes(W, pad, plotH, rounds, yMin, yMax, xPos, yPos) {
  let svg = '';
  for (let g = 0; g <= 4; g++) {
    let gv = yMin + (yMax - yMin) * g / 4;
    let gy = yPos(gv);
    svg += '<line x1="' + pad.left + '" y1="' + gy + '" x2="' + (W - pad.right) + '" y2="' + gy + '" stroke="#eee" />';
    svg += '<text x="' + (pad.left - 5) + '" y="' + (gy + 4) + '" text-anchor="end" fill="#666">' + gv.toFixed(2) + '</text>';
  }
  rounds.forEach(function (r) {
    svg += '<text x="' + xPos(r) + '" y="' + (pad.top + plotH + 20) + '" text-anchor="middle" fill="#666">R' + r + '</text>';
  });
  return svg;
}

/**
 * Render the rho_bar polyline and filled data-point circles for the CRIT chart.
 * @param {object[]} data - Array of round data objects.
 * @param {number[]} rounds - Array of round numbers.
 * @param {(number|null)[]} rhos - Array of rho_bar values (may contain null).
 * @param {function} xPos - Maps a round number to its x pixel coordinate.
 * @param {function} yPos - Maps a metric value to its y pixel coordinate.
 * @returns {string} SVG markup for the rho_bar series.
 */
function buildRhoBarPath(data, rounds, rhos, xPos, yPos) {
  let svg = '';
  let rhoPoints = data.reduce(function (pts, d, i) {
    if (rhos[i] != null) pts.push(xPos(rounds[i]) + ',' + yPos(rhos[i]));
    return pts;
  }, []);
  if (rhoPoints.length > 1) {
    svg += '<polyline points="' + rhoPoints.join(' ') + '" fill="none" stroke="#000" stroke-width="2" />';
  }
  data.forEach(function (d, i) {
    if (rhos[i] != null) {
      svg += '<circle cx="' + xPos(rounds[i]) + '" cy="' + yPos(rhos[i]) + '" r="3" fill="#000"><title>\u03c1: ' + fmt(rhos[i]) + '</title></circle>';
    }
  });
  return svg;
}

/**
 * Render dashed polylines and hollow circles for each agent's per-round rho_i.
 * @param {object[]} data - Array of round data objects with rho_i maps.
 * @param {number[]} rounds - Array of round numbers.
 * @param {string[]} roleList - Sorted list of agent role keys.
 * @param {function} xPos - Maps a round number to its x pixel coordinate.
 * @param {function} yPos - Maps a metric value to its y pixel coordinate.
 * @param {function} agentLabel - Maps a role key to a display label.
 * @returns {string} SVG markup for per-agent rho_i series.
 */
function buildAgentPaths(data, rounds, roleList, xPos, yPos, agentLabel) {
  let svg = '';
  let dashPatterns = ['5,5', '10,5', '2,5', '15,5,5,5'];
  roleList.forEach(function (role, ri) {
    let dash = dashPatterns[ri % dashPatterns.length];
    let pts = data.reduce(function (acc, d, i) {
      const v = d.rho_i ? d.rho_i[role] : null;
      if (v != null) acc.push(xPos(rounds[i]) + ',' + yPos(v));
      return acc;
    }, []);
    if (pts.length > 1) {
      svg += '<polyline points="' + pts.join(' ') + '" fill="none" stroke="#000" stroke-width="1.5" stroke-dasharray="' + dash + '" />';
    }
    data.forEach(function (d, i) {
      const v = d.rho_i ? d.rho_i[role] : null;
      if (v != null) {
        svg += '<circle cx="' + xPos(rounds[i]) + '" cy="' + yPos(v) + '" r="2" fill="none" stroke="#000"><title>' + agentLabel(role) + ' rho_i: ' + fmt(v) + '</title></circle>';
      }
    });
  });
  return svg;
}

/** Build an SVG line chart plotting beta and rho_bar across debate rounds. */
export function buildPIDChart(data) {
  let W = 700, H = 300, pad = { top: 20, right: 60, bottom: 40, left: 60 };
  let plotW = W - pad.left - pad.right;
  let plotH = H - pad.top - pad.bottom;

  let rounds = data.map(function (d) { return d.round; });
  let betas = data.map(function (d) { return d.beta_new != null ? d.beta_new : d.beta_in; });
  let rhos = data.map(function (d) { return d.rho_bar; });

  let allVals = betas.concat(rhos).filter(function (v) { return v != null; });
  if (allVals.length === 0) return '';
  let yMin = Math.max(0, Math.min.apply(null, allVals) - 0.05);
  let yMax = Math.min(1, Math.max.apply(null, allVals) + 0.05);
  if (yMax - yMin < 0.1) { yMin = Math.max(0, yMin - 0.05); yMax = Math.min(1, yMax + 0.05); }

  let xMin = Math.min.apply(null, rounds);
  let xMax = Math.max.apply(null, rounds);
  if (xMax === xMin) xMax = xMin + 1;

  /** Map a round number to its x pixel coordinate. */
  function xPos(r) { return pad.left + (r - xMin) / (xMax - xMin) * plotW; }
  /** Map a metric value to its y pixel coordinate. */
  function yPos(v) { return pad.top + (1 - (v - yMin) / (yMax - yMin)) * plotH; }

  let svg = '<svg class="chart" data-testid="pid-chart" width="' + W + '" height="' + H + '" xmlns="http://www.w3.org/2000/svg">';
  svg += buildChartAxes(W, H, pad, rounds, yMin, yMax, xPos, yPos);
  svg += buildBetaPath(data, rounds, betas, xPos, yPos);
  svg += buildRhoPath(data, rounds, rhos, xPos, yPos);

  // Legend
  svg += '<line x1="' + (W - pad.right + 5) + '" y1="' + (pad.top + 10) + '" x2="' + (W - pad.right + 25) + '" y2="' + (pad.top + 10) + '" stroke="#000" stroke-width="2" />';
  svg += '<text x="' + (W - pad.right + 28) + '" y="' + (pad.top + 14) + '" fill="#000" font-size="10">beta</text>';
  svg += '<line x1="' + (W - pad.right + 5) + '" y1="' + (pad.top + 28) + '" x2="' + (W - pad.right + 25) + '" y2="' + (pad.top + 28) + '" stroke="#000" stroke-width="2" stroke-dasharray="8,4" />';
  svg += '<text x="' + (W - pad.right + 28) + '" y="' + (pad.top + 32) + '" fill="#000" font-size="10">rho</text>';

  svg += '</svg>';
  return svg;
}

/** Build an SVG line chart plotting rho_bar and per-agent rho_i across rounds. */
export function buildCRITChart(data, agentLabel) {
  if (!agentLabel) agentLabel = function(r) { return r; };
  let W = 700, H = 300, pad = { top: 20, right: 60, bottom: 60, left: 60 };
  let plotW = W - pad.left - pad.right;
  let plotH = H - pad.top - pad.bottom;

  let rounds = data.map(function (d) { return d.round; });
  let rhos = data.map(function (d) { return d.rho_bar; });

  // Collect all agent roles
  let roles = {};
  data.forEach(function (d) {
    if (d.rho_i) Object.keys(d.rho_i).forEach(function (r) { roles[r] = true; });
  });
  let roleList = Object.keys(roles).sort();

  let allVals = rhos.filter(function (v) { return v != null; });
  data.forEach(function (d) {
    if (d.rho_i) Object.values(d.rho_i).forEach(function (v) { if (v != null) allVals.push(v); });
  });
  if (allVals.length === 0) return '';
  let yMin = Math.max(0, Math.min.apply(null, allVals) - 0.05);
  let yMax = Math.min(1, Math.max.apply(null, allVals) + 0.05);
  if (yMax - yMin < 0.1) { yMin = Math.max(0, yMin - 0.05); yMax = Math.min(1, yMax + 0.05); }

  let xMin = Math.min.apply(null, rounds);
  let xMax = Math.max.apply(null, rounds);
  if (xMax === xMin) xMax = xMin + 1;

  /** Map a round number to its x pixel coordinate. */
  function xPos(r) { return pad.left + (r - xMin) / (xMax - xMin) * plotW; }
  /** Map a metric value to its y pixel coordinate. */
  function yPos(v) { return pad.top + (1 - (v - yMin) / (yMax - yMin)) * plotH; }

  let svg = '<svg class="chart" data-testid="crit-chart" width="' + W + '" height="' + H + '" xmlns="http://www.w3.org/2000/svg">';
  svg += buildCRITAxes(W, pad, plotH, rounds, yMin, yMax, xPos, yPos);
  svg += buildRhoBarPath(data, rounds, rhos, xPos, yPos);
  svg += buildAgentPaths(data, rounds, roleList, xPos, yPos, agentLabel);

  // Legend below chart
  let legendY = pad.top + plotH + 35;
  let dashPatterns = ['5,5', '10,5', '2,5', '15,5,5,5'];
  svg += '<line x1="' + pad.left + '" y1="' + legendY + '" x2="' + (pad.left + 20) + '" y2="' + legendY + '" stroke="#000" stroke-width="2" />';
  svg += '<text x="' + (pad.left + 23) + '" y="' + (legendY + 4) + '" fill="#000" font-size="10">\u03c1</text>';
  let lx = pad.left + 80;
  roleList.forEach(function (role, ri) {
    let dash = dashPatterns[ri % dashPatterns.length];
    svg += '<line x1="' + lx + '" y1="' + legendY + '" x2="' + (lx + 20) + '" y2="' + legendY + '" stroke="#000" stroke-width="1.5" stroke-dasharray="' + dash + '" />';
    svg += '<text x="' + (lx + 23) + '" y="' + (legendY + 4) + '" fill="#000" font-size="10">' + agentLabel(role) + '</text>';
    lx += 80;
  });

  svg += '</svg>';
  return svg;
}
