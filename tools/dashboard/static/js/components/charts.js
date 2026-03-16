import { fmt } from '../utils/format.js';

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

  function xPos(r) { return pad.left + (r - xMin) / (xMax - xMin) * plotW; }
  function yPos(v) { return pad.top + (1 - (v - yMin) / (yMax - yMin)) * plotH; }

  let svg = '<svg class="chart" width="' + W + '" height="' + H + '" xmlns="http://www.w3.org/2000/svg">';

  // Grid lines
  for (let g = 0; g <= 4; g++) {
    let gv = yMin + (yMax - yMin) * g / 4;
    let gy = yPos(gv);
    svg += '<line x1="' + pad.left + '" y1="' + gy + '" x2="' + (W - pad.right) + '" y2="' + gy + '" stroke="#eee" />';
    svg += '<text x="' + (pad.left - 5) + '" y="' + (gy + 4) + '" text-anchor="end" fill="#666">' + gv.toFixed(2) + '</text>';
  }

  // X axis labels
  for (let i = 0; i < rounds.length; i++) {
    svg += '<text x="' + xPos(rounds[i]) + '" y="' + (H - pad.bottom + 20) + '" text-anchor="middle" fill="#666">R' + rounds[i] + '</text>';
  }

  // Beta line (solid)
  let betaPoints = [];
  for (let i = 0; i < data.length; i++) {
    if (betas[i] != null) betaPoints.push(xPos(rounds[i]) + ',' + yPos(betas[i]));
  }
  if (betaPoints.length > 1) {
    svg += '<polyline points="' + betaPoints.join(' ') + '" fill="none" stroke="#000" stroke-width="2" />';
  }
  for (let i = 0; i < data.length; i++) {
    if (betas[i] != null) {
      svg += '<circle cx="' + xPos(rounds[i]) + '" cy="' + yPos(betas[i]) + '" r="3" fill="#000"><title>Beta: ' + fmt(betas[i]) + '</title></circle>';
    }
  }

  // rho_bar line (dashed)
  let rhoPoints = [];
  for (let i = 0; i < data.length; i++) {
    if (rhos[i] != null) rhoPoints.push(xPos(rounds[i]) + ',' + yPos(rhos[i]));
  }
  if (rhoPoints.length > 1) {
    svg += '<polyline points="' + rhoPoints.join(' ') + '" fill="none" stroke="#000" stroke-width="2" stroke-dasharray="8,4" />';
  }
  for (let i = 0; i < data.length; i++) {
    if (rhos[i] != null) {
      svg += '<circle cx="' + xPos(rounds[i]) + '" cy="' + yPos(rhos[i]) + '" r="3" fill="none" stroke="#000" stroke-width="1.5"><title>\u03c1: ' + fmt(rhos[i]) + '</title></circle>';
    }
  }

  // Legend
  svg += '<line x1="' + (W - pad.right + 5) + '" y1="' + (pad.top + 10) + '" x2="' + (W - pad.right + 25) + '" y2="' + (pad.top + 10) + '" stroke="#000" stroke-width="2" />';
  svg += '<text x="' + (W - pad.right + 28) + '" y="' + (pad.top + 14) + '" fill="#000" font-size="10">beta</text>';
  svg += '<line x1="' + (W - pad.right + 5) + '" y1="' + (pad.top + 28) + '" x2="' + (W - pad.right + 25) + '" y2="' + (pad.top + 28) + '" stroke="#000" stroke-width="2" stroke-dasharray="8,4" />';
  svg += '<text x="' + (W - pad.right + 28) + '" y="' + (pad.top + 32) + '" fill="#000" font-size="10">rho</text>';

  svg += '</svg>';
  return svg;
}

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
  let dashPatterns = ['5,5', '10,5', '2,5', '15,5,5,5'];

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

  function xPos(r) { return pad.left + (r - xMin) / (xMax - xMin) * plotW; }
  function yPos(v) { return pad.top + (1 - (v - yMin) / (yMax - yMin)) * plotH; }

  let svg = '<svg class="chart" width="' + W + '" height="' + H + '" xmlns="http://www.w3.org/2000/svg">';

  // Grid
  for (let g = 0; g <= 4; g++) {
    let gv = yMin + (yMax - yMin) * g / 4;
    let gy = yPos(gv);
    svg += '<line x1="' + pad.left + '" y1="' + gy + '" x2="' + (W - pad.right) + '" y2="' + gy + '" stroke="#eee" />';
    svg += '<text x="' + (pad.left - 5) + '" y="' + (gy + 4) + '" text-anchor="end" fill="#666">' + gv.toFixed(2) + '</text>';
  }
  for (let i = 0; i < rounds.length; i++) {
    svg += '<text x="' + xPos(rounds[i]) + '" y="' + (pad.top + plotH + 20) + '" text-anchor="middle" fill="#666">R' + rounds[i] + '</text>';
  }

  // rho_bar line (solid)
  let rhoPoints = [];
  for (let i = 0; i < data.length; i++) {
    if (rhos[i] != null) rhoPoints.push(xPos(rounds[i]) + ',' + yPos(rhos[i]));
  }
  if (rhoPoints.length > 1) {
    svg += '<polyline points="' + rhoPoints.join(' ') + '" fill="none" stroke="#000" stroke-width="2" />';
  }
  for (let i = 0; i < data.length; i++) {
    if (rhos[i] != null) {
      svg += '<circle cx="' + xPos(rounds[i]) + '" cy="' + yPos(rhos[i]) + '" r="3" fill="#000"><title>\u03c1: ' + fmt(rhos[i]) + '</title></circle>';
    }
  }

  // Per-agent lines (dashed)
  roleList.forEach(function (role, ri) {
    let dash = dashPatterns[ri % dashPatterns.length];
    let pts = [];
    for (let i = 0; i < data.length; i++) {
      const v = data[i].rho_i ? data[i].rho_i[role] : null;
      if (v != null) pts.push(xPos(rounds[i]) + ',' + yPos(v));
    }
    if (pts.length > 1) {
      svg += '<polyline points="' + pts.join(' ') + '" fill="none" stroke="#000" stroke-width="1.5" stroke-dasharray="' + dash + '" />';
    }
    for (let i = 0; i < data.length; i++) {
      const v = data[i].rho_i ? data[i].rho_i[role] : null;
      if (v != null) {
        svg += '<circle cx="' + xPos(rounds[i]) + '" cy="' + yPos(v) + '" r="2" fill="none" stroke="#000"><title>' + agentLabel(role) + ' rho_i: ' + fmt(v) + '</title></circle>';
      }
    }
  });

  // Legend below chart
  let legendY = pad.top + plotH + 35;
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
