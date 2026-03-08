import { fmt } from '../utils/format.js';

export function buildPIDChart(data) {
  var W = 700, H = 300, pad = { top: 20, right: 60, bottom: 40, left: 60 };
  var plotW = W - pad.left - pad.right;
  var plotH = H - pad.top - pad.bottom;

  var rounds = data.map(function (d) { return d.round; });
  var betas = data.map(function (d) { return d.beta_new != null ? d.beta_new : d.beta_in; });
  var rhos = data.map(function (d) { return d.rho_bar; });

  var allVals = betas.concat(rhos).filter(function (v) { return v != null; });
  if (allVals.length === 0) return '';
  var yMin = Math.max(0, Math.min.apply(null, allVals) - 0.05);
  var yMax = Math.min(1, Math.max.apply(null, allVals) + 0.05);
  if (yMax - yMin < 0.1) { yMin = Math.max(0, yMin - 0.05); yMax = Math.min(1, yMax + 0.05); }

  var xMin = Math.min.apply(null, rounds);
  var xMax = Math.max.apply(null, rounds);
  if (xMax === xMin) xMax = xMin + 1;

  function xPos(r) { return pad.left + (r - xMin) / (xMax - xMin) * plotW; }
  function yPos(v) { return pad.top + (1 - (v - yMin) / (yMax - yMin)) * plotH; }

  var svg = '<svg class="chart" width="' + W + '" height="' + H + '" xmlns="http://www.w3.org/2000/svg">';

  // Grid lines
  for (var g = 0; g <= 4; g++) {
    var gv = yMin + (yMax - yMin) * g / 4;
    var gy = yPos(gv);
    svg += '<line x1="' + pad.left + '" y1="' + gy + '" x2="' + (W - pad.right) + '" y2="' + gy + '" stroke="#eee" />';
    svg += '<text x="' + (pad.left - 5) + '" y="' + (gy + 4) + '" text-anchor="end" fill="#666">' + gv.toFixed(2) + '</text>';
  }

  // X axis labels
  for (var i = 0; i < rounds.length; i++) {
    svg += '<text x="' + xPos(rounds[i]) + '" y="' + (H - pad.bottom + 20) + '" text-anchor="middle" fill="#666">R' + rounds[i] + '</text>';
  }

  // Beta line (solid)
  var betaPoints = [];
  for (var i = 0; i < data.length; i++) {
    if (betas[i] != null) betaPoints.push(xPos(rounds[i]) + ',' + yPos(betas[i]));
  }
  if (betaPoints.length > 1) {
    svg += '<polyline points="' + betaPoints.join(' ') + '" fill="none" stroke="#000" stroke-width="2" />';
  }
  for (var i = 0; i < data.length; i++) {
    if (betas[i] != null) {
      svg += '<circle cx="' + xPos(rounds[i]) + '" cy="' + yPos(betas[i]) + '" r="3" fill="#000"><title>Beta: ' + fmt(betas[i]) + '</title></circle>';
    }
  }

  // rho_bar line (dashed)
  var rhoPoints = [];
  for (var i = 0; i < data.length; i++) {
    if (rhos[i] != null) rhoPoints.push(xPos(rounds[i]) + ',' + yPos(rhos[i]));
  }
  if (rhoPoints.length > 1) {
    svg += '<polyline points="' + rhoPoints.join(' ') + '" fill="none" stroke="#000" stroke-width="2" stroke-dasharray="8,4" />';
  }
  for (var i = 0; i < data.length; i++) {
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

export function buildCRITChart(data) {
  var W = 700, H = 300, pad = { top: 20, right: 60, bottom: 60, left: 60 };
  var plotW = W - pad.left - pad.right;
  var plotH = H - pad.top - pad.bottom;

  var rounds = data.map(function (d) { return d.round; });
  var rhos = data.map(function (d) { return d.rho_bar; });

  // Collect all agent roles
  var roles = {};
  data.forEach(function (d) {
    if (d.rho_i) Object.keys(d.rho_i).forEach(function (r) { roles[r] = true; });
  });
  var roleList = Object.keys(roles).sort();
  var dashPatterns = ['5,5', '10,5', '2,5', '15,5,5,5'];

  var allVals = rhos.filter(function (v) { return v != null; });
  data.forEach(function (d) {
    if (d.rho_i) Object.values(d.rho_i).forEach(function (v) { if (v != null) allVals.push(v); });
  });
  if (allVals.length === 0) return '';
  var yMin = Math.max(0, Math.min.apply(null, allVals) - 0.05);
  var yMax = Math.min(1, Math.max.apply(null, allVals) + 0.05);
  if (yMax - yMin < 0.1) { yMin = Math.max(0, yMin - 0.05); yMax = Math.min(1, yMax + 0.05); }

  var xMin = Math.min.apply(null, rounds);
  var xMax = Math.max.apply(null, rounds);
  if (xMax === xMin) xMax = xMin + 1;

  function xPos(r) { return pad.left + (r - xMin) / (xMax - xMin) * plotW; }
  function yPos(v) { return pad.top + (1 - (v - yMin) / (yMax - yMin)) * plotH; }

  var svg = '<svg class="chart" width="' + W + '" height="' + H + '" xmlns="http://www.w3.org/2000/svg">';

  // Grid
  for (var g = 0; g <= 4; g++) {
    var gv = yMin + (yMax - yMin) * g / 4;
    var gy = yPos(gv);
    svg += '<line x1="' + pad.left + '" y1="' + gy + '" x2="' + (W - pad.right) + '" y2="' + gy + '" stroke="#eee" />';
    svg += '<text x="' + (pad.left - 5) + '" y="' + (gy + 4) + '" text-anchor="end" fill="#666">' + gv.toFixed(2) + '</text>';
  }
  for (var i = 0; i < rounds.length; i++) {
    svg += '<text x="' + xPos(rounds[i]) + '" y="' + (pad.top + plotH + 20) + '" text-anchor="middle" fill="#666">R' + rounds[i] + '</text>';
  }

  // rho_bar line (solid)
  var rhoPoints = [];
  for (var i = 0; i < data.length; i++) {
    if (rhos[i] != null) rhoPoints.push(xPos(rounds[i]) + ',' + yPos(rhos[i]));
  }
  if (rhoPoints.length > 1) {
    svg += '<polyline points="' + rhoPoints.join(' ') + '" fill="none" stroke="#000" stroke-width="2" />';
  }
  for (var i = 0; i < data.length; i++) {
    if (rhos[i] != null) {
      svg += '<circle cx="' + xPos(rounds[i]) + '" cy="' + yPos(rhos[i]) + '" r="3" fill="#000"><title>\u03c1: ' + fmt(rhos[i]) + '</title></circle>';
    }
  }

  // Per-agent lines (dashed)
  roleList.forEach(function (role, ri) {
    var dash = dashPatterns[ri % dashPatterns.length];
    var pts = [];
    for (var i = 0; i < data.length; i++) {
      var v = data[i].rho_i ? data[i].rho_i[role] : null;
      if (v != null) pts.push(xPos(rounds[i]) + ',' + yPos(v));
    }
    if (pts.length > 1) {
      svg += '<polyline points="' + pts.join(' ') + '" fill="none" stroke="#000" stroke-width="1.5" stroke-dasharray="' + dash + '" />';
    }
    for (var i = 0; i < data.length; i++) {
      var v = data[i].rho_i ? data[i].rho_i[role] : null;
      if (v != null) {
        svg += '<circle cx="' + xPos(rounds[i]) + '" cy="' + yPos(v) + '" r="2" fill="none" stroke="#000"><title>' + role + ' rho_i: ' + fmt(v) + '</title></circle>';
      }
    }
  });

  // Legend below chart
  var legendY = pad.top + plotH + 35;
  svg += '<line x1="' + pad.left + '" y1="' + legendY + '" x2="' + (pad.left + 20) + '" y2="' + legendY + '" stroke="#000" stroke-width="2" />';
  svg += '<text x="' + (pad.left + 23) + '" y="' + (legendY + 4) + '" fill="#000" font-size="10">\u03c1</text>';
  var lx = pad.left + 80;
  roleList.forEach(function (role, ri) {
    var dash = dashPatterns[ri % dashPatterns.length];
    svg += '<line x1="' + lx + '" y1="' + legendY + '" x2="' + (lx + 20) + '" y2="' + legendY + '" stroke="#000" stroke-width="1.5" stroke-dasharray="' + dash + '" />';
    svg += '<text x="' + (lx + 23) + '" y="' + (legendY + 4) + '" fill="#000" font-size="10">' + role + '</text>';
    lx += 80;
  });

  svg += '</svg>';
  return svg;
}
