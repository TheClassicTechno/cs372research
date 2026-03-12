import { esc } from '../utils/dom.js';

export function buildTree(items, experiment, runId) {
  let h = '<ul>';
  for (let i = 0; i < items.length; i++) {
    let item = items[i];
    if (item.type === 'dir') {
      h += '<li><span class="dir-toggle" data-action="toggle-dir">[+] ' + esc(item.name) + '/</span>';
      if (item.children) h += buildTree(item.children, experiment, runId);
      h += '</li>';
    } else {
      h += '<li><span class="file-link" data-action="load-file" data-experiment="' + esc(experiment) + '" data-run-id="' + esc(runId) + '" data-path="' + esc(item.path) + '">' + esc(item.name) + '</span>';
      if (item.size_bytes != null) h += ' <span style="color:#999">(' + item.size_bytes + 'b)</span>';
      h += '</li>';
    }
  }
  h += '</ul>';
  return h;
}
