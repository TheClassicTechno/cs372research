/**
 * utils/dom.js
 *
 * DOM utility helpers for HTML escaping and element clearing.
 */

/** Escape a string for safe insertion into HTML. */
export function esc(s) {
  if (s == null) return '';
  return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;').replace(/"/g, '&quot;');
}

/** Remove all child content from a DOM element. */
export function clear(el) {
  el.innerHTML = '';
}
