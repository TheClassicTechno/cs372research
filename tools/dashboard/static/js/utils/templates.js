/**
 * utils/templates.js
 *
 * Minimal HTML template engine for the dashboard.
 *
 * Templates are plain HTML strings with {{slot}} placeholders.
 * The engine loads templates from static/js/templates/*.html at startup,
 * then exposes a render(name, slots) function that returns an HTML string
 * with all placeholders filled.
 *
 * Design constraints:
 *   - No frameworks, no build step.
 *   - Templates live in the templates/ directory as importable JS modules
 *     exporting a single string constant (avoids fetch/async for static HTML).
 *   - Slot values are auto-escaped by default; use {{{slot}}} for raw HTML.
 *   - Components remain pure functions: data in -> HTML string out.
 */

import { esc } from './dom.js';

/** @type {Map<string, string>} Registered template strings keyed by name. */
const registry = new Map();

/**
 * Register a named template string.
 *
 * @param {string} name  - Template identifier (e.g. 'card', 'event-card').
 * @param {string} html  - Template markup with {{slot}} / {{{rawSlot}}} placeholders.
 */
export function registerTemplate(name, html) {
  registry.set(name, html);
}

/**
 * Render a registered template with the given slot values.
 *
 * Placeholder syntax:
 *   {{key}}   - auto-escaped via esc()
 *   {{{key}}} - inserted raw (for pre-escaped HTML content)
 *
 * Missing keys are replaced with an empty string.
 *
 * @param {string} name   - Template name previously registered.
 * @param {Object<string, string>} slots - Key-value pairs for placeholders.
 * @returns {string} Rendered HTML string.
 */
export function renderTemplate(name, slots) {
  const tmpl = registry.get(name);
  if (!tmpl) {
    throw new Error('Template not found: ' + name);
  }

  // First pass: replace raw (triple-brace) placeholders.
  let result = tmpl.replace(/\{\{\{(\w+)\}\}\}/g, function (_match, key) {
    return slots[key] != null ? String(slots[key]) : '';
  });

  // Second pass: replace escaped (double-brace) placeholders.
  result = result.replace(/\{\{(\w+)\}\}/g, function (_match, key) {
    return slots[key] != null ? esc(String(slots[key])) : '';
  });

  return result;
}
