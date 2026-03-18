/**
 * templates/card.js
 *
 * HTML template for the generic collapsible card component.
 */

import { registerTemplate } from '../utils/templates.js';

/**
 * Card template.
 *
 * Slots:
 *   {{openClass}} - ' open' or '' to control initial collapsed/expanded state
 *   {{title}}     - Card header text (auto-escaped)
 *   {{{body}}}    - Card body HTML (raw, already escaped by caller)
 */
registerTemplate('card', [
  '<div class="card{{openClass}}" data-testid="card">',
  '  <div class="card-header">',
  '    <span>{{title}}</span>',
  '    <span class="arrow">&#9654;</span>',
  '  </div>',
  '  <div class="card-body">{{{body}}}</div>',
  '</div>',
].join(''));
