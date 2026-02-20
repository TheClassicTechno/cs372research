/**
 * System and user prompts for trading agents.
 * Includes anti-failure-mode rules: forced uncertainty, causal claim formatting,
 * trap-awareness, disagreement preservation.
 */

import type { Observation } from './types';

// -----------------------------------------------------------------------------
// ANTI-FAILURE-MODE RULES (from proposal)
// -----------------------------------------------------------------------------

export const CAUSAL_CLAIM_FORMAT = `
Every causal claim MUST be one of:
- L1 (Association): "X is associated with Y"
- L2 (Intervention): "If we do X, Y changes"
- L3 (Counterfactual): "Had X not occurred, Y would..."
`;

export const FORCED_UNCERTAINTY = `
You MUST include a "what would falsify this" line: what evidence or event would change your mind?
`;

export const TRAP_AWARENESS = `
Consider: what confounder could explain this? (Markets often have reverse causation and confounding.)
`;

// -----------------------------------------------------------------------------
// SINGLE AGENT PROMPT
// -----------------------------------------------------------------------------

export function buildSingleAgentPrompt(obs: Observation): string {
  const marketSummary = Object.entries(obs.market_state.prices)
    .map(([t, p]) => `${t}: $${p}`)
    .join(', ');
  const returnsStr = obs.market_state.returns
    ? Object.entries(obs.market_state.returns)
        .map(([t, r]) => `${t}: ${(r * 100).toFixed(2)}%`)
        .join(', ')
    : 'N/A';
  const portfolioStr = `Cash: $${obs.portfolio_state.cash}, Positions: ${JSON.stringify(obs.portfolio_state.positions)}`;

  return `You are a trading agent. Given this observation, produce a trading decision.

## Observation
- Timestamp: ${obs.timestamp}
- Universe: ${obs.universe.join(', ')}
- Prices: ${marketSummary}
- Returns: ${returnsStr}
- Portfolio: ${portfolioStr}
${obs.text_context ? `- Text context (news/earnings): ${obs.text_context}` : ''}

## Your task
Output a JSON object with:
1. what_i_saw: Brief summary of what you observed
2. hypothesis: Your hypothesis about market direction
3. decision: Your trading decision and why
4. risks_or_falsifiers: What would falsify your view? (REQUIRED)
5. orders: Array of {ticker, side: "buy"|"sell", size}
6. justification: Free-text explanation
7. confidence: 0-1 score
8. claims: Array of causal claims, each with: claim_text, pearl_level (L1|L2|L3), variables, assumptions, confidence

${CAUSAL_CLAIM_FORMAT}
${FORCED_UNCERTAINTY}
${TRAP_AWARENESS}

If you hold (no trade), use orders: [].
Respond with valid JSON only.`;
}

// -----------------------------------------------------------------------------
// ROLE-SPECIFIC PROMPTS (for majority vote)
// -----------------------------------------------------------------------------

export const ROLE_PROMPTS: Record<string, string> = {
  macro: `You are the MACRO agent. Focus on interest rates, Fed policy, inflation, and broad economic signals.
  Output: orders, justification, confidence (0-1), claims (causal claims with pearl_level).
  ${FORCED_UNCERTAINTY}`,

  value: `You are the VALUE agent. Focus on fundamentals, earnings, valuation multiples, and long-term value.
  Output: orders, justification, confidence (0-1), claims (causal claims with pearl_level).
  ${FORCED_UNCERTAINTY}`,

  risk: `You are the RISK agent. Focus on volatility, drawdowns, position sizing, and tail risks.
  Output: orders, justification, confidence (0-1), claims (causal claims with pearl_level).
  ${FORCED_UNCERTAINTY}`,

  technical: `You are the TECHNICAL agent. Focus on price action, momentum, support/resistance.
  Output: orders, justification, confidence (0-1), claims (causal claims with pearl_level).
  ${FORCED_UNCERTAINTY}`,
};

// -----------------------------------------------------------------------------
// DEBATE PROMPTS (Round 0 proposal, Round 1 critique, Round 2 revision, Judge)
// -----------------------------------------------------------------------------

export function buildDebateCritiquePrompt(
  role: string,
  context: string,
  allProposals: Array<{ role: string; proposal: string }>,
  myProposal: string
): string {
  const others = allProposals.filter((p) => p.role !== role);
  const othersText = others
    .map((p) => `## ${p.role.toUpperCase()} agent proposed:\n${p.proposal}`)
    .join('\n\n');

  return `You are the ${role.toUpperCase()} agent in a multi-agent debate.

## Market context
${context}

## Your initial proposal
${myProposal}

## Other agents' proposals (you MUST critique at least 2)
${othersText}

## Your task
Write critiques with EXPLICIT OBJECTIONS. For at least 2 other agents, state:
1. What causal claim or assumption you disagree with
2. What confounder or alternative explanation they might have missed
3. What would falsify their view

Output JSON:
{
  "critiques": [
    { "target_role": "macro", "objection": "...", "falsifier": "..." },
    { "target_role": "value", "objection": "...", "falsifier": "..." }
  ],
  "self_critique": "What weakness in your own proposal are you now reconsidering?"
}`;
}

export function buildDebateRevisionPrompt(
  role: string,
  context: string,
  myProposal: string,
  critiquesReceived: Array<{ from_role: string; objection: string; falsifier?: string }>
): string {
  const critiquesText = critiquesReceived
    .map((c) => `- [${c.from_role}]: ${c.objection}${c.falsifier ? ` | Falsifier: ${c.falsifier}` : ''}`)
    .join('\n');

  return `You are the ${role.toUpperCase()} agent. You received critiques on your proposal.

## Market context
${context}

## Your original proposal
${myProposal}

## Critiques you received
${critiquesText}

## Your task
Revise your proposal. You may:
- Adjust orders (size, direction) if critiques changed your view
- Lower confidence if valid objections were raised
- Add or modify causal claims to address objections
- Or stand by your proposal if critiques don't change your reasoning

Output JSON:
{
  "orders": [{ "ticker": "...", "side": "buy"|"sell", "size": N }],
  "justification": "Revised explanation addressing critiques",
  "confidence": 0-1,
  "claims": [{ "claim_text": "...", "pearl_level": "L1"|"L2"|"L3", "variables": [...], "confidence": 0-1 }],
  "revision_notes": "What you changed and why"
}`;
}

export function buildJudgePrompt(
  context: string,
  revisions: Array<{ role: string; action: string; confidence: number }>,
  allCritiques: string
): string {
  const revisionsText = revisions
    .map((r) => `## ${r.role.toUpperCase()} (confidence: ${r.confidence.toFixed(2)})\n${r.action}`)
    .join('\n\n');

  return `You are the JUDGE/AGGREGATOR. Synthesize the debate into a final trading decision.

## Market context
${context}

## Revised proposals after debate
${revisionsText}

## Critiques exchanged during debate
${allCritiques}

## Your task
1. Decide final orders (you may agree with majority, side with a minority, or hold)
2. Write an AUDITED MEMO: brief summary of the debate, key disagreements, and why you chose this outcome
3. Record the STRONGEST OBJECTION even if you overruled it (disagreement preservation)
4. Output confidence 0-1

Output JSON:
{
  "orders": [{ "ticker": "...", "side": "buy"|"sell", "size": N }],
  "audited_memo": "Summary of debate, key disagreements, rationale for final decision",
  "strongest_objection": "The objection that was most compelling even if overruled",
  "confidence": 0-1,
  "claims": [/* merged causal claims from proposals */]
}`;
}

export function buildRoleObservationContext(obs: Observation): string {
  const marketSummary = Object.entries(obs.market_state.prices)
    .map(([t, p]) => `${t}: $${p}`)
    .join(', ');
  const returnsStr = obs.market_state.returns
    ? Object.entries(obs.market_state.returns)
        .map(([t, r]) => `${t}: ${(r * 100).toFixed(2)}%`)
        .join(', ')
    : 'N/A';
  return `Timestamp: ${obs.timestamp}
Universe: ${obs.universe.join(', ')}
Prices: ${marketSummary}
Returns: ${returnsStr}
Portfolio cash: $${obs.portfolio_state.cash}
Positions: ${JSON.stringify(obs.portfolio_state.positions)}
${obs.text_context ? `News/context: ${obs.text_context}` : ''}`;
}
