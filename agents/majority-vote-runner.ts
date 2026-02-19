/**
 * MajorityVoteRunner: 3+ role agents each propose Action, aggregator merges by vote.
 * Logs disagreements for auditability.
 */

import OpenAI from 'openai';
import * as fs from 'fs';
import * as path from 'path';
import type {
  Observation,
  Action,
  AgentTrace,
  Order,
  Claim,
  DebateTurn,
  AgentRole,
} from './types';
import { buildRoleObservationContext, ROLE_PROMPTS } from './prompts';

function getOpenAI(): OpenAI {
  return new OpenAI({ apiKey: process.env.OPENAI_API_KEY ?? 'sk-dummy' });
}

// -----------------------------------------------------------------------------
// TYPES
// -----------------------------------------------------------------------------

interface RoleProposal {
  role: AgentRole;
  action: Action;
  rawResponse: string;
}

// -----------------------------------------------------------------------------
// PARSING
// -----------------------------------------------------------------------------

function parseRoleResponse(text: string): Partial<Action> {
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/) ?? [null, text];
  const jsonStr = (jsonMatch[1] ?? text).trim();
  try {
    const parsed = JSON.parse(jsonStr) as Record<string, unknown>;
    const orders = (parsed.orders as Array<{ ticker: string; side: string; size: number }>) ?? [];
    return {
      orders: orders.map((o) => ({
        ticker: o.ticker,
        side: (o.side === 'sell' ? 'sell' : 'buy') as 'buy' | 'sell',
        size: Number(o.size) || 0,
        type: 'market' as const,
      })),
      justification: String(parsed.justification ?? ''),
      confidence: Math.max(0, Math.min(1, Number(parsed.confidence) ?? 0.5)),
      claims: ((parsed.claims as Claim[]) ?? []).map((c) => ({
        ...c,
        pearl_level: (c.pearl_level ?? 'L1') as 'L1' | 'L2' | 'L3',
      })),
    };
  } catch {
    return { orders: [], justification: 'Parse failed', confidence: 0.5, claims: [] };
  }
}

// -----------------------------------------------------------------------------
// AGGREGATION
// -----------------------------------------------------------------------------

/** Vote on direction per ticker: 'buy' | 'sell' | 'hold' */
function getVoteDirection(proposals: RoleProposal[], ticker: string): 'buy' | 'sell' | 'hold' {
  const votes: ('buy' | 'sell')[] = [];
  for (const p of proposals) {
    for (const o of p.action.orders) {
      if (o.ticker === ticker && o.size > 0) {
        votes.push(o.side);
      }
    }
  }
  const buys = votes.filter((v) => v === 'buy').length;
  const sells = votes.filter((v) => v === 'sell').length;
  if (buys > sells) return 'buy';
  if (sells > buys) return 'sell';
  return 'hold';
}

/** Median size across proposals for a ticker+side */
function getMedianSize(proposals: RoleProposal[], ticker: string, side: 'buy' | 'sell'): number {
  const sizes = proposals
    .flatMap((p) => p.action.orders)
    .filter((o) => o.ticker === ticker && o.side === side && o.size > 0)
    .map((o) => o.size);
  if (sizes.length === 0) return 0;
  sizes.sort((a, b) => a - b);
  const mid = Math.floor(sizes.length / 2);
  return sizes.length % 2 ? sizes[mid]! : (sizes[mid - 1]! + sizes[mid]!) / 2;
}

/** Aggregate proposals into final Action. Log strongest objection. */
function aggregate(
  proposals: RoleProposal[],
  obs: Observation
): { action: Action; strongestObjection: string | undefined; disagreements: string[] } {
  const tickers = new Set<string>();
  for (const p of proposals) {
    for (const o of p.action.orders) tickers.add(o.ticker);
  }

  const orders: Order[] = [];
  const disagreements: string[] = [];

  for (const ticker of tickers) {
    const dir = getVoteDirection(proposals, ticker);
    if (dir === 'hold') continue;
    const size = Math.round(getMedianSize(proposals, ticker, dir));
    if (size > 0) {
      orders.push({ ticker, side: dir, size, type: 'market' });
    }

    // Log disagreement if not unanimous
    const votes = proposals.map((p) => {
      const o = p.action.orders.find((x) => x.ticker === ticker);
      return o ? `${p.role}:${o.side}${o.size}` : `${p.role}:hold`;
    });
    const unique = [...new Set(votes)];
    if (unique.length > 1) {
      disagreements.push(`${ticker}: ${votes.join(' vs ')}`);
    }
  }

  // Strongest objection: lowest confidence proposal that disagreed
  const avgConf = proposals.reduce((s, p) => s + p.action.confidence, 0) / proposals.length;
  const dissenter = proposals.find((p) => p.action.confidence < avgConf - 0.1);
  const strongestObjection = dissenter
    ? `[${dissenter.role}] ${dissenter.action.justification} (confidence: ${dissenter.action.confidence})`
    : undefined;

  // Merge justifications
  const justification = proposals
    .map((p) => `[${p.role}] ${p.action.justification}`)
    .join('\n');

  // Merge claims (dedupe by claim_text)
  const seen = new Set<string>();
  const claims: Claim[] = [];
  for (const p of proposals) {
    for (const c of p.action.claims) {
      if (!seen.has(c.claim_text)) {
        seen.add(c.claim_text);
        claims.push(c);
      }
    }
  }

  const confidence =
    proposals.reduce((s, p) => s + p.action.confidence, 0) / proposals.length;

  return {
    action: {
      orders,
      justification,
      confidence,
      claims,
    },
    strongestObjection,
    disagreements,
  };
}

// -----------------------------------------------------------------------------
// RUNNER
// -----------------------------------------------------------------------------

export interface MajorityVoteRunnerOptions {
  roles?: AgentRole[];
  traceDir?: string;
  model?: string;
  mock?: boolean;
}

export class MajorityVoteRunner {
  private roles: AgentRole[];
  private traceDir: string;
  private model: string;
  private mock: boolean;

  constructor(options: MajorityVoteRunnerOptions = {}) {
    this.roles = options.roles ?? ['macro', 'value', 'risk'];
    this.traceDir = options.traceDir ?? path.join(process.cwd(), 'traces');
    this.model = options.model ?? 'gpt-4o-mini';
    this.mock = options.mock ?? false;
  }

  async run(observation: Observation): Promise<{ action: Action; trace: AgentTrace }> {
    const context = buildRoleObservationContext(observation);
    const proposals: RoleProposal[] = [];

    for (const role of this.roles) {
      const systemPrompt = ROLE_PROMPTS[role] ?? ROLE_PROMPTS.macro;
      const userPrompt = `${context}\n\nOutput JSON: { orders: [...], justification: "...", confidence: 0-1, claims: [...] }`;

      let rawResponse: string;
      if (this.mock) {
        rawResponse = JSON.stringify(this.getMockProposal(role, observation));
      } else {
        const openai = getOpenAI();
        const completion = await openai.chat.completions.create({
          model: this.model,
          messages: [
            { role: 'system', content: systemPrompt },
            { role: 'user', content: userPrompt },
          ],
          temperature: 0.3,
        });
        rawResponse = completion.choices[0]?.message?.content ?? '{}';
      }

      const partial = parseRoleResponse(rawResponse);
      proposals.push({
        role,
        action: {
          orders: partial.orders ?? [],
          justification: partial.justification ?? '',
          confidence: partial.confidence ?? 0.5,
          claims: partial.claims ?? [],
        },
        rawResponse,
      });
    }

    const { action, strongestObjection, disagreements } = aggregate(proposals, observation);

    const debateTurns: DebateTurn[] = proposals.map((p, i) => ({
      round: 0,
      agent_id: `agent_${i}`,
      role: p.role,
      proposal: p.action,
    }));

    const trace: AgentTrace = {
      observation_timestamp: observation.timestamp,
      architecture: 'majority_vote',
      what_i_saw: context.slice(0, 200) + '...',
      hypothesis: `Majority vote across ${this.roles.join(', ')} agents`,
      decision: action.orders.length
        ? action.orders.map((o) => `${o.side} ${o.size} ${o.ticker}`).join('; ')
        : 'Hold',
      risks_or_falsifiers: proposals.map((p) => p.action.justification).join(' | '),
      strongest_objection: strongestObjection,
      debate_turns: debateTurns,
      action,
      logged_at: new Date().toISOString(),
    };

    if (disagreements.length > 0) {
      (trace as Record<string, unknown>).disagreements = disagreements;
    }

    await this.logTrace(trace, observation.timestamp);
    return { action, trace };
  }

  private async logTrace(trace: AgentTrace, timestamp: string): Promise<void> {
    const dir = this.traceDir;
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const safeTs = timestamp.replace(/[:.]/g, '-');
    const filename = `majority_vote_${safeTs}.json`;
    fs.writeFileSync(path.join(dir, filename), JSON.stringify(trace, null, 2), 'utf-8');
    console.log(`[MajorityVote] Trace written to ${path.join(dir, filename)}`);
  }

  private getMockProposal(role: AgentRole, obs: Observation): Action {
    const returns = obs.market_state.returns ?? {};
    const tickers = obs.universe;
    const t = tickers[0] ?? 'AAPL';
    const r = returns[t] ?? 0;

    // Role-specific bias
    const bias =
      role === 'macro' ? 0.02 : role === 'value' ? -0.01 : role === 'risk' ? -0.03 : 0;
    const threshold = r + bias;
    const orders =
      threshold > 0.02 ? [{ ticker: t, side: 'buy' as const, size: 5 + Math.floor(Math.random() * 10) }] :
      threshold < -0.02 ? [{ ticker: t, side: 'sell' as const, size: 5 + Math.floor(Math.random() * 10) }] :
      [];

    return {
      orders,
      justification: `[${role} mock] threshold=${(threshold * 100).toFixed(2)}%`,
      confidence: 0.4 + Math.random() * 0.3,
      claims: [
        {
          claim_text: `${t} return associated with ${role} signal`,
          pearl_level: 'L1',
          variables: [t, 'return'],
          confidence: 0.5,
        },
      ],
    };
  }
}
