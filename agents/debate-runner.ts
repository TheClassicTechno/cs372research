/**
 * DebateRunner: Multi-round debate with proposals, critiques, revisions, and judge.
 * Round 0: each agent writes proposal + causal claims
 * Round 1: each agent critiques at least 2 others (explicit objections)
 * Round 2: each agent revises
 * Judge: decides final trades, writes audited memo, preserves strongest objection
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
import {
  buildRoleObservationContext,
  ROLE_PROMPTS,
  buildDebateCritiquePrompt,
  buildDebateRevisionPrompt,
  buildJudgePrompt,
} from './prompts';

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

interface CritiqueResult {
  role: AgentRole;
  critiques: Array<{ target_role: string; objection: string; falsifier?: string }>;
  self_critique?: string;
}

interface RevisionResult {
  role: AgentRole;
  action: Action;
  revision_notes?: string;
}

// -----------------------------------------------------------------------------
// PARSING
// -----------------------------------------------------------------------------

function parseActionResponse(text: string): Partial<Action> {
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

function parseCritiqueResponse(text: string): CritiqueResult {
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/) ?? [null, text];
  const jsonStr = (jsonMatch[1] ?? text).trim();
  try {
    const parsed = JSON.parse(jsonStr) as Record<string, unknown>;
    const critiques = (parsed.critiques as Array<{ target_role: string; objection: string; falsifier?: string }>) ?? [];
    return {
      role: '' as AgentRole, // set by caller
      critiques,
      self_critique: String(parsed.self_critique ?? ''),
    };
  } catch {
    return { role: '' as AgentRole, critiques: [] };
  }
}

function parseJudgeResponse(text: string): Partial<Action> & { audited_memo?: string; strongest_objection?: string } {
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
      justification: String(parsed.audited_memo ?? parsed.justification ?? ''),
      confidence: Math.max(0, Math.min(1, Number(parsed.confidence) ?? 0.5)),
      claims: ((parsed.claims as Claim[]) ?? []).map((c) => ({
        ...c,
        pearl_level: (c.pearl_level ?? 'L1') as 'L1' | 'L2' | 'L3',
      })),
      audited_memo: String(parsed.audited_memo ?? ''),
      strongest_objection: String(parsed.strongest_objection ?? ''),
    };
  } catch {
    return { orders: [], justification: '', confidence: 0.5, claims: [] };
  }
}

// -----------------------------------------------------------------------------
// AGGREGATION (vote on direction, median size - same as majority vote)
// -----------------------------------------------------------------------------

function getVoteDirection(proposals: RoleProposal[], ticker: string): 'buy' | 'sell' | 'hold' {
  const votes: ('buy' | 'sell')[] = [];
  for (const p of proposals) {
    for (const o of p.action.orders) {
      if (o.ticker === ticker && o.size > 0) votes.push(o.side);
    }
  }
  const buys = votes.filter((v) => v === 'buy').length;
  const sells = votes.filter((v) => v === 'sell').length;
  if (buys > sells) return 'buy';
  if (sells > buys) return 'sell';
  return 'hold';
}

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

function aggregateRevisions(revisions: RevisionResult[]): Action {
  const proposals = revisions.map((r) => ({ role: r.role, action: r.action }));
  const tickers = new Set<string>();
  for (const p of proposals) {
    for (const o of p.action.orders) tickers.add(o.ticker);
  }

  const orders: Order[] = [];
  for (const ticker of tickers) {
    const dir = getVoteDirection(proposals, ticker);
    if (dir === 'hold') continue;
    const size = Math.round(getMedianSize(proposals, ticker, dir));
    if (size > 0) orders.push({ ticker, side: dir, size, type: 'market' });
  }

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

  const confidence = proposals.reduce((s, p) => s + p.action.confidence, 0) / proposals.length;
  const justification = proposals.map((p) => `[${p.role}] ${p.action.justification}`).join('\n');

  return { orders, justification, confidence, claims };
}

// -----------------------------------------------------------------------------
// RUNNER
// -----------------------------------------------------------------------------

export interface DebateRunnerOptions {
  roles?: AgentRole[];
  traceDir?: string;
  model?: string;
  mock?: boolean;
}

export class DebateRunner {
  private roles: AgentRole[];
  private traceDir: string;
  private model: string;
  private mock: boolean;

  constructor(options: DebateRunnerOptions = {}) {
    this.roles = options.roles ?? ['macro', 'value', 'risk'];
    this.traceDir = options.traceDir ?? path.join(process.cwd(), 'traces');
    this.model = options.model ?? 'gpt-4o-mini';
    this.mock = options.mock ?? false;
  }

  async run(observation: Observation): Promise<{ action: Action; trace: AgentTrace }> {
    const context = buildRoleObservationContext(observation);
    const debateTurns: DebateTurn[] = [];

    // ---- Round 0: Proposals ----
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

      const partial = parseActionResponse(rawResponse);
      const action: Action = {
        orders: partial.orders ?? [],
        justification: partial.justification ?? '',
        confidence: partial.confidence ?? 0.5,
        claims: partial.claims ?? [],
      };
      proposals.push({ role, action, rawResponse });
      debateTurns.push({
        round: 0,
        agent_id: `agent_${role}`,
        role,
        proposal: action,
      });
    }

    // ---- Round 1: Critiques ----
    const allProposalsForCritique = proposals.map((p) => ({
      role: p.role,
      proposal: JSON.stringify({ orders: p.action.orders, justification: p.action.justification, claims: p.action.claims }),
    }));

    const critiqueResults: CritiqueResult[] = [];
    for (const p of proposals) {
      let rawCritique: string;
      if (this.mock) {
        rawCritique = JSON.stringify(this.getMockCritique(p.role, proposals));
      } else {
        const prompt = buildDebateCritiquePrompt(
          p.role,
          context,
          allProposalsForCritique,
          JSON.stringify({ orders: p.action.orders, justification: p.action.justification })
        );
        const openai = getOpenAI();
        const completion = await openai.chat.completions.create({
          model: this.model,
          messages: [
            { role: 'system', content: `You are the ${p.role} agent. Provide explicit, substantive critiques.` },
            { role: 'user', content: prompt },
          ],
          temperature: 0.4,
        });
        rawCritique = completion.choices[0]?.message?.content ?? '{}';
      }

      const parsed = parseCritiqueResponse(rawCritique);
      parsed.role = p.role;
      critiqueResults.push(parsed);

      const objections = parsed.critiques.flatMap((c) => [
        `${c.target_role}: ${c.objection}${c.falsifier ? ` [Falsifier: ${c.falsifier}]` : ''}`,
      ]);
      debateTurns.push({
        round: 1,
        agent_id: `agent_${p.role}`,
        role: p.role,
        critique: parsed.self_critique ?? objections.join('; '),
        objections,
      });
    }

    // ---- Round 2: Revisions ----
    const revisions: RevisionResult[] = [];
    for (const p of proposals) {
      const critiquesReceived = critiqueResults
        .flatMap((c) =>
          c.critiques
            .filter((x) => x.target_role === p.role)
            .map((x) => ({ from_role: c.role, objection: x.objection, falsifier: x.falsifier }))
        );

      let rawRevision: string;
      if (this.mock) {
        rawRevision = JSON.stringify(this.getMockRevision(p.role, p.action, observation));
      } else {
        const prompt = buildDebateRevisionPrompt(
          p.role,
          context,
          JSON.stringify({ orders: p.action.orders, justification: p.action.justification, claims: p.action.claims }),
          critiquesReceived
        );
        const openai = getOpenAI();
        const completion = await openai.chat.completions.create({
          model: this.model,
          messages: [
            { role: 'system', content: `You are the ${p.role} agent. Revise based on critiques.` },
            { role: 'user', content: prompt },
          ],
          temperature: 0.3,
        });
        rawRevision = completion.choices[0]?.message?.content ?? '{}';
      }

      const partial = parseActionResponse(rawRevision);
      let revisionNotes: string | undefined;
      try {
        const jsonStr = (rawRevision.match(/```(?:json)?\s*([\s\S]*?)```/)?.[1] ?? rawRevision).trim();
        const revParsed = JSON.parse(jsonStr) as { revision_notes?: string };
        revisionNotes = revParsed.revision_notes;
      } catch {
        /* ignore */
      }

      const action: Action = {
        orders: partial.orders ?? p.action.orders,
        justification: partial.justification ?? p.action.justification,
        confidence: partial.confidence ?? p.action.confidence,
        claims: partial.claims ?? p.action.claims,
      };
      revisions.push({
        role: p.role,
        action,
        revision_notes: revisionNotes,
      });
      debateTurns.push({
        round: 2,
        agent_id: `agent_${p.role}`,
        role: p.role,
        revision: action,
      });
    }

    // ---- Judge: Final decision ----
    const allCritiquesText = critiqueResults
      .flatMap((c) =>
        c.critiques.map((x) => `[${c.role} → ${x.target_role}]: ${x.objection}`)
      )
      .join('\n');

    const revisionsForJudge = revisions.map((r) => ({
      role: r.role,
      action: JSON.stringify({
        orders: r.action.orders,
        justification: r.action.justification,
        confidence: r.action.confidence,
      }),
      confidence: r.action.confidence,
    }));

    let finalAction: Action;
    if (this.mock) {
      finalAction = this.getMockJudgeDecision(revisions, observation);
    } else {
      const judgePrompt = buildJudgePrompt(context, revisionsForJudge, allCritiquesText);
      const openai = getOpenAI();
      const completion = await openai.chat.completions.create({
        model: this.model,
        messages: [
          { role: 'system', content: 'You are the Judge. Synthesize the debate and produce final orders with an audited memo.' },
          { role: 'user', content: judgePrompt },
        ],
        temperature: 0.2,
      });
      const rawJudge = completion.choices[0]?.message?.content ?? '{}';
      const judgeResult = parseJudgeResponse(rawJudge);

      const seen = new Set<string>();
      const claims: Claim[] = [...(judgeResult.claims ?? [])];
      for (const r of revisions) {
        for (const c of r.action.claims) {
          if (!seen.has(c.claim_text)) {
            seen.add(c.claim_text);
            claims.push(c);
          }
        }
      }

      const strongestObjection = judgeResult.strongest_objection;
      finalAction = {
        orders: judgeResult.orders ?? aggregateRevisions(revisions).orders,
        justification: judgeResult.justification ?? judgeResult.audited_memo ?? '',
        confidence: judgeResult.confidence ?? 0.5,
        claims,
      };
      (finalAction as Action & { strongest_objection?: string }).strongest_objection = strongestObjection;
    }

    const strongestObjection =
      (finalAction as Action & { strongest_objection?: string }).strongest_objection;

    const trace: AgentTrace = {
      observation_timestamp: observation.timestamp,
      architecture: 'debate',
      what_i_saw: context.slice(0, 300) + '...',
      hypothesis: `Debate across ${this.roles.join(', ')}: proposals → critiques → revisions → judge`,
      decision: finalAction.orders.length
        ? finalAction.orders.map((o) => `${o.side} ${o.size} ${o.ticker}`).join('; ')
        : 'Hold',
      risks_or_falsifiers: allCritiquesText.slice(0, 500),
      strongest_objection: strongestObjection,
      debate_turns: debateTurns,
      action: finalAction,
      logged_at: new Date().toISOString(),
    };

    await this.logTrace(trace, observation.timestamp);
    return { action: finalAction, trace };
  }

  private async logTrace(trace: AgentTrace, timestamp: string): Promise<void> {
    const dir = this.traceDir;
    if (!fs.existsSync(dir)) fs.mkdirSync(dir, { recursive: true });
    const safeTs = timestamp.replace(/[:.]/g, '-');
    const filename = `debate_${safeTs}.json`;
    fs.writeFileSync(path.join(dir, filename), JSON.stringify(trace, null, 2), 'utf-8');
    console.log(`[Debate] Trace written to ${path.join(dir, filename)}`);
  }

  private getMockProposal(role: AgentRole, obs: Observation): Action {
    const returns = obs.market_state.returns ?? {};
    const tickers = obs.universe;
    const t = tickers[0] ?? 'AAPL';
    const r = returns[t] ?? 0;
    const bias = role === 'macro' ? 0.02 : role === 'value' ? -0.01 : role === 'risk' ? -0.03 : 0;
    const threshold = r + bias;
    const orders =
      threshold > 0.02 ? [{ ticker: t, side: 'buy' as const, size: 5 + Math.floor(Math.random() * 10) }] :
      threshold < -0.02 ? [{ ticker: t, side: 'sell' as const, size: 5 + Math.floor(Math.random() * 10) }] :
      [];
    return {
      orders,
      justification: `[${role} mock] threshold=${(threshold * 100).toFixed(2)}%`,
      confidence: 0.4 + Math.random() * 0.3,
      claims: [{ claim_text: `${t} return associated with ${role}`, pearl_level: 'L1', variables: [t, 'return'], confidence: 0.5 }],
    };
  }

  private getMockCritique(role: AgentRole, proposals: RoleProposal[]): CritiqueResult {
    const others = proposals.filter((p) => p.role !== role).slice(0, 2);
    return {
      role,
      critiques: others.map((o) => ({
        target_role: o.role,
        objection: `[mock] ${role} objects to ${o.role}'s confidence; possible confounder`,
        falsifier: 'Unexpected macro shock would invalidate',
      })),
      self_critique: `[mock] ${role} reconsidering position size`,
    };
  }

  private getMockRevision(role: AgentRole, original: Action, obs: Observation): Action & { revision_notes?: string } {
    const rev = this.getMockProposal(role, obs);
    return {
      ...rev,
      confidence: Math.max(0.3, original.confidence - 0.1),
      revision_notes: `[mock] ${role} revised based on critiques`,
    };
  }

  private getMockJudgeDecision(revisions: RevisionResult[], obs: Observation): Action {
    const aggregated = aggregateRevisions(revisions);
    return {
      ...aggregated,
      justification: `[Judge mock] Audited memo: debate completed. Final decision based on ${revisions.length} revised proposals.`,
    };
  }
}
