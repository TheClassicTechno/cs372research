/**
 * SingleAgentRunner: Reads Observation, produces Action, logs structured trace.
 * Baseline architecture for the CS372 trading agent system.
 */

import OpenAI from 'openai';
import * as fs from 'fs';
import * as path from 'path';
import type { Observation, Action, AgentTrace, Claim, Order } from './types';
import { buildSingleAgentPrompt } from './prompts';

function getOpenAI(): OpenAI {
  return new OpenAI({ apiKey: process.env.OPENAI_API_KEY ?? 'sk-dummy' });
}

// -----------------------------------------------------------------------------
// PARSING
// -----------------------------------------------------------------------------

interface LLMResponse {
  what_i_saw?: string;
  hypothesis?: string;
  decision?: string;
  risks_or_falsifiers?: string;
  orders?: Array<{ ticker: string; side: 'buy' | 'sell'; size: number }>;
  justification?: string;
  confidence?: number;
  claims?: Array<{
    claim_text: string;
    pearl_level: 'L1' | 'L2' | 'L3';
    variables?: string[];
    assumptions?: string[];
    confidence?: number;
  }>;
}

function parseLLMResponse(text: string): LLMResponse {
  // Extract JSON from response (handle markdown code blocks)
  const jsonMatch = text.match(/```(?:json)?\s*([\s\S]*?)```/) ?? [null, text];
  const jsonStr = jsonMatch[1] ?? text;
  const cleaned = jsonStr.trim();
  return JSON.parse(cleaned) as LLMResponse;
}

function toAction(res: LLMResponse): Action {
  const orders: Order[] = (res.orders ?? []).map((o) => ({
    ticker: o.ticker,
    side: o.side,
    size: o.size,
    type: 'market' as const,
  }));

  const claims: Claim[] = (res.claims ?? []).map((c) => ({
    claim_text: c.claim_text,
    pearl_level: c.pearl_level ?? 'L1',
    variables: c.variables ?? [],
    assumptions: c.assumptions,
    confidence: c.confidence ?? 0.5,
  }));

  return {
    orders,
    justification: res.justification ?? 'No justification provided.',
    confidence: Math.max(0, Math.min(1, res.confidence ?? 0.5)),
    claims,
  };
}

// -----------------------------------------------------------------------------
// RUNNER
// -----------------------------------------------------------------------------

export interface SingleAgentRunnerOptions {
  /** Directory to write trace files (default: ./traces) */
  traceDir?: string;
  /** Model to use (default: gpt-4o-mini for cost) */
  model?: string;
  /** If true, use mock response instead of API call */
  mock?: boolean;
}

export class SingleAgentRunner {
  private traceDir: string;
  private model: string;
  private mock: boolean;

  constructor(options: SingleAgentRunnerOptions = {}) {
    this.traceDir = options.traceDir ?? path.join(process.cwd(), 'traces');
    this.model = options.model ?? 'gpt-4o-mini';
    this.mock = options.mock ?? false;
  }

  async run(observation: Observation): Promise<{ action: Action; trace: AgentTrace }> {
    const prompt = buildSingleAgentPrompt(observation);

    let res: LLMResponse;
    if (this.mock) {
      res = this.getMockResponse(observation);
    } else {
      const openai = getOpenAI();
      const completion = await openai.chat.completions.create({
        model: this.model,
        messages: [
          { role: 'system', content: 'You are a trading agent. Respond with valid JSON only.' },
          { role: 'user', content: prompt },
        ],
        temperature: 0.3,
      });
      const content = completion.choices[0]?.message?.content ?? '{}';
      try {
        res = parseLLMResponse(content);
      } catch (e) {
        console.error('Failed to parse LLM response:', content);
        res = this.getMockResponse(observation);
      }
    }

    const action = toAction(res);
    const trace: AgentTrace = {
      observation_timestamp: observation.timestamp,
      architecture: 'single',
      what_i_saw: res.what_i_saw ?? 'N/A',
      hypothesis: res.hypothesis ?? 'N/A',
      decision: res.decision ?? 'N/A',
      risks_or_falsifiers: res.risks_or_falsifiers,
      action,
      logged_at: new Date().toISOString(),
    };

    await this.logTrace(trace, observation.timestamp);
    return { action, trace };
  }

  private async logTrace(trace: AgentTrace, timestamp: string): Promise<void> {
    const dir = this.traceDir;
    if (!fs.existsSync(dir)) {
      fs.mkdirSync(dir, { recursive: true });
    }
    const safeTs = timestamp.replace(/[:.]/g, '-');
    const filename = `single_${safeTs}.json`;
    const filepath = path.join(dir, filename);
    fs.writeFileSync(filepath, JSON.stringify(trace, null, 2), 'utf-8');
    console.log(`[SingleAgent] Trace written to ${filepath}`);
  }

  private getMockResponse(obs: Observation): LLMResponse {
    // Trivial baseline: hold or small position based on returns
    const tickers = obs.universe;
    const returns = obs.market_state.returns ?? {};
    const bestTicker = tickers.reduce((a, b) =>
      (returns[a] ?? 0) > (returns[b] ?? 0) ? a : b
    );
    const bestReturn = returns[bestTicker] ?? 0;
    const orders =
      bestReturn > 0.01
        ? [{ ticker: bestTicker, side: 'buy' as const, size: 10 }]
        : bestReturn < -0.01
          ? [{ ticker: bestTicker, side: 'sell' as const, size: 10 }]
          : [];

    return {
      what_i_saw: `Observed prices for ${tickers.join(', ')}. Returns: ${JSON.stringify(returns)}`,
      hypothesis: bestReturn > 0 ? 'Momentum suggests continuation' : bestReturn < 0 ? 'Negative momentum' : 'No clear signal',
      decision: orders.length ? `Trade ${orders[0]!.side} ${orders[0]!.size} ${orders[0]!.ticker}` : 'Hold',
      risks_or_falsifiers: 'A reversal in momentum or unexpected news would change my mind.',
      orders,
      justification: `Mock agent: best return ${(bestReturn * 100).toFixed(2)}% for ${bestTicker}.`,
      confidence: 0.5,
      claims: [
        {
          claim_text: `${bestTicker} return is associated with recent price movement`,
          pearl_level: 'L1',
          variables: [bestTicker, 'price'],
          confidence: 0.5,
        },
      ],
    };
  }
}
