#!/usr/bin/env npx tsx
/**
 * Demo: Run SingleAgent and MajorityVote on 5 timesteps of fake data.
 * Shows actions emitted and traces saved.
 *
 * Usage:
 *   OPENAI_API_KEY=sk-... npx tsx Final_project/agents/demo.ts
 *   # Or with mock (no API key needed):
 *   MOCK=1 npx tsx Final_project/agents/demo.ts
 */

import * as path from 'path';
import type { Observation } from './types';
import { SingleAgentRunner } from './single-agent-runner';
import { MajorityVoteRunner } from './majority-vote-runner';
import { DebateRunner } from './debate-runner';

// -----------------------------------------------------------------------------
// FAKE DATA GENERATOR
// -----------------------------------------------------------------------------

function generateFakeObservation(timestep: number): Observation {
  const basePrices: Record<string, number> = {
    AAPL: 180 + timestep * 2,
    GOOGL: 140 + timestep * 1.5,
    MSFT: 380 + timestep * 3,
  };
  const volatility = 0.02;
  const returns: Record<string, number> = {};
  for (const [t, p] of Object.entries(basePrices)) {
    const prev = p - (timestep > 0 ? 2 : 0);
    returns[t] = (Math.random() - 0.5) * volatility * 2;
  }

  const newsSnippets = [
    'Fed signals potential rate cuts in Q2. Tech stocks rally.',
    'Earnings beat for AAPL. Guidance raised.',
    'Macro uncertainty. Risk-off sentiment.',
    'Strong jobs report. Yields rise.',
    'Mixed signals. Market consolidating.',
  ];

  const date = new Date('2025-02-15T10:00:00Z');
  date.setMinutes(date.getMinutes() + timestep * 15);
  return {
    timestamp: date.toISOString(),
    universe: ['AAPL', 'GOOGL', 'MSFT'],
    market_state: {
      prices: basePrices,
      returns,
      volatility: { AAPL: 0.25, GOOGL: 0.22, MSFT: 0.20 },
    },
    text_context: newsSnippets[timestep % newsSnippets.length],
    portfolio_state: {
      cash: 10000 - timestep * 500,
      positions: { AAPL: timestep * 5, GOOGL: 0, MSFT: 0 },
      exposures: {},
    },
    constraints: { maxLeverage: 2, maxPositionSize: 100 },
  };
}

// -----------------------------------------------------------------------------
// MAIN
// -----------------------------------------------------------------------------

async function main() {
  const mock = process.env.MOCK === '1' || !process.env.OPENAI_API_KEY;
  // Traces go in Final_project/traces
  const baseDir = process.cwd().includes('Final_project')
    ? path.join(process.cwd().split('Final_project')[0]!, 'Final_project')
    : path.join(process.cwd(), 'Final_project');
  const traceDir = path.join(baseDir, 'traces');

  console.log('\n=== CS372 Agent Demo ===');
  console.log(`Mode: ${mock ? 'MOCK (no API)' : 'LIVE (OpenAI)'}`);
  console.log(`Traces: ${traceDir}\n`);

  const singleRunner = new SingleAgentRunner({ traceDir, mock });
  const majorityRunner = new MajorityVoteRunner({ traceDir, mock, roles: ['macro', 'value', 'risk'] });
  const debateRunner = new DebateRunner({ traceDir, mock, roles: ['macro', 'value', 'risk'] });

  const timesteps = 5;
  const observations: Observation[] = [];
  for (let t = 0; t < timesteps; t++) {
    observations.push(generateFakeObservation(t));
  }

  console.log('--- Single Agent ---');
  for (let i = 0; i < timesteps; i++) {
    const obs = observations[i]!;
    console.log(`\nTimestep ${i + 1}/${timesteps} @ ${obs.timestamp}`);
    const { action } = await singleRunner.run(obs);
    console.log('  Orders:', action.orders.length ? action.orders : 'HOLD');
    console.log('  Confidence:', action.confidence.toFixed(2));
    console.log('  Justification:', action.justification.slice(0, 80) + '...');
  }

  console.log('\n--- Majority Vote (3 agents) ---');
  for (let i = 0; i < timesteps; i++) {
    const obs = observations[i]!;
    console.log(`\nTimestep ${i + 1}/${timesteps} @ ${obs.timestamp}`);
    const { action } = await majorityRunner.run(obs);
    console.log('  Orders:', action.orders.length ? action.orders : 'HOLD');
    console.log('  Confidence:', action.confidence.toFixed(2));
    console.log('  Justification (first 80 chars):', action.justification.slice(0, 80) + '...');
  }

  console.log('\n--- Debate (3 agents, 3 rounds) ---');
  for (let i = 0; i < timesteps; i++) {
    const obs = observations[i]!;
    console.log(`\nTimestep ${i + 1}/${timesteps} @ ${obs.timestamp}`);
    const { action } = await debateRunner.run(obs);
    console.log('  Orders:', action.orders.length ? action.orders : 'HOLD');
    console.log('  Confidence:', action.confidence.toFixed(2));
    console.log('  Justification (first 80 chars):', action.justification.slice(0, 80) + '...');
  }

  console.log('\n=== Demo complete ===');
  console.log(`Traces saved to: ${traceDir}`);
}

main().catch((e) => {
  console.error(e);
  process.exit(1);
});
