/**
 * Shared types for the CS372 Multi-Agent Trading System.
 * These define the contract between agents and the simulator.
 * Compatible with T³/CRIT-style evaluation and Pearl causality levels.
 */

// =============================================================================
// PEARL LEVELS (for causal claim classification)
// =============================================================================

export type PearlLevel = 'L1' | 'L2' | 'L3';

export const PEARL_LEVELS: Record<PearlLevel, string> = {
  L1: 'Association',
  L2: 'Intervention',
  L3: 'Counterfactual',
};

// =============================================================================
// OBSERVATION (input from simulator to agents)
// =============================================================================

export interface MarketState {
  /** Ticker -> price snapshot */
  prices: Record<string, number>;
  /** Ticker -> return over period */
  returns?: Record<string, number>;
  /** Ticker -> volatility measure */
  volatility?: Record<string, number>;
}

export interface PortfolioState {
  cash: number;
  /** Ticker -> position size (positive = long, negative = short) */
  positions: Record<string, number>;
  /** Optional exposure metrics */
  exposures?: Record<string, number>;
}

export interface Constraints {
  maxLeverage?: number;
  maxPositionSize?: number;
  riskLimits?: Record<string, number>;
  [key: string]: unknown;
}

export interface Observation {
  timestamp: string; // ISO 8601
  universe: string[]; // tickers
  market_state: MarketState;
  text_context?: string; // news / earnings snippets
  portfolio_state: PortfolioState;
  constraints?: Constraints;
}

// =============================================================================
// ACTION (output from agents to broker)
// =============================================================================

export type OrderSide = 'buy' | 'sell';
export type OrderType = 'market' | 'limit';

export interface Order {
  ticker: string;
  side: OrderSide;
  size: number; // shares
  type?: OrderType;
  limitPrice?: number;
}

export interface Action {
  orders: Order[];
  justification: string;
  confidence: number; // 0–1, calibrated
  claims: Claim[];
}

// =============================================================================
// CLAIM (machine-readable causal/counterfactual claims for T³ scoring)
// =============================================================================

export interface Claim {
  claim_text: string;
  pearl_level: PearlLevel;
  variables: string[]; // [X, Y, Z...]
  assumptions?: string[];
  timestamp_dependency?: string; // what ordering matters
  confidence: number;
}

// =============================================================================
// DEBATE & TRACE (for multi-agent architectures)
// =============================================================================

export interface DebateTurn {
  round: number;
  agent_id: string;
  role?: string;
  proposal?: Action;
  critique?: string;
  objections?: string[];
  revision?: Action;
}

export interface AgentTrace {
  /** Timestamp of the observation */
  observation_timestamp: string;
  /** Architecture used */
  architecture: 'single' | 'majority_vote' | 'debate';
  /** What the agent(s) saw (summary) */
  what_i_saw: string;
  /** Hypothesis formed */
  hypothesis: string;
  /** Final decision */
  decision: string;
  /** Risks / what would change my mind */
  risks_or_falsifiers?: string;
  /** Strongest objection (if overruled) - for disagreement preservation */
  strongest_objection?: string;
  /** Debate rounds (if multi-agent) */
  debate_turns?: DebateTurn[];
  /** Final action emitted */
  action: Action;
  /** When this trace was logged */
  logged_at: string;
}

// =============================================================================
// ROLE PROMPTS (for majority vote / debate)
// =============================================================================

export const AGENT_ROLES = ['macro', 'value', 'risk', 'technical'] as const;
export type AgentRole = (typeof AGENT_ROLES)[number];
