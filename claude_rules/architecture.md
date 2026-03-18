# Architecture Rules

This document defines allowed system structure.

## Principles
- Clear module boundaries
- Explicit data flow
- No hidden dependencies

## State Management
- All state must be explicit
- No implicit mutation across modules

## LangGraph / Agent Systems
- State must be minimal and well-defined
- Each node must have clear input/output contracts

## Code Organization
- Small, composable functions
- No cross-module side effects
