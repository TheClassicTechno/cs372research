# Agent Audit & Control System V2

LangGraph-based modular pipeline system for:

- multi-agent debate
- audit + control loops (RAudit / PID)
- experiment pipelines
- reproducible evaluation

## ⚡ Setup (uv)

Install uv (if needed):
```bash
curl -Ls https://astral.sh/uv/install.sh | sh
```

Create env + install:
```bash
uv sync
```

Run demo:
```bash
uv run python -m scripts.run_demo
```

Run tests:
```bash
uv run pytest -q
```

## Design Principles

- Stage = unit of computation
- LangGraph node = execution wrapper
- Pipeline spec = declarative DAG
- Compiler = spec → LangGraph graph
- Hooks = control system (retry, PID, etc.)
