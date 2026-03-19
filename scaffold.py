#!/usr/bin/env python3
"""
Scaffold Agent Audit & Control System V2 INSIDE current repo.

Key upgrades:
- Creates ./agent_audit_v2 (subdirectory, NOT standalone repo)
- Uses uv (not pip / venv)
- Clean separation for later integration into main system
- Ready for LangGraph + experiments + audits

Usage:
    python scaffold_agent_audit_v2.py

Optional:
    python scaffold_agent_audit_v2.py custom/subdir/path
"""

from __future__ import annotations

import sys
from pathlib import Path
from textwrap import dedent


def write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(content).lstrip("\n"), encoding="utf-8")


def main() -> None:
    # 🔑 DEFAULT: create inside current repo
    root = (
        Path(sys.argv[1]).expanduser().resolve()
        if len(sys.argv) > 1
        else Path.cwd() / "agent_audit_v2"
    )

    root.mkdir(parents=True, exist_ok=True)

    files: dict[str, str] = {
        "README.md": """
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
        """,

        "pyproject.toml": """
        [project]
        name = "agent-audit-v2"
        version = "0.1.0"
        description = "Agent Audit + Control System V2"
        requires-python = ">=3.10"
        dependencies = [
            "langgraph>=0.2.14",
            "langchain-core>=0.3.0",
            "pydantic>=2.7.0",
            "pyyaml>=6.0.1",
            "pytest>=8.0.0",
        ]

        [tool.uv]
        dev-dependencies = []

        [tool.setuptools.packages.find]
        include = ["core*", "stages*", "audits*", "evaluators*", "hooks*", "scripts*"]
        """,

        "core/state.py": """
        from dataclasses import dataclass, field
        from typing import Any


        @dataclass
        class PipelineState:
            data: dict[str, Any] = field(default_factory=dict)
            history: list[dict[str, Any]] = field(default_factory=list)
            artifacts: dict[str, Any] = field(default_factory=dict)

            def snapshot(self):
                self.history.append(dict(self.data))

            def update(self, payload: dict[str, Any]):
                self.data.update(payload)

            def to_dict(self):
                return {
                    "data": self.data,
                    "history": self.history,
                    "artifacts": self.artifacts,
                }

            @classmethod
            def from_dict(cls, d):
                return cls(
                    data=d.get("data", {}),
                    history=d.get("history", []),
                    artifacts=d.get("artifacts", {}),
                )
        """,

        "core/registry.py": """
        STAGE_REGISTRY = {}
        AUDIT_REGISTRY = {}
        HOOK_REGISTRY = {}
        EVAL_REGISTRY = {}

        def register(registry, name):
            def deco(cls):
                registry[name] = cls
                return cls
            return deco
        """,

        "core/runtime.py": """
        from core.state import PipelineState
        from core.registry import STAGE_REGISTRY


        def run_stage(name, state):
            stage = STAGE_REGISTRY[name]()
            out = stage.run(state)
            state.update(out)
            state.snapshot()
            return state
        """,

        "core/pipeline_compiler.py": """
        from langgraph.graph import StateGraph, START, END
        from core.state import PipelineState
        from core.runtime import run_stage


        def compile_pipeline(spec):
            graph = StateGraph(dict)

            prev = START

            for step in spec["pipeline"]:
                name = step["name"]

                def node_fn(state, name=name):
                    ps = PipelineState.from_dict(state)
                    ps = run_stage(name, ps)
                    return ps.to_dict()

                graph.add_node(name, node_fn)
                graph.add_edge(prev, name)
                prev = name

            graph.add_edge(prev, END)
            return graph.compile()
        """,

        "core/pipeline.py": """
        import yaml
        from core.graph_builder import compile_pipeline
        from core.state import PipelineState


        class PipelineRunner:
            def __init__(self, spec):
                self.graph = compile_pipeline(spec)

            @classmethod
            def from_yaml(cls, path):
                with open(path) as f:
                    return cls(yaml.safe_load(f))

            def run(self, init):
                state = PipelineState(data=init).to_dict()
                return self.graph.invoke(state)
        """,

        "stages/__init__.py": """
        import stages.simple
        """,

        "stages/simple.py": """
        from core.registry import register, STAGE_REGISTRY
        from core.state import PipelineState


        @register(STAGE_REGISTRY, "example_stage")
        class ExampleStage:
            def run(self, state: PipelineState):
                x = state.data.get("x", 0)
                return {"x": x + 1}
        """,

        "configs/demo.yaml": """
        pipeline:
          - name: example_stage
          - name: example_stage
        """,

        "scripts/run_demo.py": """
        import stages
        from core.pipeline import PipelineRunner

        def main():
            runner = PipelineRunner.from_yaml("configs/demo.yaml")
            result = runner.run({"x": 1})
            print(result)

        if __name__ == "__main__":
            main()
        """,

        "tests/test_basic.py": """
        import stages
        from core.pipeline import PipelineRunner


        def test_pipeline():
            runner = PipelineRunner.from_yaml("configs/demo.yaml")
            result = runner.run({"x": 1})
            assert result["data"]["x"] == 3
        """,

        ".gitignore": """
        __pycache__/
        .venv/
        *.pyc
        """,
    }

    for path, content in files.items():
        write(root / path, content)

    print(f"✅ Created scaffold at: {root}")
    print("\\nNext steps:")
    print(f"cd {root}")
    print("uv sync")
    print("uv run python -m scripts.run_demo")
    print("uv run pytest -q")


if __name__ == "__main__":
    main()