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
