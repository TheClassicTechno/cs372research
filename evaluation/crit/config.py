from dataclasses import dataclass


@dataclass
class CritConfig:
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.0

    max_reasons: int = 6
    max_counter_reasons: int = 4

    enable_recursion: bool = False
    max_recursion_depth: int = 1

    socratic_mode: str = "weakest_link"

    trace_llm_calls: bool = True
