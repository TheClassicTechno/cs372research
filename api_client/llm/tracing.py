from typing import Dict, Any


def build_trace_entry(
    model_name: str,
    prompt_id: str,
    raw_response: str
) -> Dict[str, Any]:
    return {
        "model_name": model_name,
        "prompt_id": prompt_id,
        "raw_response": raw_response,
    }
