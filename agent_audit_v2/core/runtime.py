from core.state import PipelineState
from core.registry import STAGE_REGISTRY


def run_stage(name, state):
    stage = STAGE_REGISTRY[name]()
    out = stage.run(state)
    state.update(out)
    state.snapshot()
    return state
