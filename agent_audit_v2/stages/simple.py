from core.registry import register, STAGE_REGISTRY
from core.state import PipelineState


@register(STAGE_REGISTRY, "example_stage")
class ExampleStage:
    def run(self, state: PipelineState):
        x = state.data.get("x", 0)
        return {"x": x + 1}
