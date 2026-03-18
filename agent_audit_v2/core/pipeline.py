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
