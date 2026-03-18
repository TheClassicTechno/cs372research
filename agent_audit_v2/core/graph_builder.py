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
