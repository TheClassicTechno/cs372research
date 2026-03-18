import stages
from core.pipeline import PipelineRunner


def test_pipeline():
    runner = PipelineRunner.from_yaml("configs/demo.yaml")
    result = runner.run({"x": 1})
    assert result["data"]["x"] == 3
