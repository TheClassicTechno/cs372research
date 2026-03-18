import stages
from core.pipeline import PipelineRunner

def main():
    runner = PipelineRunner.from_yaml("configs/demo.yaml")
    result = runner.run({"x": 1})
    print(result)

if __name__ == "__main__":
    main()
