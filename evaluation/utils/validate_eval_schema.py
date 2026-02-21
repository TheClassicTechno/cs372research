import json
import sys
from pathlib import Path
from typing import Any, Dict

from jsonschema import Draft202012Validator
from jsonschema.exceptions import ValidationError


SCHEMA_PATH = Path(__file__).resolve().parents[1] / "schemas" / "eval.schema.json"


def load_json(path: Path) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        raise RuntimeError(f"Failed to load JSON from {path}: {e}")


def load_schema(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Schema file not found at {path}")
    return load_json(path)


def validate_schema(instance: Dict[str, Any], schema: Dict[str, Any], file_path: Path):
    validator = Draft202012Validator(schema)
    errors = sorted(validator.iter_errors(instance), key=lambda e: e.path)

    if errors:
        print(f"\n❌ Schema validation failed for {file_path.name}:")
        for error in errors:
            path = ".".join([str(p) for p in error.path])
            print(f"  - Path: {path or '(root)'}")
            print(f"    Message: {error.message}")
        raise ValidationError(f"Schema validation failed for {file_path.name}")
    else:
        print(f"✅ Schema validation passed: {file_path.name}")


def validate_semantics(instance: Dict[str, Any], file_path: Path):
    """
    Light semantic validation beyond JSON Schema.
    This is intentionally conservative and easy to extend.
    """

    evaluation_mode = instance.get("evaluation_mode")
    experiment_config = instance.get("experiment_config", {})
    control = experiment_config.get("control")

    # Rule 1: If control.enabled == True → evaluation_mode must be in_loop
    if control and control.get("enabled") is True:
        if evaluation_mode != "in_loop":
            raise ValueError(
                f"{file_path.name}: control.enabled=True requires evaluation_mode='in_loop'"
            )

    # Rule 2: If control is null → fine
    # Rule 3: If control.enabled == False → no extra constraint
    # Rule 4: If policy type is 'pid', ensure parameters contain Kp/Ki/Kd (soft check)
    if control and control.get("enabled") is True:
        policy = control.get("policy", {})
        if policy.get("type") == "pid":
            params = policy.get("parameters") or {}
            required_keys = {"Kp", "Ki", "Kd"}
            if not required_keys.issubset(params.keys()):
                raise ValueError(
                    f"{file_path.name}: PID policy must define Kp, Ki, Kd"
                )

    # Rule 5: If termination.stop_when exists, ensure signal_id is declared
    if control and control.get("termination"):
        signals = control.get("signals", [])
        declared_ids = {s["signal_id"] for s in signals}
        stop_conditions = control["termination"].get("stop_when") or []
        for cond in stop_conditions:
            if cond["signal_id"] not in declared_ids:
                raise ValueError(
                    f"{file_path.name}: termination.stop_when references unknown signal_id '{cond['signal_id']}'"
                )

    print(f"✅ Semantic validation passed: {file_path.name}")


def validate_file(file_path: Path, schema: Dict[str, Any]):
    print(f"\nValidating {file_path} ...")
    instance = load_json(file_path)

    validate_schema(instance, schema, file_path)
    validate_semantics(instance, file_path)


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_eval_examples.py <file_or_directory>")
        sys.exit(1)

    target_path = Path(sys.argv[1]).resolve()

    if not target_path.exists():
        print(f"Path does not exist: {target_path}")
        sys.exit(1)

    schema = load_schema(SCHEMA_PATH)

    if target_path.is_file():
        validate_file(target_path, schema)
    else:
        json_files = list(target_path.glob("*.json"))
        if not json_files:
            print(f"No JSON files found in {target_path}")
            sys.exit(1)

        failures = 0
        for file_path in json_files:
            try:
                validate_file(file_path, schema)
            except Exception as e:
                print(f"❌ {e}")
                failures += 1

        if failures > 0:
            print(f"\n{failures} file(s) failed validation.")
            sys.exit(1)

    print("\nAll validations completed successfully.")


if __name__ == "__main__":
    main()