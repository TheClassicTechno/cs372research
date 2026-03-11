import json
from pathlib import Path


def organize_debate_json(input_path, output_path):
    with open(input_path, "r") as f:
        data = json.load(f)

    organized = {
        "round": None,
        "agents": [],
        "tickers": [],
        "phases": {},
        "crit": {},
        "metrics": {},
        "allocations": {
            "proposals": {},
            "revisions": {}
        }
    }

    for item in data:

        # --- CRIT entries ---
        if item.get("phase") == "critique":
            agent = item["agent"]

            organized["round"] = item["round"]

            organized["crit"][agent] = {
                "rho": item["rho"],
                "pillars": item["pillars"],
                "reasoning": item["reasoning"]
            }

        # --- phase metrics (propose / revision) ---
        elif item.get("phase") in ["propose", "revision"]:
            phase = item["phase"]

            organized["round"] = item["round"]
            organized["agents"] = item["agents"]
            organized["tickers"] = item["tickers"]

            organized["phases"][phase] = {
                "timestamp": item["timestamp"],
                "js_divergence": item["js_divergence"],
                "evidence_overlap": item["evidence_overlap"],
                "allocation_sums": item["allocation_sums"],
                "nonzero_positions": item["nonzero_positions"],
                "vectors": item["vectors"]
            }

        # --- round_state summary ---
        elif "proposals" in item and "revisions" in item:
            organized["metrics"] = item["metrics"]

            organized["allocations"]["proposals"] = {
                k: v["allocation"] for k, v in item["proposals"].items()
            }

            organized["allocations"]["revisions"] = {
                k: v["allocation"] for k, v in item["revisions"].items()
            }

    with open(output_path, "w") as f:
        json.dump(organized, f, indent=2)


if __name__ == "__main__":
    input_file = "input.json"
    output_file = "organized_round.json"

    organize_debate_json(input_file, output_file)

    print(f"Organized JSON written to {output_file}")