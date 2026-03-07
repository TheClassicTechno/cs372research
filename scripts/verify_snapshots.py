import json
import glob
from pathlib import Path

def verify_snapshots():
    snapshot_dir = Path("data-pipeline/final_snapshots/json_data")
    snapshots = sorted(list(snapshot_dir.glob("snapshot_*.json")))
    
    issues = []
    
    for snap_path in snapshots:
        with open(snap_path, "r") as f:
            data = json.load(f)
        
        year = data.get("year")
        quarter = data.get("quarter")
        tickers = data.get("tickers", [])
        
        for t in tickers:
            td = data.get("ticker_data", {}).get(t, {})
            af = td.get("asset_features", {})
            
            if not af or af.get("close") is None or af.get("close") == 0.0:
                issues.append(f"{year}_{quarter}: {t} has missing or zero close price")
                
    if not issues:
        print("All snapshots verified: No zero or missing prices found for any ticker.")
    else:
        print(f"Found {len(issues)} issues across snapshots:")
        for issue in issues:
            print(f"  - {issue}")

if __name__ == "__main__":
    verify_snapshots()
