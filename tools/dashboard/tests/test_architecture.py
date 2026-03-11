import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]

def test_dashboard_architecture():
    script = PROJECT_ROOT / "tools" / "dashboard" / "rules" / "check_dashboard_architecture.py"
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=str(PROJECT_ROOT),
    )
    assert result.returncode == 0