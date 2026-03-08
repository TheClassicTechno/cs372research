import subprocess
import sys

def test_dashboard_architecture():
    result = subprocess.run(
        ["python", "tools/dashboard/rules/check_dashboard_architecture.py"]
    )
    assert result.returncode == 0