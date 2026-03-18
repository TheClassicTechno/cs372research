"""
Basic invariant checker stub.
"""

def check_no_silent_failures(code: str):
    if "except:" in code:
        raise AssertionError("Bare except detected")

    if "pass" in code:
        raise AssertionError("Pass statement detected")


def run_all_checks(code: str):
    check_no_silent_failures(code)
