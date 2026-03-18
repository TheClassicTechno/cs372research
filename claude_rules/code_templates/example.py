"""
Example template for Claude-generated code.
"""

from typing import List


def example_function(x: float) -> float:
    """
    Example function with invariant enforcement.
    """

    assert isinstance(x, (int, float)), "Input must be numeric"

    result = x * 2

    assert result >= 0, "Result must be non-negative"

    return result
