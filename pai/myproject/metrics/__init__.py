"""Precision AI MyProject metrics — public re-exports.

All stable public symbols are re-exported here so callers can import directly
from ``pai.myproject.metrics`` without knowing the submodule layout.
"""

from pai.myproject.metrics.compute import char_count, word_count

__all__ = [
    "char_count",
    "word_count",
]
