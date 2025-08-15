"""
Unit and regression test for the rylm package.
"""

# Import package, test suite, and other packages as needed
import sys

import pytest

import rylm


def test_rylm_imported():
    """Sample test, will always pass so long as import statement worked."""
    assert "rylm" in sys.modules
