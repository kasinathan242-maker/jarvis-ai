"""JARVIS Lite — Pytest Configuration & Test Fixtures.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Set test environment flag
os.environ["ENV"] = "test"
