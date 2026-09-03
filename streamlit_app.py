"""
ITANTRA Indic Speech Recognition & Synthesis Benchmark.
Streamlit Cloud Entry Point.
"""

import sys
from pathlib import Path

# Ensure root directory and src directory are in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from ui.app import main

if __name__ == "__main__":
    main()
