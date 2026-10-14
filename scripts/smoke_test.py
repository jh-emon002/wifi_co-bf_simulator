"""Backward-compatible test entrypoint: python scripts/smoke_test.py."""
from pathlib import Path
import runpy
runpy.run_path(str(Path(__file__).resolve().parent.parent / 'tests' / 'smoke_test.py'), run_name='__main__')
