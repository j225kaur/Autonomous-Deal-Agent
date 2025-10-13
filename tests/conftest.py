import os
import sys
import pytest

# Ensure the repository root (parent of `src/`) is on sys.path so tests can import
# using the `src.` package prefix. This mirrors running tests with `PYTHONPATH=.` or
# `PYTHONPATH=<repo_root>` and makes test runs reproducible in CI/dev.
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
	sys.path.insert(0, repo_root)

os.environ.setdefault("INDEX_DIR", "embeddings/faiss_test")
