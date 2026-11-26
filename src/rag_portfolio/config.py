"""The values you might change without reading the code.

Python rather than YAML on purpose: nothing here needs to change without a code
change, so a parser and a schema would be more machinery than five constants
deserve. OLLAMA_HOST takes an environment override because that one genuinely
differs between a container and a laptop.

EMBED_MODEL and TOP_K live here because they were declared twice, in search.py
and run_smoke.py. Changing one and not the other would leave a saved baseline
looking comparable to a demo run when it is not.
"""
import os
from pathlib import Path

# Resolved from this file, so the scripts stop recomputing it under two
# different names. Assumes the repo layout, which holds for an editable install
# and would not for a wheel installed somewhere else.
ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT / "data"
CONSTITUTION_DIR = DATA_DIR / "constitution"
FICTIONAL_DIR = DATA_DIR / "fictional"
EVAL_DIR = ROOT / "eval"

OLLAMA_HOST = os.environ.get("OLLAMA_BASE_URL", "http://host.docker.internal:11434")

EMBED_MODEL = "bge-m3:latest"
GENERATION_MODEL = "gemma4:12b"
TOP_K = 5

# The name of the saved vector index inside a corpus directory. One per corpus
# version, so isolation is a property of which file you loaded.
INDEX_FILENAME = "vectors.json"
