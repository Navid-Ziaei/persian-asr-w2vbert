import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CACHE_ROOT = PROJECT_ROOT / ".cache"

os.environ.setdefault("HF_HOME", str(CACHE_ROOT / "huggingface"))
os.environ.setdefault("HF_HUB_CACHE", str(CACHE_ROOT / "huggingface" / "hub"))
os.environ.setdefault(
    "HF_DATASETS_CACHE", str(CACHE_ROOT / "huggingface" / "datasets")
)
os.environ.setdefault("XDG_CACHE_HOME", str(CACHE_ROOT))