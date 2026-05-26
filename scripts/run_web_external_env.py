from __future__ import annotations

import os
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
CONDA_ENV = ROOT / ".conda-py311"

if CONDA_ENV.exists():
    dll_dir = CONDA_ENV / "Library" / "bin"
    if dll_dir.exists() and hasattr(os, "add_dll_directory"):
        os.add_dll_directory(str(dll_dir))
    sys.path.append(str(CONDA_ENV / "Lib" / "site-packages"))

sys.path.insert(0, str(ROOT))

import uvicorn
from web.app import app


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000, log_level="warning")
