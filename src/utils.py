# src/utils.py
from __future__ import annotations
import os, json, yaml
from pathlib import Path
from typing import Dict, Any, Sequence
import numpy as np

# ---------- Paths ----------
ROOT: Path = Path(__file__).resolve().parents[1]  # repo root

def repo_path(*parts: str | os.PathLike) -> Path:
    """Path helper relative to the repo root."""
    return ROOT.joinpath(*parts)

def resolve(p: str | os.PathLike) -> str:
    """Absolute string path from repo root (for TF/OpenCV apis)."""
    return str(repo_path(p).resolve())

# ---------- Config ----------
def load_config(path: str | os.PathLike = "config/config.yaml") -> Dict[str, Any]:
    cfg_path = repo_path(path)
    with open(cfg_path, "r") as f:
        return yaml.safe_load(f)

# ---------- Dirs / logging / weights ----------
def create_directories(cfg: Dict[str, Any]) -> None:
    dirs = [
        cfg["data"]["raw_path"],
        cfg["data"]["processed_path"],
        cfg["data"]["splits_path"],
        "models/checkpoints",
        "models/final",
        "logs",
    ]
    for d in dirs:
        repo_path(d).mkdir(parents=True, exist_ok=True)
    print("✓ Directories created")

def calculate_class_weights(y_train: np.ndarray, num_classes: int) -> Dict[int, float]:
    from sklearn.utils.class_weight import compute_class_weight
    classes = np.arange(num_classes)
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=y_train)
    return {int(i): float(w) for i, w in zip(classes, weights)}

class Logger:
    def __init__(self, log_file: str = "logs/training.log"):
        self.path = repo_path(log_file)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def log(self, message: str) -> None:
        from datetime import datetime
        line = f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line)
        if self.path.exists():
            self.path.write_text(self.path.read_text() + line + "\n", encoding="utf-8")
        else:
            self.path.write_text(line + "\n", encoding="utf-8")

# ---------- Labels (letters) ----------
# Pick ONE canonical order and keep it everywhere.
# Recommended (matches our training/inference fallback): DEL, SPACE, NOTHING
LETTERS_DEFAULT: Sequence[str] = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ") + ["DEL", "SPACE", "NOTHING"]

def load_letter_classes(path: str | os.PathLike = "models/final/classes_letters.json") -> Sequence[str]:
    """
    Load letter classes from JSON if present (supports list or dict {"0":"A",...}).
    Falls back to LETTERS_DEFAULT otherwise.
    """
    p = repo_path(path)
    if p.exists():
        data = json.loads(p.read_text(encoding="utf-8"))
        if isinstance(data, dict):
            return [data[str(i)] for i in range(len(data))]
        return data
    return LETTERS_DEFAULT

# convenient constants if needed by tests
IDX2TOK: Sequence[str] = LETTERS_DEFAULT
TOK2IDX: Dict[str, int] = {t: i for i, t in enumerate(IDX2TOK)}
