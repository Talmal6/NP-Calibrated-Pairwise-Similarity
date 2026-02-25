from __future__ import annotations
import csv
import json
from pathlib import Path
from datetime import datetime
from typing import Any, Dict, Iterable, List, Optional


def make_run_dir(base_dir: str = "outputs", run_name: Optional[str] = None) -> Path:
    """
    Create an outputs directory for a run, e.g.:
      outputs/run_20260209_143012/
    Returns the created Path.
    """
    base = Path(base_dir)
    base.mkdir(parents=True, exist_ok=True)

    if run_name is None:
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_name = f"run_{ts}"

    run_dir = base / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(path: str | Path, obj: Dict[str, Any], indent: int = 2) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", encoding="utf-8") as f:
        json.dump(obj, f, indent=indent, sort_keys=True)
    print(f"[JSON] Saved: {p}")


def save_csv_rows(path: str | Path, rows: Iterable[Dict[str, Any]], fieldnames: List[str]) -> None:
    """
    Save rows to CSV with explicit field order.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with p.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})
    print(f"[CSV] Saved: {p}")
