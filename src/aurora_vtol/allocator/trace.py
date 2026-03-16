from __future__ import annotations
import json
from pathlib import Path
from typing import Any, Dict, List

def save_trace_json(path: str, meta: Dict[str, Any], hist: Dict[str, List[Any]]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": meta, "hist": hist}, indent=2), encoding="utf-8")
