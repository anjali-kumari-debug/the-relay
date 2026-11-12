"""Raw outputs: per-request event timeline (JSONL), results rows (CSV), and percentiles."""
from __future__ import annotations

import csv
import dataclasses
import json
import time
from pathlib import Path
from typing import Any, Optional, Sequence

import numpy as np

from relay.types import RESULTS_COLUMNS, ResultRow

_MS_COLUMNS = {"first_token_ms", "total_ms", "wait_ms", "work_ms", "retry_wait_ms",
               "provider_ttft_ms", "released_ms"}


def percentile(values: Sequence[float], p: float) -> Optional[float]:
    """p-th percentile (0-100) by linear interpolation between order statistics
    (numpy's default, "type 7"). None for an empty input."""
    if len(values) == 0:
        return None
    return float(np.percentile(np.asarray(values, dtype=float), p, method="linear"))


class Recorder:
    def __init__(self, events_path, results_path):
        self.events_path = Path(events_path)
        self.results_path = Path(results_path)
        self._arrival: dict[str, float] = {}
        for p in (self.events_path, self.results_path):
            p.parent.mkdir(parents=True, exist_ok=True)

    def event(self, request_id: str, name: str, **fields: Any) -> None:
        """Append one event; ``t_ms`` is time since this request's first event (its arrival)."""
        now = time.monotonic()
        t0 = self._arrival.setdefault(request_id, now)
        rec = {"request_id": request_id, "t_ms": round((now - t0) * 1000.0, 1), "event": name}
        rec.update(fields)
        with self.events_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
        if name == "done":
            self._arrival.pop(request_id, None)

    def write_row(self, row: ResultRow) -> None:
        data = dataclasses.asdict(row)
        out = []
        for col in RESULTS_COLUMNS:
            v = data[col]
            if v is None:
                out.append("")
            elif col in _MS_COLUMNS:
                out.append(f"{round(float(v), 1):.1f}")
            else:
                out.append(v)
        new = not self.results_path.exists() or self.results_path.stat().st_size == 0
        with self.results_path.open("a", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            if new:
                w.writerow(RESULTS_COLUMNS)
            w.writerow(out)
