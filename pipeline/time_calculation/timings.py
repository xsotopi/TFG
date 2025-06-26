# timing.py  – seconds version
import time, csv, atexit, threading, collections
from pathlib import Path

_TIMES = collections.defaultdict(list)
_LOCK  = threading.Lock()

class time_block:
    """Usage:  with time_block('load_whisper'): ...  (records seconds)"""
    def __init__(self, label): self.label = label
    def __enter__(self):
        self.t0 = time.perf_counter()
    def __exit__(self, exc_type, exc_val, exc_tb):
        dt_s = time.perf_counter() - self.t0
        with _LOCK:
            _TIMES[self.label].append(dt_s)

def dump_timings(csv_path="timings.csv"):
    """Write one row per event (label; seconds) at program exit."""
    with Path(csv_path).open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["label", "s"])
        for label, samples in _TIMES.items():
            w.writerows((label, f"{s:.6f}") for s in samples)

atexit.register(dump_timings)
