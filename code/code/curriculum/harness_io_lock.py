"""Cross-process file lock for Auto-Harness promote / shared jsonl writes."""
from __future__ import annotations

import contextlib
import fcntl
import os
import time
from typing import Iterator


def promote_lock_path(out_root: str) -> str:
    return os.path.join(out_root, ".promote.lock")


@contextlib.contextmanager
def promote_lock(out_root: str, *, timeout_s: float = 7200.0) -> Iterator[None]:
    os.makedirs(out_root, exist_ok=True)
    path = promote_lock_path(out_root)
    with open(path, "a+") as fh:
        t0 = time.time()
        while True:
            try:
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.time() - t0 > timeout_s:
                    raise TimeoutError(f"promote_lock timeout: {path}")
                time.sleep(0.5)
        try:
            yield
        finally:
            fcntl.flock(fh.fileno(), fcntl.LOCK_UN)
