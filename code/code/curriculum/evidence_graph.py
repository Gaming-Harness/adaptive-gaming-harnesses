"""Minimal research evidence graph for Gaming WM Auto Research.

Each node links: hypothesis → code/mechanism patch → probe outcomes → boundary.
Supports revoke when a counterexample arrives (课题 A 最小核).
"""
from __future__ import annotations

import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class EvidenceRecord:
    record_id: str
    hypothesis_id: str
    hypothesis: str
    mechanism_id: str
    patch: Dict[str, Any]                  # non-hyperparam change description
    probe_scores: Dict[str, float]
    confidence: float
    passed: bool
    boundary: Dict[str, Any]               # where it is claimed to apply
    counterexamples: List[Dict[str, Any]] = field(default_factory=list)
    status: str = "active"                 # active | revoked | revised
    parent_id: Optional[str] = None
    step: int = 0
    created_at: float = field(default_factory=time.time)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EvidenceGraph:
    """Append-only evidence log with revoke/revise semantics."""

    def __init__(self, path: str = ""):
        self.path = path
        self.records: List[EvidenceRecord] = []
        self.stats = {
            "proposed": 0,
            "passed": 0,
            "failed": 0,
            "revoked": 0,
            "revised": 0,
        }

    def propose(
        self,
        hypothesis: str,
        mechanism_id: str,
        patch: Dict[str, Any],
        probe_scores: Dict[str, float],
        confidence: float,
        passed: bool,
        boundary: Optional[Dict[str, Any]] = None,
        step: int = 0,
        parent_id: Optional[str] = None,
        meta: Optional[Dict[str, Any]] = None,
    ) -> EvidenceRecord:
        hid = str(uuid.uuid4())[:8]
        rec = EvidenceRecord(
            record_id=str(uuid.uuid4())[:8],
            hypothesis_id=hid,
            hypothesis=hypothesis,
            mechanism_id=mechanism_id,
            patch=dict(patch or {}),
            probe_scores={k: float(v) for k, v in (probe_scores or {}).items()},
            confidence=float(confidence),
            passed=bool(passed),
            boundary=dict(boundary or {"domain": "gaming_wm", "split": "train"}),
            status="active" if passed else "active",
            parent_id=parent_id,
            step=int(step),
            meta=dict(meta or {}),
        )
        if not passed:
            rec.status = "failed"
            self.stats["failed"] += 1
        else:
            self.stats["passed"] += 1
        self.stats["proposed"] += 1
        self.records.append(rec)
        self._flush()
        return rec

    def add_counterexample(
        self,
        record_id: str,
        probe_scores: Dict[str, float],
        note: str = "",
        revoke: bool = True,
    ) -> Optional[EvidenceRecord]:
        for rec in self.records:
            if rec.record_id != record_id:
                continue
            rec.counterexamples.append({
                "probe_scores": {k: float(v) for k, v in probe_scores.items()},
                "note": note,
                "t": time.time(),
            })
            if revoke and rec.status == "active":
                rec.status = "revoked"
                self.stats["revoked"] += 1
            self._flush()
            return rec
        return None

    def revise(
        self,
        parent_id: str,
        hypothesis: str,
        patch: Dict[str, Any],
        probe_scores: Dict[str, float],
        confidence: float,
        passed: bool,
        boundary: Optional[Dict[str, Any]] = None,
        step: int = 0,
    ) -> EvidenceRecord:
        parent = next((r for r in self.records if r.record_id == parent_id), None)
        mech = parent.mechanism_id if parent else "unknown"
        if parent and parent.status == "active":
            parent.status = "revised"
            self.stats["revised"] += 1
        return self.propose(
            hypothesis=hypothesis,
            mechanism_id=mech,
            patch=patch,
            probe_scores=probe_scores,
            confidence=confidence,
            passed=passed,
            boundary=boundary,
            step=step,
            parent_id=parent_id,
            meta={"revision_of": parent_id},
        )

    def active_passed(self) -> List[EvidenceRecord]:
        return [r for r in self.records if r.passed and r.status == "active"]

    def summary(self) -> Dict[str, Any]:
        return {
            "n_records": len(self.records),
            "stats": dict(self.stats),
            "n_active_passed": len(self.active_passed()),
        }

    def _flush(self) -> None:
        if not self.path:
            return
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        with open(self.path, "w") as f:
            json.dump(
                {
                    "summary": self.summary(),
                    "records": [r.to_dict() for r in self.records],
                },
                f,
                indent=2,
            )

    @classmethod
    def load(cls, path: str) -> "EvidenceGraph":
        g = cls(path=path)
        if not os.path.exists(path):
            return g
        with open(path) as f:
            payload = json.load(f)
        g.stats = payload.get("summary", {}).get("stats", g.stats)
        for d in payload.get("records", []):
            g.records.append(EvidenceRecord(**d))
        return g
