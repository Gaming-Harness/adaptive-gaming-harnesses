#!/usr/bin/env python3
"""Build matched seed101 harnesses for directional K--P ablations."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from typing import Any, Dict

from curriculum.harness_schema import HarnessSpec


VARIANTS: Dict[str, Dict[str, bool]] = {
    "probe_only": {
        "transferable_knowledge": False,
        "k_to_p": False,
        "p_to_k": False,
    },
    "no_k_to_p": {
        "transferable_knowledge": True,
        "k_to_p": False,
        "p_to_k": True,
    },
    "no_p_to_k": {
        "transferable_knowledge": True,
        "k_to_p": True,
        "p_to_k": False,
    },
    "full": {
        "transferable_knowledge": True,
        "k_to_p": True,
        "p_to_k": True,
    },
}


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(base_path: str, knowledge_root: str, out_root: str) -> Dict[str, Any]:
    base_path = os.path.abspath(base_path)
    knowledge_root = os.path.abspath(knowledge_root)
    out_root = os.path.abspath(out_root)
    bank_path = os.path.join(knowledge_root, "skill_bank.jsonl")
    if not os.path.isfile(base_path):
        raise FileNotFoundError(base_path)
    if not os.path.isfile(bank_path):
        raise FileNotFoundError(bank_path)
    os.makedirs(out_root, exist_ok=True)

    records: Dict[str, Any] = {}
    for name, switches in VARIANTS.items():
        variant_root = os.path.join(out_root, name)
        os.makedirs(variant_root, exist_ok=True)
        harness = HarnessSpec.load(base_path)
        harness.name = f"directional_kp_{name}_seed101"
        harness.probe.enabled = True
        harness.probe.selection_strategy = "ucb"
        harness.probe.probe_on_fail = True
        harness.probe.probe_on_stall = True
        harness.probe.probe_periodic = False
        harness.probe.warmup_probes = 0
        harness.probe.budget_per_episode = 8
        harness.probe.bandit_state_path = os.path.join(
            variant_root, "probe_bandit.json"
        )

        has_knowledge = bool(switches["transferable_knowledge"])
        harness.knowledge_probe.enabled = has_knowledge
        harness.knowledge_probe.selection_strategy = "ucb"
        harness.knowledge_probe.inject_into_recovery = bool(switches["k_to_p"])
        harness.knowledge_probe.update_from_outcome = bool(switches["p_to_k"])
        harness.knowledge_probe.bandit_state_path = (
            os.path.join(variant_root, "knowledge_bandit.json")
            if switches["p_to_k"]
            else ""
        )
        harness.meta = dict(harness.meta or {})
        harness.meta.update(
            {
                "ablation_variant": name,
                "transferable_knowledge": has_knowledge,
                "k_to_p": bool(switches["k_to_p"]),
                "p_to_k": bool(switches["p_to_k"]),
                "frozen_vla": True,
                "fixed_harness": True,
                "fixed_knowledge_bank": True,
                "search_on_eval": False,
                "episode_seed": 101,
            }
        )
        harness_path = os.path.join(variant_root, "harness.json")
        harness.save(harness_path)
        records[name] = {
            **switches,
            "harness": harness_path,
            "harness_sha256": _sha256(harness_path),
            "probe_bandit_state": harness.probe.bandit_state_path,
            "knowledge_bandit_state": harness.knowledge_probe.bandit_state_path,
            "knowledge_overlay": knowledge_root if has_knowledge else "",
        }

    protocol = {
        "experiment": "directional_knowledge_probing_ablation",
        "seed": 101,
        "temperature": 0.7,
        "protocol": "legacy_k1_t0",
        "task_set": "historical Bare-t0.7 failure 110",
        "base_harness": base_path,
        "base_harness_sha256": _sha256(base_path),
        "knowledge_bank": bank_path,
        "knowledge_bank_sha256": _sha256(bank_path),
        "same_task_order": True,
        "same_initial_bandit_state": "empty per variant",
        "patch_search": False,
        "vla_frozen": True,
        "variants": records,
        "created_at": time.time(),
    }
    protocol_path = os.path.join(out_root, "protocol.json")
    with open(protocol_path, "w") as stream:
        json.dump(protocol, stream, indent=2, ensure_ascii=False)
    return protocol


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-harness", required=True)
    parser.add_argument("--knowledge-root", required=True)
    parser.add_argument("--out-root", required=True)
    args = parser.parse_args()
    protocol = build(args.base_harness, args.knowledge_root, args.out_root)
    print(json.dumps(protocol, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
