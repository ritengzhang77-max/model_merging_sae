#!/usr/bin/env python3
"""Build reusable full-state cache for SmolLM2 synthetic merge targets."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage1" / "scripts"))
import run_smollm2_synthetic_experts as smol  # noqa: E402
from analyze_smollm2_rq0 import MERGE_SPECS, load_expert_state, merge_states  # noqa: E402


CACHE_DIR = ROOT / "stage1" / "cache" / "smollm2_state_cache"


def tensor_stats(state: dict[str, torch.Tensor]) -> dict[str, int]:
    tensors = 0
    values = 0
    float_values = 0
    for v in state.values():
        tensors += 1
        values += v.numel()
        if torch.is_floating_point(v):
            float_values += v.numel()
    return {"tensors": tensors, "values": values, "float_values": float_values}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-dir", type=Path, default=CACHE_DIR)
    ap.add_argument("--force", action="store_true", help="Overwrite existing cached merge states.")
    args = ap.parse_args()

    args.cache_dir.mkdir(parents=True, exist_ok=True)

    print("[load] base", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        smol.BASE_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    base_state = {k: v.detach().cpu().half() for k, v in base_model.state_dict().items()}
    del base_model

    print("[load] experts", flush=True)
    experts = {task: load_expert_state(task) for task in MERGE_SPECS["merge_all_linear"]}

    merge_state_files = {}
    stats = {}
    for merge_name, expert_names in MERGE_SPECS.items():
        out_path = args.cache_dir / f"{merge_name}.pt"
        merge_state_files[merge_name] = out_path.name
        if out_path.exists() and not args.force:
            print(f"[skip] {merge_name} already cached at {out_path}", flush=True)
            cached = torch.load(out_path, map_location="cpu", weights_only=False)
            stats[merge_name] = tensor_stats(cached)
            del cached
            continue
        print(f"[merge] {merge_name}", flush=True)
        state = merge_states(
            base_state,
            [experts[name] for name in expert_names],
            label=merge_name,
        )
        torch.save(state, out_path)
        stats[merge_name] = tensor_stats(state)
        print(f"[save] {out_path}", flush=True)
        del state

    manifest = {
        "base_model": smol.BASE_ID,
        "tokenizer": smol.TOKENIZER_ID,
        "source_artifact_dir": str(ROOT / "stage0" / "artifacts" / "smollm2_synthetic_experts"),
        "merge_specs": MERGE_SPECS,
        "merge_state_files": merge_state_files,
        "stats": stats,
    }
    manifest_path = args.cache_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[save] {manifest_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
