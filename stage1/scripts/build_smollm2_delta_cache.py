#!/usr/bin/env python3
"""Build sampled task-vector cache for SmolLM2 synthetic experts/merges."""

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
from analyze_smollm2_rq0 import (  # noqa: E402
    TASKS,
    layer_keys,
    load_expert_state,
    merge_states,
    sampled_delta,
)


CACHE_DIR = ROOT / "stage1" / "cache"
DEFAULT_CACHE = CACHE_DIR / "smollm2_delta_cache.pt"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache-path", type=Path, default=DEFAULT_CACHE)
    ap.add_argument("--rep-layers", default="0,5,10,15,20,25,29")
    ap.add_argument("--max-delta-values", type=int, default=80_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    args.cache_path.parent.mkdir(parents=True, exist_ok=True)
    rep_layers = tuple(int(x) for x in args.rep_layers.split(",") if x)

    print("[load] base", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        smol.BASE_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    base_state = {k: v.detach().cpu().half() for k, v in base_model.state_dict().items()}
    del base_model

    print("[load] experts", flush=True)
    experts = {task: load_expert_state(task) for task in TASKS}

    print("[merge] target states", flush=True)
    states = {"base": base_state}
    states.update({f"expert_{k}": v for k, v in experts.items()})
    states["merge_all_linear"] = merge_states(base_state, list(experts.values()), label="merge_all_linear")
    states["merge_arith_polite"] = merge_states(
        base_state,
        [experts["arith"], experts["polite"]],
        label="merge_arith_polite",
    )
    states["merge_arith_refusal"] = merge_states(
        base_state,
        [experts["arith"], experts["refusal"]],
        label="merge_arith_refusal",
    )
    states["merge_polite_refusal"] = merge_states(
        base_state,
        [experts["polite"], experts["refusal"]],
        label="merge_polite_refusal",
    )

    all_float_keys = [k for k, v in base_state.items() if torch.is_floating_point(v)]
    scopes = {"all": all_float_keys}
    for layer in rep_layers:
        scopes[f"l{layer}"] = layer_keys(base_state, layer)

    print("[sample] delta vectors", flush=True)
    vectors: dict[str, dict[str, torch.Tensor]] = {}
    norms: dict[str, dict[str, float]] = {}
    for model_i, (name, state) in enumerate(states.items(), start=1):
        print(f"  [model] {model_i}/{len(states)} {name}", flush=True)
        vectors[name] = {}
        norms[name] = {}
        for scope, keys in scopes.items():
            seed = args.seed + (0 if scope == "all" else int(scope[1:]))
            vec = sampled_delta(
                base_state,
                state,
                keys,
                max_values=args.max_delta_values,
                seed=seed,
            ).half()
            vectors[name][scope] = vec
            norms[name][scope] = float(vec.float().norm().item())

    payload = {
        "base_model": smol.BASE_ID,
        "experts": list(TASKS),
        "models": list(states.keys()),
        "rep_layers": rep_layers,
        "max_delta_values": args.max_delta_values,
        "seed": args.seed,
        "vectors": vectors,
        "norms": norms,
    }
    torch.save(payload, args.cache_path)
    meta_path = args.cache_path.with_suffix(".json")
    meta = {k: v for k, v in payload.items() if k not in {"vectors"}}
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[save] {args.cache_path}")
    print(f"[save] {meta_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

