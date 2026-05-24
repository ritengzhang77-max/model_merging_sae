#!/usr/bin/env python3
"""Compare SAE bundle decoded deltas with dense and all-feature deltas."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))
from run_gemma2_2b_activation_patch_target_loss import MODEL_IDS  # noqa: E402
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import chat_prompt, patch_position_mask  # noqa: E402
from run_gemma2_2b_linear_merge_sae_bundle_patch import load_prompt_rows, parse_bundles, write_csv  # noqa: E402
from run_gemma2_2b_linear_weight_merge_sweep import set_linear_merge_weights  # noqa: E402
from validate_gemma2_2b_gemmascope_mlp_sae import (  # noqa: E402
    first_tensor,
    load_sae,
    module_mlp,
    remove_hooks,
    select_sae_file,
)


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_bundle_delta_geometry_v0"


def read_bundles(paths: list[Path]) -> dict[str, dict[int, list[int]]]:
    merged: dict[str, dict[int, list[int]]] = {}
    for path in paths:
        parsed = parse_bundles(path.read_text(encoding="utf-8").strip())
        overlap = set(merged).intersection(parsed)
        if overlap:
            raise ValueError(f"duplicate bundle labels across files: {sorted(overlap)}")
        merged.update(parsed)
    return merged


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(torch.linalg.vector_norm(a).item() * torch.linalg.vector_norm(b).item())
    if denom == 0:
        return 0.0
    return float(torch.dot(a, b).item() / denom)


def norm(x: torch.Tensor) -> float:
    return float(torch.linalg.vector_norm(x).item())


@torch.no_grad()
def collect_prompt_acts(model, tokenizer, user: str, args) -> torch.Tensor:
    cache: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        cache["x"] = tensor.detach()

    handle = module_mlp(model, args.layer, args.output_mode).register_forward_hook(hook)
    try:
        prompt = chat_prompt(tokenizer, user)
        enc = tokenizer(prompt, return_tensors="pt").to(args.device)
        prompt_length = int(enc["input_ids"].shape[1])
        model(**enc, use_cache=False)
        mask = patch_position_mask(
            tokenizer,
            enc["input_ids"],
            enc["attention_mask"],
            args.patch_token_filter,
            prompt_length=prompt_length,
        )[0]
        idx = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if idx.numel() != 1:
            raise ValueError(f"expected exactly one patched token for {args.patch_token_filter}, got {idx.numel()}")
        return cache["x"][0, int(idx.item()), :].detach()
    finally:
        remove_hooks([handle])


def bundle_delta(sae, donor_f: torch.Tensor, recipient_f: torch.Tensor, feature_ids: list[int]) -> torch.Tensor:
    if not feature_ids:
        return torch.zeros_like((donor_f - recipient_f) @ sae.W_dec)
    idx = torch.tensor(feature_ids, dtype=torch.long, device=donor_f.device)
    return (donor_f.index_select(-1, idx) - recipient_f.index_select(-1, idx)) @ sae.W_dec.index_select(0, idx)


def write_summary(path: Path, args, rows: list[dict[str, object]]) -> None:
    lines = [
        "# SAE Bundle Delta Geometry",
        "",
        f"Prompt file: `{args.prompt_jsonl}`.",
        f"Donor alpha: `{args.donor_alpha:g}`.",
        f"Recipient alpha: `{args.recipient_alpha:g}`.",
        f"Layer/filter: `{args.layer}:{args.patch_token_filter}`.",
        "",
        "| bundle | split | prompt | features | bundle norm | all-SAE norm frac | dense norm frac | cos dense | cos all-SAE | residual/all-SAE |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['bundle']}` | `{row['split']}` | `{row['prompt']}` | {row['selected_features']} | "
            f"{float(row['bundle_delta_norm']):.4f} | {float(row['bundle_norm_frac_vs_all_sae']):.4f} | "
            f"{float(row['bundle_norm_frac_vs_dense']):.4f} | {float(row['cos_bundle_dense_delta']):.4f} | "
            f"{float(row['cos_bundle_all_sae_delta']):.4f} | {float(row['residual_norm_frac_vs_all_sae']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float32")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--donor-alpha", type=float, default=1.0)
    ap.add_argument("--recipient-alpha", type=float, default=0.75)
    ap.add_argument("--prompt-jsonl", type=Path, required=True)
    ap.add_argument("--patch-token-filter", default="assistant_boundary_final_newline")
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--bundles-file", type=Path, action="append", required=True)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")
    prompts = load_prompt_rows(args.prompt_jsonl)
    bundles = read_bundles(args.bundles_file)

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[load] SAE", flush=True)
    sae = load_sae(select_sae_file(args.layer, args.l0_target), device=args.device, dtype=sae_dtype, cache_dir=cache_dir)

    print("[load] base donor for weight construction", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    base_model.eval()
    print("[load] donor model", flush=True)
    donor_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    donor_model.eval()
    recipient_params_cpu = {name: param.detach().cpu().clone() for name, param in donor_model.named_parameters()}
    set_linear_merge_weights(donor_model, base_model, recipient_params_cpu, args.donor_alpha, args.device)
    print("[load] recipient model", flush=True)
    recipient_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    recipient_model.eval()
    set_linear_merge_weights(recipient_model, base_model, recipient_params_cpu, args.recipient_alpha, args.device)
    del base_model
    torch.cuda.empty_cache()

    rows = []
    for prompt_index, (split, user) in enumerate(prompts, start=1):
        print(f"[acts] prompt {prompt_index}/{len(prompts)} ({split})", flush=True)
        donor_x = collect_prompt_acts(donor_model, tokenizer, user, args).to(args.device).float()
        recipient_x = collect_prompt_acts(recipient_model, tokenizer, user, args).to(args.device).float()
        dense_delta = donor_x - recipient_x
        donor_f = sae.encode(donor_x.unsqueeze(0))[0].float()
        recipient_f = sae.encode(recipient_x.unsqueeze(0))[0].float()
        all_sae_delta = (donor_f - recipient_f) @ sae.W_dec
        dense_norm = norm(dense_delta)
        all_norm = norm(all_sae_delta)
        for label, layer_map in bundles.items():
            feature_ids = layer_map.get(args.layer, [])
            vec = bundle_delta(sae, donor_f, recipient_f, feature_ids)
            vec_norm = norm(vec)
            rows.append(
                {
                    "prompt_index": prompt_index,
                    "split": split,
                    "prompt": user,
                    "bundle": label,
                    "selected_features": len(feature_ids),
                    "dense_delta_norm": dense_norm,
                    "all_sae_delta_norm": all_norm,
                    "bundle_delta_norm": vec_norm,
                    "bundle_norm_frac_vs_dense": vec_norm / dense_norm if dense_norm else 0.0,
                    "bundle_norm_frac_vs_all_sae": vec_norm / all_norm if all_norm else 0.0,
                    "cos_bundle_dense_delta": cosine(vec, dense_delta),
                    "cos_bundle_all_sae_delta": cosine(vec, all_sae_delta),
                    "cos_all_sae_dense_delta": cosine(all_sae_delta, dense_delta),
                    "residual_norm_frac_vs_all_sae": norm(all_sae_delta - vec) / all_norm if all_norm else 0.0,
                }
            )

    csv_path = args.result_dir / "bundle_delta_geometry.csv"
    summary_path = args.result_dir / "BUNDLE_DELTA_GEOMETRY_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(csv_path, rows)
    write_summary(summary_path, args, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "prompt_jsonl": str(args.prompt_jsonl),
                "bundles_files": [str(path) for path in args.bundles_file],
                "donor_alpha": args.donor_alpha,
                "recipient_alpha": args.recipient_alpha,
                "layer": args.layer,
                "patch_token_filter": args.patch_token_filter,
                "outputs": {"csv": str(csv_path), "summary": str(summary_path)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
