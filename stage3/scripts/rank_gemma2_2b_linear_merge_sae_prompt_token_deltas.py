#!/usr/bin/env python3
"""Rank GemmaScope SAE features by donor-recipient delta at prompt token masks."""

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
from run_gemma2_2b_linear_merge_sae_bundle_patch import load_prompt_rows  # noqa: E402
from run_gemma2_2b_linear_weight_merge_sweep import set_linear_merge_weights  # noqa: E402
from validate_gemma2_2b_gemmascope_mlp_sae import (  # noqa: E402
    first_tensor,
    load_sae,
    module_mlp,
    remove_hooks,
    select_sae_file,
)


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_prompt_token_delta_rank_v0"


def parse_cutoffs(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(",") if x.strip())


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = list(rows[0])
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_features(model, tokenizer, sae, prompts, args) -> torch.Tensor:
    cache: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        cache["acts"] = tensor.detach()

    handle = module_mlp(model, args.layer, args.output_mode).register_forward_hook(hook)
    parts = []
    try:
        for split, user in prompts:
            cache.clear()
            prompt = chat_prompt(tokenizer, user)
            enc = tokenizer(prompt, return_tensors="pt").to(args.device)
            model(**enc, use_cache=False)
            mask = patch_position_mask(
                tokenizer,
                enc["input_ids"],
                enc["attention_mask"],
                args.patch_token_filter,
                prompt_length=int(enc["input_ids"].shape[1]),
            )[0]
            if not bool(mask.any().item()):
                raise ValueError(f"empty mask for prompt: {user}")
            feats = sae.encode(cache["acts"][:, mask, :]).detach().float().cpu()
            parts.append(feats.reshape(-1, feats.shape[-1]))
    finally:
        remove_hooks([handle])
    return torch.cat(parts, dim=0)


def rank_rows(donor_f: torch.Tensor, recipient_f: torch.Tensor, layer: int, top_k: int) -> list[dict[str, object]]:
    delta = donor_f - recipient_f
    abs_delta = delta.abs().sum(dim=0)
    signed_delta = delta.sum(dim=0)
    donor_sum = donor_f.sum(dim=0)
    recipient_sum = recipient_f.sum(dim=0)
    donor_active = (donor_f > 0).float().sum(dim=0)
    recipient_active = (recipient_f > 0).float().sum(dim=0)
    k = min(top_k, int(abs_delta.numel()))
    ids = torch.topk(abs_delta, k).indices.tolist()
    token_count = int(donor_f.shape[0])
    rows = []
    for rank, feature_id in enumerate(ids, start=1):
        idx = int(feature_id)
        rows.append(
            {
                "rank": rank,
                "layer": layer,
                "feature_id": idx,
                "abs_delta_sum": float(abs_delta[idx].item()),
                "signed_delta_sum": float(signed_delta[idx].item()),
                "donor_mean": float(donor_sum[idx].item() / max(token_count, 1)),
                "recipient_mean": float(recipient_sum[idx].item() / max(token_count, 1)),
                "donor_active_fraction": float(donor_active[idx].item() / max(token_count, 1)),
                "recipient_active_fraction": float(recipient_active[idx].item() / max(token_count, 1)),
                "token_count": token_count,
            }
        )
    return rows


def bundle_string(rows: list[dict[str, object]], layer: int, cutoffs: tuple[int, ...]) -> str:
    ids = [int(row["feature_id"]) for row in rows]
    parts = []
    for cutoff in cutoffs:
        selected = ids[: min(cutoff, len(ids))]
        spec = ",".join(f"{layer}:{feature_id}" for feature_id in selected)
        parts.append(f"prompt_delta_top{cutoff}={spec}")
    return ";".join(parts)


def write_summary(path: Path, args, rows, bundles: str) -> None:
    lines = [
        "# Gemma-2-2B Prompt-Token SAE Delta Ranking",
        "",
        f"Prompt file: `{args.prompt_jsonl}`.",
        f"Prompt split filter: `{args.split}`.",
        f"Layer/output: `{args.layer}:{args.output_mode}`.",
        f"Patch token filter: `{args.patch_token_filter}`.",
        f"Alphas: donor `{args.donor_alpha:g}`, recipient `{args.recipient_alpha:g}`.",
        "",
        "## Bundles",
        "",
        "```text",
        bundles,
        "```",
        "",
        "## Top Features",
        "",
        "| rank | feature | abs delta sum | signed delta sum | donor mean | recipient mean | donor active | recipient active |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:50]:
        lines.append(
            f"| {row['rank']} | {row['feature_id']} | {row['abs_delta_sum']:.6g} | "
            f"{row['signed_delta_sum']:.6g} | {row['donor_mean']:.6g} | "
            f"{row['recipient_mean']:.6g} | {row['donor_active_fraction']:.3f} | "
            f"{row['recipient_active_fraction']:.3f} |"
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
    ap.add_argument("--split", choices=("all", "harmful", "benign"), default="all")
    ap.add_argument("--patch-token-filter", default="assistant_boundary_final_newline")
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--top-k", type=int, default=16000)
    ap.add_argument("--bundle-cutoffs", type=parse_cutoffs, default=parse_cutoffs("10,50,100,200,500,1000,2000,4000,8000,12000,16000"))
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")
    prompts = load_prompt_rows(args.prompt_jsonl)
    if args.split != "all":
        prompts = [(split, user) for split, user in prompts if split == args.split]
    if not prompts:
        raise ValueError(f"No prompts remain after split filter: {args.split}")

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[load] SAE", flush=True)
    sae = load_sae(select_sae_file(args.layer, args.l0_target), device=args.device, dtype=sae_dtype, cache_dir=cache_dir)

    print("[load] base donor", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    base_model.eval()
    print("[load] merge model", flush=True)
    merge_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    merge_model.eval()
    recipient_params_cpu = {name: param.detach().cpu().clone() for name, param in merge_model.named_parameters()}

    print("[features] donor", flush=True)
    set_linear_merge_weights(merge_model, base_model, recipient_params_cpu, args.donor_alpha, args.device)
    donor_f = collect_features(merge_model, tokenizer, sae, prompts, args)
    print("[features] recipient", flush=True)
    set_linear_merge_weights(merge_model, base_model, recipient_params_cpu, args.recipient_alpha, args.device)
    recipient_f = collect_features(merge_model, tokenizer, sae, prompts, args)

    rows = rank_rows(donor_f, recipient_f, args.layer, args.top_k)
    bundles = bundle_string(rows, args.layer, args.bundle_cutoffs)
    csv_path = args.result_dir / "prompt_token_delta_features.csv"
    bundle_path = args.result_dir / "bundles.txt"
    summary_path = args.result_dir / "PROMPT_TOKEN_DELTA_RANKING_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(csv_path, rows)
    bundle_path.write_text(bundles + "\n", encoding="utf-8")
    write_summary(summary_path, args, rows, bundles)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_file": select_sae_file(args.layer, args.l0_target),
                "prompt_jsonl": str(args.prompt_jsonl),
                "split": args.split,
                "layer": args.layer,
                "patch_token_filter": args.patch_token_filter,
                "donor_alpha": args.donor_alpha,
                "recipient_alpha": args.recipient_alpha,
                "top_k": args.top_k,
                "bundle_cutoffs": list(args.bundle_cutoffs),
                "outputs": {
                    "features": str(csv_path),
                    "bundles": str(bundle_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
