#!/usr/bin/env python3
"""Rank SAE features by decoder contribution to a successful full-decode patch."""

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
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))
from run_gemma2_2b_activation_patch_target_loss import MODEL_IDS  # noqa: E402
from run_gemma2_2b_linear_weight_merge_sweep import set_linear_merge_weights  # noqa: E402
from validate_gemma2_2b_gemmascope_mlp_sae import (  # noqa: E402
    chat_prompt,
    first_tensor,
    load_sae,
    module_mlp,
    remove_hooks,
    select_sae_file,
)


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_decoder_contribution_rank_v0"


def load_targets(path: Path, target_model: str, max_records: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if target_model and str(row.get("model", "")) != target_model:
                continue
            split = str(row.get("split", "")).strip()
            prompt = str(row.get("prompt", "")).strip()
            text = str(row.get("text", "")).strip()
            if split not in {"harmful", "benign"}:
                raise ValueError(f"{path}:{line_no}: split must be harmful or benign")
            if not prompt or not text:
                raise ValueError(f"{path}:{line_no}: prompt/text is empty")
            rows.append({"split": split, "prompt": prompt, "text": text})
            if max_records > 0 and len(rows) >= max_records:
                break
    if not rows:
        raise ValueError(f"no target rows found in {path} for model {target_model!r}")
    return rows


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
def collect_layer_acts(model, tokenizer, targets, *, layer: int, device: str, output_mode: str, max_length: int) -> torch.Tensor:
    cache = {}

    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        cache["acts"] = tensor.detach()

    handle = module_mlp(model, layer, output_mode).register_forward_hook(hook)
    parts = []
    try:
        for target in targets:
            cache.clear()
            prompt = chat_prompt(tokenizer, target["prompt"])
            prompt_len = int(tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].shape[1])
            enc = tokenizer(prompt + target["text"], return_tensors="pt", truncation=True, max_length=max_length).to(device)
            _ = model(**enc, use_cache=False)
            valid_len = int(enc["attention_mask"][0].sum().item())
            start = min(prompt_len, valid_len)
            if start < valid_len:
                parts.append(cache["acts"][0, start:valid_len].detach().float().cpu())
    finally:
        remove_hooks([handle])
    if not parts:
        raise ValueError("no generated-token activations collected")
    return torch.cat(parts, dim=0)


def rank_features(sae, high_acts: torch.Tensor, low_acts: torch.Tensor, *, layer: int, top_k: int) -> list[dict[str, object]]:
    device = sae.W_dec.device
    high = high_acts.to(device=device, dtype=sae.W_dec.dtype)
    low = low_acts.to(device=device, dtype=sae.W_dec.dtype)
    donor_f = sae.encode(high).float()
    recipient_f = sae.encode(low).float()
    donor_recon = sae.decode(donor_f.to(dtype=sae.W_dec.dtype)).float()
    write_delta = donor_recon - low.float()
    W_dec = sae.W_dec.float()

    # Sum_t f_j(t) * <decoder_j, full_decode_write_delta(t)>.
    feature_delta_projection = donor_f.T @ write_delta
    alignment = (feature_delta_projection * W_dec).sum(dim=1)
    positive_alignment = torch.clamp(alignment, min=0.0)
    contribution_norm = donor_f.abs().sum(dim=0) * W_dec.norm(dim=1)
    feature_delta_abs = (donor_f - recipient_f).abs().sum(dim=0)
    donor_sum = donor_f.sum(dim=0)
    recipient_sum = recipient_f.sum(dim=0)
    active = (donor_f > 0).float().sum(dim=0)

    score = positive_alignment
    k = min(top_k, int(score.numel()))
    ids = torch.topk(score, k).indices.detach().cpu().tolist()
    rows: list[dict[str, object]] = []
    token_count = int(donor_f.shape[0])
    for rank, feature_id in enumerate(ids, start=1):
        idx = int(feature_id)
        rows.append(
            {
                "rank": rank,
                "layer": layer,
                "feature_id": idx,
                "positive_alignment": float(positive_alignment[idx].item()),
                "signed_alignment": float(alignment[idx].item()),
                "contribution_norm": float(contribution_norm[idx].item()),
                "feature_delta_abs": float(feature_delta_abs[idx].item()),
                "donor_mean": float(donor_sum[idx].item() / max(token_count, 1)),
                "recipient_mean": float(recipient_sum[idx].item() / max(token_count, 1)),
                "active_fraction": float(active[idx].item() / max(token_count, 1)),
                "token_count": token_count,
            }
        )
    return rows


def bundle_string(rows: list[dict[str, object]], layer: int, cutoffs: tuple[int, ...]) -> str:
    parts = []
    ids = [int(row["feature_id"]) for row in rows]
    for cutoff in cutoffs:
        selected = ids[: min(cutoff, len(ids))]
        spec = ",".join(f"{layer}:{feature_id}" for feature_id in selected)
        parts.append(f"decoder_top{cutoff}={spec}")
    return ";".join(parts)


def write_summary(path: Path, rows: list[dict[str, object]], bundles: str, args) -> None:
    lines = [
        "# Gemma-2-2B Layer-20 Decoder Contribution Ranking",
        "",
        f"Target records: `{args.target_records}`.",
        f"Target model: `{args.target_model}`.",
        f"Low/high alphas: `{args.low_alpha:g}` -> `{args.high_alpha:g}`.",
        f"Layer: `{args.layer}`.",
        "",
        "## Bundles",
        "",
        "```text",
        bundles,
        "```",
        "",
        "## Top Features",
        "",
        "| rank | feature | positive alignment | signed alignment | contribution norm | feature delta abs | donor mean | recipient mean | active fraction |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows[:50]:
        lines.append(
            f"| {row['rank']} | {row['feature_id']} | {row['positive_alignment']:.6g} | "
            f"{row['signed_alignment']:.6g} | {row['contribution_norm']:.6g} | "
            f"{row['feature_delta_abs']:.6g} | {row['donor_mean']:.6g} | "
            f"{row['recipient_mean']:.6g} | {row['active_fraction']:.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_cutoffs(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(",") if x.strip())


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--target-records", type=Path, required=True)
    ap.add_argument("--target-model", default="bundle_patch_l20_full_decode")
    ap.add_argument("--max-records", type=int, default=1)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--low-alpha", type=float, default=0.75)
    ap.add_argument("--high-alpha", type=float, default=1.0)
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--top-k", type=int, default=500)
    ap.add_argument("--bundle-cutoffs", type=parse_cutoffs, default=parse_cutoffs("50,100,200,500"))
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")
    targets = load_targets(args.target_records, args.target_model, args.max_records)

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[files] selecting SAE", flush=True)
    sae_path = select_sae_file(args.layer, args.l0_target)
    print(f"[files] L{args.layer}: {sae_path}", flush=True)
    sae = load_sae(sae_path, dtype=sae_dtype, device=args.device, cache_dir=cache_dir)

    print("[load] base donor", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir
    ).to(args.device)
    base_model.eval()

    print("[load] merge model", flush=True)
    merge_model = AutoModelForCausalLM.from_pretrained(
        MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir
    ).to(args.device)
    merge_model.eval()
    recipient_params_cpu = {name: param.detach().cpu().clone() for name, param in merge_model.named_parameters()}

    print(f"[acts] high alpha {args.high_alpha:g}", flush=True)
    set_linear_merge_weights(merge_model, base_model, recipient_params_cpu, args.high_alpha, args.device)
    high_acts = collect_layer_acts(
        merge_model,
        tokenizer,
        targets,
        layer=args.layer,
        device=args.device,
        output_mode=args.output_mode,
        max_length=args.max_length,
    )

    print(f"[acts] low alpha {args.low_alpha:g}", flush=True)
    set_linear_merge_weights(merge_model, base_model, recipient_params_cpu, args.low_alpha, args.device)
    low_acts = collect_layer_acts(
        merge_model,
        tokenizer,
        targets,
        layer=args.layer,
        device=args.device,
        output_mode=args.output_mode,
        max_length=args.max_length,
    )

    rows = rank_features(sae, high_acts, low_acts, layer=args.layer, top_k=args.top_k)
    bundles = bundle_string(rows, args.layer, args.bundle_cutoffs)
    csv_path = args.result_dir / "gemma2_2b_l20_decoder_contribution_features.csv"
    summary_path = args.result_dir / "GEMMA2_2B_L20_DECODER_CONTRIBUTION_RANKING.md"
    bundle_path = args.result_dir / "bundles.txt"
    write_csv(csv_path, rows)
    write_summary(summary_path, rows, bundles, args)
    bundle_path.write_text(bundles + "\n", encoding="utf-8")
    manifest = {
        "target_records": str(args.target_records),
        "target_model": args.target_model,
        "max_records": args.max_records,
        "layer": args.layer,
        "l0_target": args.l0_target,
        "low_alpha": args.low_alpha,
        "high_alpha": args.high_alpha,
        "output_mode": args.output_mode,
        "max_length": args.max_length,
        "top_k": args.top_k,
        "bundle_cutoffs": list(args.bundle_cutoffs),
    }
    (args.result_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(f"[save] {summary_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
