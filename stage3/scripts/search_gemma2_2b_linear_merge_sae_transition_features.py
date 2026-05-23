#!/usr/bin/env python3
"""Search SAE features that move across the Gemma linear-merge transition."""

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
from run_gemma2_2b_linear_weight_merge_sweep import parse_alphas, set_linear_merge_weights  # noqa: E402
from validate_gemma2_2b_gemmascope_mlp_sae import (  # noqa: E402
    SAE_REPO,
    chat_prompt,
    first_tensor,
    load_sae,
    module_mlp,
    parse_ints,
    remove_hooks,
    select_sae_file,
)


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_transition_feature_search_v0"


def load_targets(path: Path, target_alpha: float, target_model: str, max_records: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    want = round(float(target_alpha), 8)
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if target_model:
                if str(row.get("model", "")) != target_model:
                    continue
            else:
                if round(float(row.get("alpha", -999.0)), 8) != want:
                    continue
            if str(row.get("split", "")) not in {"harmful", "benign"}:
                raise ValueError(f"{path}:{line_no}: split must be harmful or benign")
            if not str(row.get("prompt", "")).strip() or not str(row.get("text", "")).strip():
                raise ValueError(f"{path}:{line_no}: prompt/text is empty")
            rows.append(row)
            if max_records > 0 and len(rows) >= max_records:
                break
    if not rows:
        target_desc = f"model {target_model}" if target_model else f"target alpha {target_alpha}"
        raise ValueError(f"no target rows found in {path} for {target_desc}")
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_stats_for_model_alpha(
    model,
    tokenizer,
    saes,
    layers: tuple[int, ...],
    targets: list[dict[str, object]],
    *,
    model_alpha: float,
    device: str,
    output_mode: str,
    max_length: int,
) -> dict[tuple[str, int], dict[str, object]]:
    sums: dict[tuple[str, int], torch.Tensor] = {}
    nonzeros: dict[tuple[str, int], torch.Tensor] = {}
    counts: dict[tuple[str, int], int] = {}
    cache: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            cache[layer] = tensor.detach()

        return hook

    for layer in layers:
        handles.append(module_mlp(model, layer, output_mode).register_forward_hook(make_hook(layer)))

    try:
        for target in targets:
            cache.clear()
            split = str(target["split"])
            prompt = chat_prompt(tokenizer, str(target["prompt"]))
            prompt_len = int(tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].shape[1])
            enc = tokenizer(prompt + str(target["text"]), return_tensors="pt", truncation=True, max_length=max_length).to(device)
            input_ids = enc["input_ids"]
            attention_mask = enc["attention_mask"]
            _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            valid_len = int(attention_mask[0].sum().item())
            start = min(prompt_len, valid_len)
            if start >= valid_len:
                continue
            for layer in layers:
                acts = saes[layer].encode(cache[layer][0, start:valid_len]).float()
                key = (split, layer)
                if key not in sums:
                    sums[key] = torch.zeros(acts.shape[1], dtype=torch.float32, device="cpu")
                    nonzeros[key] = torch.zeros(acts.shape[1], dtype=torch.float32, device="cpu")
                    counts[key] = 0
                sums[key] += acts.sum(dim=0).detach().cpu()
                nonzeros[key] += (acts > 0).float().sum(dim=0).detach().cpu()
                counts[key] += int(acts.shape[0])
    finally:
        remove_hooks(handles)

    out: dict[tuple[str, int], dict[str, object]] = {}
    for key, total in sums.items():
        count = counts[key]
        out[key] = {
            "model_alpha": model_alpha,
            "split": key[0],
            "layer": key[1],
            "mean": total / max(count, 1),
            "nonzero_rate": nonzeros[key] / max(count, 1),
            "count": count,
        }
    return out


def rank_features(
    stats_by_alpha: dict[float, dict[tuple[str, int], dict[str, object]]],
    layers: tuple[int, ...],
    *,
    low_alpha: float,
    high_alpha: float,
    top_k_per_layer: int,
    top_k_global: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    per_layer_rows: list[dict[str, object]] = []
    all_rows: list[dict[str, object]] = []
    low = stats_by_alpha[low_alpha]
    high = stats_by_alpha[high_alpha]
    for layer in layers:
        harm_low = low[("harmful", layer)]["mean"]
        harm_high = high[("harmful", layer)]["mean"]
        benign_low = low[("benign", layer)]["mean"]
        benign_high = high[("benign", layer)]["mean"]
        harm_nz_low = low[("harmful", layer)]["nonzero_rate"]
        harm_nz_high = high[("harmful", layer)]["nonzero_rate"]
        harm_delta = harm_high - harm_low
        benign_delta = benign_high - benign_low
        specificity = harm_delta - benign_delta.abs()
        scores = [
            ("harm_delta", harm_delta),
            ("specificity", specificity),
        ]
        seen: set[int] = set()
        for rank_name, score in scores:
            k = min(top_k_per_layer, int(score.numel()))
            for rank, idx in enumerate(torch.topk(score, k).indices.detach().cpu().tolist(), start=1):
                row = {
                    "rank_type": rank_name,
                    "rank": rank,
                    "layer": layer,
                    "feature_id": int(idx),
                    "low_alpha": low_alpha,
                    "high_alpha": high_alpha,
                    "harm_mean_low": float(harm_low[idx].item()),
                    "harm_mean_high": float(harm_high[idx].item()),
                    "harm_delta": float(harm_delta[idx].item()),
                    "benign_mean_low": float(benign_low[idx].item()),
                    "benign_mean_high": float(benign_high[idx].item()),
                    "benign_delta": float(benign_delta[idx].item()),
                    "specificity": float(specificity[idx].item()),
                    "harm_nonzero_low": float(harm_nz_low[idx].item()),
                    "harm_nonzero_high": float(harm_nz_high[idx].item()),
                }
                per_layer_rows.append(row)
                if idx not in seen:
                    all_rows.append(row)
                    seen.add(idx)
    global_rows = sorted(all_rows, key=lambda x: float(x["specificity"]), reverse=True)[:top_k_global]
    for i, row in enumerate(global_rows, start=1):
        row["global_specificity_rank"] = i
    return per_layer_rows, global_rows


def write_summary(path: Path, global_rows: list[dict[str, object]], args) -> None:
    lines = [
        "# Gemma-2-2B Linear Merge SAE Transition Feature Search",
        "",
        f"Target records: `{args.target_records}`.",
        f"Target alpha/model: `{args.target_alpha:g}` / `{args.target_model}`.",
        f"Low/high model alphas: `{args.low_alpha:g}` -> `{args.high_alpha:g}`.",
        f"Layers: `{args.layers}`.",
        "",
        "## Top Global Specificity Features",
        "",
        "| rank | layer | feature | harm low | harm high | harm delta | benign delta | specificity | harm nz low | harm nz high |",
        "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in global_rows[:50]:
        lines.append(
            f"| {row.get('global_specificity_rank', '')} | {row['layer']} | {row['feature_id']} | "
            f"{row['harm_mean_low']:.4f} | {row['harm_mean_high']:.4f} | {row['harm_delta']:.4f} | "
            f"{row['benign_delta']:.4f} | {row['specificity']:.4f} | "
            f"{row['harm_nonzero_low']:.3f} | {row['harm_nonzero_high']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Ranking Rule",
            "",
            "- `harm_delta = harmful_mean(high_alpha) - harmful_mean(low_alpha)`.",
            "- `specificity = harm_delta - abs(benign_delta)`.",
            "- Rows use fixed assistant continuations, so token sequence is held constant across model alphas.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--layers", default="12,13,14,15,16,17,18,19,20")
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--model-alphas", type=parse_alphas, default=parse_alphas("0.25,0.5,0.75"))
    ap.add_argument("--low-alpha", type=float, default=0.25)
    ap.add_argument("--high-alpha", type=float, default=0.75)
    ap.add_argument("--target-alpha", type=float, default=0.75)
    ap.add_argument("--target-model", default="")
    ap.add_argument("--target-records", type=Path, required=True)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--top-k-per-layer", type=int, default=50)
    ap.add_argument("--top-k-global", type=int, default=200)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_ints(args.layers)
    if args.low_alpha not in args.model_alphas or args.high_alpha not in args.model_alphas:
        raise ValueError("low-alpha and high-alpha must both be included in --model-alphas")
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")
    targets = load_targets(args.target_records, args.target_alpha, args.target_model, args.max_records)

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[files] selecting GemmaScope MLP SAEs", flush=True)
    files = {layer: select_sae_file(layer, args.l0_target) for layer in layers}
    for layer, filename in files.items():
        print(f"[files] L{layer}: {filename}", flush=True)
    print("[load] SAEs", flush=True)
    saes = {layer: load_sae(filename, device=args.device, dtype=sae_dtype, cache_dir=cache_dir) for layer, filename in files.items()}

    print("[load] base donor", flush=True)
    donor_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    donor_model.eval()
    print("[load] abliterated recipient/merge model", flush=True)
    merged_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    merged_model.eval()
    recipient_params_cpu = {name: param.detach().cpu().clone() for name, param in merged_model.named_parameters()}

    stats_by_alpha = {}
    for model_alpha in args.model_alphas:
        print(f"[collect] alpha={model_alpha:g}", flush=True)
        set_linear_merge_weights(merged_model, donor_model, recipient_params_cpu, model_alpha, args.device)
        stats_by_alpha[float(model_alpha)] = collect_stats_for_model_alpha(
            merged_model,
            tokenizer,
            saes,
            layers,
            targets,
            model_alpha=float(model_alpha),
            device=args.device,
            output_mode=args.output_mode,
            max_length=args.max_length,
        )
    per_layer_rows, global_rows = rank_features(
        stats_by_alpha,
        layers,
        low_alpha=float(args.low_alpha),
        high_alpha=float(args.high_alpha),
        top_k_per_layer=args.top_k_per_layer,
        top_k_global=args.top_k_global,
    )

    per_layer_path = args.result_dir / "gemma2_2b_linear_merge_sae_transition_feature_per_layer.csv"
    global_path = args.result_dir / "gemma2_2b_linear_merge_sae_transition_feature_global.csv"
    summary_path = args.result_dir / "GEMMA2_2B_LINEAR_MERGE_SAE_TRANSITION_FEATURE_SEARCH_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(per_layer_path, per_layer_rows)
    write_csv(global_path, global_rows)
    write_summary(summary_path, global_rows, args)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "target_records": str(args.target_records),
                "target_alpha": args.target_alpha,
                "target_model": args.target_model,
                "model_alphas": list(args.model_alphas),
                "low_alpha": args.low_alpha,
                "high_alpha": args.high_alpha,
                "layers": list(layers),
                "outputs": {
                    "per_layer": str(per_layer_path),
                    "global": str(global_path),
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
