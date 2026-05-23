#!/usr/bin/env python3
"""Patch selected SAE feature bundles from high-alpha to low-alpha Gemma merges."""

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
from run_gemma2_2b_linear_weight_merge_sweep import parse_alphas, set_linear_merge_weights  # noqa: E402
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import (  # noqa: E402
    BENIGN_PROMPTS,
    HARMFUL_PROMPTS,
    feature_patch_generate,
    prompt_slice,
    score_record,
    summarize_generation,
)
from screen_chat_merge_candidate import generate  # noqa: E402
from validate_gemma2_2b_gemmascope_mlp_sae import (  # noqa: E402
    SAE_REPO,
    load_sae,
    parse_ints,
    select_sae_file,
)


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_bundle_patch_v0"
DEFAULT_BUNDLES = (
    "f16048=19:16048;"
    "top1=17:4342;"
    "top2=17:4342,17:16011;"
    "top5=17:4342,17:16011,16:16332,18:10415,18:11127;"
    "top10=17:4342,17:16011,16:16332,18:10415,18:11127,15:11128,14:3001,20:14425,18:7189,18:11214"
)


def load_prompt_rows(path: Path) -> list[tuple[str, str]]:
    rows: list[tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            split = str(row.get("split", "")).strip()
            prompt = str(row.get("prompt", row.get("user", ""))).strip()
            if split not in {"harmful", "benign"}:
                raise ValueError(f"{path}:{line_no}: split must be harmful or benign")
            if not prompt:
                raise ValueError(f"{path}:{line_no}: prompt is empty")
            rows.append((split, prompt))
    return rows


def default_prompt_rows(eval_start: int, examples_per_split: int) -> list[tuple[str, str]]:
    return [("harmful", x) for x in prompt_slice(HARMFUL_PROMPTS, eval_start, examples_per_split)] + [
        ("benign", x) for x in prompt_slice(BENIGN_PROMPTS, eval_start, examples_per_split)
    ]


def parse_bundles(raw: str) -> dict[str, dict[int, list[int]]]:
    bundles: dict[str, dict[int, list[int]]] = {}
    for item in raw.split(";"):
        if not item.strip():
            continue
        label, spec = item.split("=", 1)
        layer_map: dict[int, list[int]] = {}
        for part in spec.split(","):
            if not part.strip():
                continue
            layer_raw, feature_raw = part.split(":", 1)
            layer_map.setdefault(int(layer_raw), []).append(int(feature_raw))
        bundles[label.strip()] = {layer: sorted(set(vals)) for layer, vals in layer_map.items()}
    if not bundles:
        raise ValueError("at least one bundle is required")
    return bundles


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def make_selected(bundles: dict[str, dict[int, list[int]]], layers: tuple[int, ...], device: str):
    selected = {}
    variants = []
    count_rows = []
    for label, layer_map in bundles.items():
        variants.append({"label": label, "mode": "mix_decode"})
        selected[label] = {}
        for layer in layers:
            idx = torch.tensor(layer_map.get(layer, []), dtype=torch.long, device=device)
            selected[label][layer] = idx
            count_rows.append(
                {
                    "bundle": label,
                    "layer": layer,
                    "selected_features": int(idx.numel()),
                    "feature_ids": ",".join(str(x) for x in idx.detach().cpu().tolist()),
                }
            )
    return variants, selected, count_rows


@torch.no_grad()
def evaluate_models(high_model, low_model, tokenizer, saes, variants, selected, prompts, args):
    by_model: dict[str, list[dict[str, object]]] = {}
    if not args.skip_baselines:
        for model_name, model in ((f"linear_alpha_{args.recipient_alpha:g}", low_model), (f"linear_alpha_{args.donor_alpha:g}", high_model)):
            rows = []
            print(f"[eval] {model_name}", flush=True)
            for split, user in prompts:
                text = generate(model, tokenizer, str(user), device=args.device, max_new_tokens=args.max_new_tokens)
                record = {"model": model_name, "split": split, "prompt": str(user), "text": text}
                record.update(score_record(split, str(user), text))
                rows.append(record)
            by_model[model_name] = rows
    for variant in variants:
        label = str(variant["label"])
        model_name = f"bundle_patch_{label}"
        rows = []
        print(f"[eval] {model_name}", flush=True)
        for split, user in prompts:
            text = feature_patch_generate(
                high_model,
                low_model,
                tokenizer,
                saes,
                str(user),
                layers=args.layers,
                variant=variant,
                selected=selected,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                output_mode=args.output_mode,
                patch_token_filter=args.patch_token_filter,
            )
            record = {
                "model": model_name,
                "split": split,
                "prompt": str(user),
                "text": text,
                "donor_alpha": args.donor_alpha,
                "recipient_alpha": args.recipient_alpha,
                "patch_token_filter": args.patch_token_filter,
            }
            record.update(score_record(split, str(user), text))
            rows.append(record)
        by_model[model_name] = rows
    metrics = []
    for model_name, rows in by_model.items():
        row = summarize_generation(model_name, rows)
        row["donor_alpha"] = args.donor_alpha
        row["recipient_alpha"] = args.recipient_alpha
        row["patch_token_filter"] = args.patch_token_filter
        metrics.append(row)
    return metrics, [row for rows in by_model.values() for row in rows]


def write_summary(path: Path, metrics, count_rows, args, prompt_source: str) -> None:
    lines = [
        "# Gemma-2-2B Linear Merge SAE Bundle Patch",
        "",
        f"Donor alpha: `{args.donor_alpha:g}`.",
        f"Recipient alpha: `{args.recipient_alpha:g}`.",
        f"Patch token filter: `{args.patch_token_filter}`.",
        f"Prompts: {prompt_source}.",
        "",
        "## Generation",
        "",
        "| model | harmful clean | harmful attempt | unsafe | benign helpful | benign over-refusal |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| `{row['model']}` | {row.get('harmful_ok_rate', 0.0):.3f} | "
            f"{row.get('harmful_attempted_refusal_rate', 0.0):.3f} | "
            f"{row.get('harmful_unsafe_continuation_rate', 0.0):.3f} | "
            f"{row.get('benign_ok_rate', 0.0):.3f} | {row.get('benign_over_refusal_rate', 0.0):.3f} |"
        )
    lines.extend(["", "## Bundle Sizes", "", "| bundle | selected features |", "|---|---:|"])
    by_bundle: dict[str, int] = {}
    for row in count_rows:
        by_bundle[str(row["bundle"])] = by_bundle.get(str(row["bundle"]), 0) + int(row["selected_features"])
    for label, count in by_bundle.items():
        lines.append(f"| `{label}` | {count} |")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--layers", default="14,15,16,17,18,19,20")
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--donor-alpha", type=float, default=0.75)
    ap.add_argument("--recipient-alpha", type=float, default=0.25)
    ap.add_argument("--eval-start", type=int, default=0)
    ap.add_argument("--examples-per-split", type=int, default=12)
    ap.add_argument("--prompt-jsonl", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--patch-token-filter", default="assistant_boundary_or_generated")
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--bundles", default=DEFAULT_BUNDLES)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = ap.parse_args()
    args.layers = parse_ints(args.layers)
    return args


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")
    prompts = load_prompt_rows(args.prompt_jsonl) if args.prompt_jsonl else default_prompt_rows(args.eval_start, args.examples_per_split)
    prompt_source = str(args.prompt_jsonl) if args.prompt_jsonl else f"`{args.eval_start}:{args.eval_start + args.examples_per_split}` per split"
    bundles = parse_bundles(args.bundles)

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[files] selecting GemmaScope MLP SAEs", flush=True)
    files = {layer: select_sae_file(layer, args.l0_target) for layer in args.layers}
    for layer, filename in files.items():
        print(f"[files] L{layer}: {filename}", flush=True)
    print("[load] SAEs", flush=True)
    saes = {layer: load_sae(filename, device=args.device, dtype=sae_dtype, cache_dir=cache_dir) for layer, filename in files.items()}

    print("[load] base donor for weight construction", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    base_model.eval()
    print("[load] high-alpha donor model", flush=True)
    high_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    high_model.eval()
    recipient_params_cpu = {name: param.detach().cpu().clone() for name, param in high_model.named_parameters()}
    set_linear_merge_weights(high_model, base_model, recipient_params_cpu, args.donor_alpha, args.device)
    print("[load] low-alpha recipient model", flush=True)
    low_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    low_model.eval()
    set_linear_merge_weights(low_model, base_model, recipient_params_cpu, args.recipient_alpha, args.device)
    del base_model
    torch.cuda.empty_cache()

    variants, selected, count_rows = make_selected(bundles, args.layers, args.device)
    metrics, records = evaluate_models(high_model, low_model, tokenizer, saes, variants, selected, prompts, args)

    metrics_path = args.result_dir / "gemma2_2b_linear_merge_sae_bundle_patch_metrics.csv"
    records_path = args.result_dir / "gemma2_2b_linear_merge_sae_bundle_patch_records.jsonl"
    counts_path = args.result_dir / "gemma2_2b_linear_merge_sae_bundle_patch_counts.csv"
    summary_path = args.result_dir / "GEMMA2_2B_LINEAR_MERGE_SAE_BUNDLE_PATCH_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, metrics)
    write_jsonl(records_path, records)
    write_csv(counts_path, count_rows)
    write_summary(summary_path, metrics, count_rows, args, prompt_source)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "donor_alpha": args.donor_alpha,
                "recipient_alpha": args.recipient_alpha,
                "patch_token_filter": args.patch_token_filter,
                "bundles": bundles,
                "prompt_source": prompt_source,
                "outputs": {
                    "metrics": str(metrics_path),
                    "records": str(records_path),
                    "counts": str(counts_path),
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
