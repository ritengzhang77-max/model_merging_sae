#!/usr/bin/env python3
"""Generation validation for second-stage residual PCA patches."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
from analyze_qwen1_5b_refusal_rq1_rq2 import MODEL_IDS  # noqa: E402
from run_qwen1_5b_dynamic_activation_patch_generation import chat_prompt  # noqa: E402
from run_qwen1_5b_lowdim_activation_patch_target_loss import (  # noqa: E402
    DEFAULT_LAYERS,
    build_bases,
    collect_basis_stats,
    first_tensor,
    make_batches as make_primary_basis_batches,
    parse_ints,
    remove_hooks,
)
from run_qwen1_5b_pca_residual_patch_generation import (  # noqa: E402
    parse_layer_spec,
    prompts_for_mode,
)
from run_qwen1_5b_second_stage_residual_pca import (  # noqa: E402
    build_residual_bases,
    collect_residual_rows,
    install_second_stage_hooks,
    make_harmful_batches,
    residual_basis_prompts,
)
from screen_chat_merge_candidate import clean_assistant_text, generate, score_record  # noqa: E402


RESULT_DIR = ROOT / "stage2" / "results" / "qwen1_5b_second_stage_residual_pca_generation"


def parse_variant(raw: str, residual_layers_label: str):
    if raw == "pca64":
        return "pca64", None, None, ()
    if raw == "residual_mean_dir":
        return f"pca64_plus_residual_mean_dir_{residual_layers_label}", "mean_dir", None, ()
    if raw == "residual_mean_vec":
        return f"pca64_plus_residual_mean_vec_{residual_layers_label}", "mean_vec", None, ()
    match = re.fullmatch(r"residual_topk(\d+)", raw)
    if match:
        k = int(match.group(1))
        return f"pca64_plus_residual_topk{k}_{residual_layers_label}", "top_neuron", k, ()
    match = re.fullmatch(r"residual_(centered|raw)_pca(\d+)", raw)
    if match:
        kind = "centered_pca" if match.group(1) == "centered" else "raw_pca"
        rank = int(match.group(2))
        return f"pca64_plus_residual_{match.group(1)}_pca{rank}_{residual_layers_label}", kind, rank, ()
    if raw.startswith("full_"):
        spec = raw[len("full_") :]
        return f"pca64_plus_full_{spec}", None, None, parse_layer_spec(spec)
    raise ValueError(f"Unknown variant: {raw}")


def summarize(model_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    out = {"model": model_name, "n": len(rows)}
    for split in ("harmful", "benign"):
        group = [row for row in rows if row["split"] == split]
        out[f"{split}_ok_rate"] = sum(bool(row["ok"]) for row in group) / max(len(group), 1)
        out[f"{split}_clean_generation_rate"] = sum(bool(row["clean_generation"]) for row in group) / max(len(group), 1)
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    out["harmful_attempted_refusal_rate"] = sum(bool(row.get("harmful_attempted_refusal")) for row in harmful) / max(
        len(harmful), 1
    )
    out["harmful_bad_attempt_rate"] = sum(bool(row.get("harmful_bad_refusal_attempt")) for row in harmful) / max(
        len(harmful), 1
    )
    out["harmful_unsafe_continuation_rate"] = sum(
        bool(row.get("harmful_unsafe_continuation")) for row in harmful
    ) / max(len(harmful), 1)
    out["benign_over_refusal_rate"] = sum(bool(row.get("benign_over_refusal")) for row in benign) / max(len(benign), 1)
    return out


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def write_summary(path: Path, rows: list[dict[str, object]], prompt_modes: tuple[str, ...]) -> None:
    lines = [
        "# Qwen2.5-1.5B Second-Stage Residual PCA Generation",
        "",
        f"Prompt modes: `{', '.join(prompt_modes)}`",
        "",
        "| prompt mode | model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['prompt_mode']}` | `{row['model']}` | {row.get('harmful_ok_rate', 0.0):.3f} | "
            f"{row.get('harmful_attempted_refusal_rate', 0.0):.3f} | "
            f"{row.get('harmful_bad_attempt_rate', 0.0):.3f} | "
            f"{row.get('harmful_unsafe_continuation_rate', 0.0):.3f} | "
            f"{row.get('benign_ok_rate', 0.0):.3f} | {row.get('benign_over_refusal_rate', 0.0):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def second_stage_generate(
    donor,
    recipient,
    tokenizer,
    user: str,
    *,
    device: str,
    max_new_tokens: int,
    layers: tuple[int, ...],
    full_layers: tuple[int, ...],
    primary_bases,
    pca_rank: int,
    residual_bases,
    residual_kind: str | None,
    residual_rank: int | None,
) -> str:
    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    cache, donor_handles, recipient_handles = install_second_stage_hooks(
        donor,
        recipient,
        layers,
        full_layers,
        primary_bases,
        pca_rank,
        residual_bases,
        residual_kind,
        residual_rank,
    )
    try:
        for _ in range(max_new_tokens):
            cache.clear()
            _ = donor(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            out = recipient(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)
            if int(next_id.item()) == tokenizer.eos_token_id:
                break
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)
    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    return clean_assistant_text(tokenizer, prompt, decoded)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt-modes", default="heldout_failures_harmful_only,stress_permission_harmful_only")
    ap.add_argument("--variants", default="pca64,residual_topk256,residual_topk512,residual_mean_dir,residual_raw_pca32,residual_centered_pca64,full_16-23")
    ap.add_argument("--residual-basis-mode", default="residual_targets")
    ap.add_argument("--examples-per-split", type=int, default=12)
    ap.add_argument("--basis-examples-per-split", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--pca-rank", type=int, default=64)
    ap.add_argument("--residual-layers", default="16-23")
    ap.add_argument("--residual-ranks", default="1,2,4,8,16,32,64")
    ap.add_argument("--residual-top-ks", default="64,256,512,1024")
    ap.add_argument("--max-pca-rows-per-layer", type=int, default=512)
    ap.add_argument("--max-residual-rows-per-layer", type=int, default=512)
    ap.add_argument("--residual-basis-mask", choices=("all", "target"), default="all")
    ap.add_argument("--pca-oversample", type=int, default=8)
    ap.add_argument("--pca-n-iter", type=int, default=1)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--include-baselines", action="store_true")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_ints(args.layers)
    residual_layers = parse_layer_spec(args.residual_layers)
    residual_ranks = parse_ints(args.residual_ranks)
    residual_top_ks = parse_ints(args.residual_top_ks)
    prompt_modes = tuple(item.strip() for item in args.prompt_modes.split(",") if item.strip())
    variant_specs = tuple(item.strip() for item in args.variants.split(",") if item.strip())

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    primary_basis_batches = make_primary_basis_batches(
        tokenizer,
        args.basis_examples_per_split,
        args.max_length,
        args.batch_size,
    )
    residual_prompts = residual_basis_prompts(args.residual_basis_mode, args.examples_per_split)
    residual_batches = make_harmful_batches(tokenizer, residual_prompts, args.max_length, args.batch_size)

    print("[load] base donor", flush=True)
    donor = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=torch.float16).to(args.device)
    donor.eval()
    print("[load] abliterated recipient", flush=True)
    recipient = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=torch.float16).to(args.device)
    recipient.eval()

    print("[basis] collect primary PCA64 calibration activations", flush=True)
    primary_stats = collect_basis_stats(
        donor,
        recipient,
        primary_basis_batches,
        layers,
        device=args.device,
        max_rows_per_layer=args.max_pca_rows_per_layer,
    )
    primary_bases = build_bases(
        primary_stats,
        layers,
        (),
        (args.pca_rank,),
        (),
        random_seed=args.random_seed,
        pca_oversample=args.pca_oversample,
        pca_n_iter=args.pca_n_iter,
    )

    print("[basis] collect second-stage residual rows", flush=True)
    residual_rows, _residual_raw_energy, _residual_n_rows = collect_residual_rows(
        donor,
        recipient,
        residual_batches,
        residual_layers,
        primary_bases,
        args.pca_rank,
        device=args.device,
        max_rows_per_layer=args.max_residual_rows_per_layer,
        basis_mask=args.residual_basis_mask,
    )
    residual_bases, explained_rows = build_residual_bases(
        residual_rows,
        residual_layers,
        residual_ranks,
        residual_top_ks,
        random_seed=args.random_seed + 1009,
        pca_oversample=args.pca_oversample,
        pca_n_iter=args.pca_n_iter,
    )

    variants = [parse_variant(raw, args.residual_layers) for raw in variant_specs]
    if args.include_baselines:
        variants = [("base", "base", None, ()), ("abliterated", "abliterated", None, ())] + variants

    all_records: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for prompt_mode in prompt_modes:
        prompts = prompts_for_mode(prompt_mode, args.examples_per_split)
        for model_name, residual_kind, residual_rank, full_layers in variants:
            print(f"[eval:{prompt_mode}] {model_name}", flush=True)
            records = []
            for split, user, expected in prompts:
                if residual_kind == "base":
                    text = generate(donor, tokenizer, user, device=args.device, max_new_tokens=args.max_new_tokens)
                elif residual_kind == "abliterated":
                    text = generate(recipient, tokenizer, user, device=args.device, max_new_tokens=args.max_new_tokens)
                else:
                    text = second_stage_generate(
                        donor,
                        recipient,
                        tokenizer,
                        user,
                        device=args.device,
                        max_new_tokens=args.max_new_tokens,
                        layers=layers,
                        full_layers=full_layers,
                        primary_bases=primary_bases,
                        pca_rank=args.pca_rank,
                        residual_bases=residual_bases,
                        residual_kind=residual_kind,
                        residual_rank=residual_rank,
                    )
                scores = score_record(split, user, text, expected)
                record = {
                    "prompt_mode": prompt_mode,
                    "model": model_name,
                    "split": split,
                    "user": user,
                    "expected": expected,
                    "generation": text,
                    **scores,
                }
                records.append(record)
                all_records.append(record)
            row = summarize(model_name, records)
            row["prompt_mode"] = prompt_mode
            summary_rows.append(row)

    metrics_path = args.result_dir / "qwen1_5b_second_stage_residual_pca_generation_metrics.csv"
    records_path = args.result_dir / "qwen1_5b_second_stage_residual_pca_generation_records.jsonl"
    explained_path = args.result_dir / "qwen1_5b_second_stage_residual_pca_generation_explained.csv"
    summary_path = args.result_dir / "QWEN1_5B_SECOND_STAGE_RESIDUAL_PCA_GENERATION_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, summary_rows)
    write_jsonl(records_path, all_records)
    write_csv(explained_path, explained_rows)
    write_summary(summary_path, summary_rows, prompt_modes)
    manifest_path.write_text(
        json.dumps(
            {
                "models": {"donor": MODEL_IDS["base"], "recipient": MODEL_IDS["abliterated"]},
                "prompt_modes": prompt_modes,
                "variants": variant_specs,
                "residual_basis_mode": args.residual_basis_mode,
                "residual_basis_prompts": residual_prompts,
                "layers": layers,
                "residual_layers": residual_layers,
                "pca_rank": args.pca_rank,
                "residual_ranks": residual_ranks,
                "residual_top_ks": residual_top_ks,
                "residual_basis_mask": args.residual_basis_mask,
                "outputs": {
                    "metrics": str(metrics_path),
                    "records": str(records_path),
                    "explained": str(explained_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
