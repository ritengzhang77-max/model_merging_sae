#!/usr/bin/env python3
"""Audit selected GemmaScope MLP-SAE feature activations on saved generations."""

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


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_feature_event_audit_v0"


def parse_ints(raw: str) -> list[int]:
    return [int(x.strip()) for x in raw.split(",") if x.strip()]


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
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


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def one_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")


def token_context(tokenizer, ids: list[int], pos: int, radius: int) -> str:
    left = max(0, pos - radius)
    right = min(len(ids), pos + radius + 1)
    return tokenizer.decode(ids[left:right], skip_special_tokens=False).replace("\n", "\\n")


def mean_or_zero(values: torch.Tensor) -> float:
    return float(values.float().mean().item()) if values.numel() else 0.0


def max_or_zero(values: torch.Tensor) -> float:
    return float(values.float().max().item()) if values.numel() else 0.0


def update_top_events(events: list[dict[str, object]], row: dict[str, object], limit: int) -> None:
    events.append(row)
    events.sort(key=lambda item: float(item["value"]), reverse=True)
    if len(events) > limit:
        del events[limit:]


@torch.no_grad()
def collect_feature_values(model, tokenizer, sae, layer: int, feature_ids: list[int], text: str, device: str, output_mode: str):
    cache: dict[str, torch.Tensor] = {}

    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        cache["out"] = tensor.detach()

    handle = module_mlp(model, layer, output_mode).register_forward_hook(hook)
    try:
        enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(device)
        _ = model(**enc, use_cache=False)
        out = cache["out"][0]
        idx = torch.tensor(feature_ids, device=device, dtype=torch.long)
        feats = sae.encode(out).index_select(-1, idx).float().cpu()
        ids = [int(x) for x in enc["input_ids"][0].detach().cpu().tolist()]
    finally:
        remove_hooks([handle])
    return ids, feats


def summarize_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["source_model"]), str(row["split"]), int(row["feature_id"]))
        groups.setdefault(key, []).append(row)
    out = []
    for (source_model, split, feature_id), bucket in sorted(groups.items()):
        n = len(bucket)
        out.append(
            {
                "source_model": source_model,
                "split": split,
                "feature_id": feature_id,
                "n": n,
                "donor_gen_mean": sum(float(row["donor_gen_mean"]) for row in bucket) / n,
                "recipient_gen_mean": sum(float(row["recipient_gen_mean"]) for row in bucket) / n,
                "donor_minus_recipient_gen_mean": sum(float(row["donor_minus_recipient_gen_mean"]) for row in bucket) / n,
                "recipient_minus_donor_gen_mean": sum(float(row["recipient_minus_donor_gen_mean"]) for row in bucket) / n,
                "abs_delta_gen_mean": sum(float(row["abs_delta_gen_mean"]) for row in bucket) / n,
                "donor_gen_max": max(float(row["donor_gen_max"]) for row in bucket),
                "recipient_gen_max": max(float(row["recipient_gen_max"]) for row in bucket),
                "abs_delta_gen_max": max(float(row["abs_delta_gen_max"]) for row in bucket),
            }
        )
    return out


def write_summary(path: Path, aggregate_rows, event_rows, args) -> None:
    lines = [
        "# Gemma-2-2B Linear Merge SAE Feature Event Audit",
        "",
        f"Source records: `{args.records}`.",
        f"Layer: `{args.layer}`.",
        f"Feature IDs: `{','.join(str(x) for x in args.feature_ids)}`.",
        f"Donor alpha: `{args.donor_alpha}`.",
        f"Recipient alpha: `{args.recipient_alpha}`.",
        "",
        "## Aggregate Generation-Token Means",
        "",
        "| source model | split | feature | donor mean | recipient mean | donor-recipient | recipient-donor | abs delta | donor max | recipient max | abs max |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in aggregate_rows:
        lines.append(
            f"| `{row['source_model']}` | `{row['split']}` | {int(row['feature_id'])} | "
            f"{float(row['donor_gen_mean']):.4f} | {float(row['recipient_gen_mean']):.4f} | "
            f"{float(row['donor_minus_recipient_gen_mean']):.4f} | "
            f"{float(row['recipient_minus_donor_gen_mean']):.4f} | "
            f"{float(row['abs_delta_gen_mean']):.4f} | {float(row['donor_gen_max']):.4f} | "
            f"{float(row['recipient_gen_max']):.4f} | {float(row['abs_delta_gen_max']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Exported Events",
            "",
            f"- Exported `{len(event_rows)}` top token-level events.",
            "- Events include token text, context, prompt, source model, split, feature ID, and metric.",
            "- This is an audit substrate. It does not assign semantic labels by itself.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--records", type=Path, required=True)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--donor-alpha", type=float, default=1.0)
    ap.add_argument("--recipient-alpha", type=float, default=0.75)
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--feature-ids", type=parse_ints, default=parse_ints("1293"))
    ap.add_argument("--source-models", default="")
    ap.add_argument("--split", choices=("all", "harmful", "benign"), default="all")
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--event-k", type=int, default=8)
    ap.add_argument("--context-radius", type=int, default=8)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    rows = read_jsonl(args.records)
    if args.source_models.strip():
        keep = {item.strip() for item in args.source_models.split(",") if item.strip()}
        rows = [row for row in rows if str(row.get("model")) in keep]
    if args.split != "all":
        rows = [row for row in rows if row.get("split") == args.split]
    if args.max_records > 0:
        rows = rows[: args.max_records]

    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[files] selecting SAE", flush=True)
    sae_file = select_sae_file(args.layer, args.l0_target)
    print(f"[files] L{args.layer}: {sae_file}", flush=True)
    sae = load_sae(sae_file, device=args.device, dtype=sae_dtype, cache_dir=cache_dir)

    print("[load] base donor for weight construction", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    base_model.eval()
    print("[load] high-alpha donor model", flush=True)
    donor_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    donor_model.eval()
    recipient_params_cpu = {name: param.detach().cpu().clone() for name, param in donor_model.named_parameters()}
    set_linear_merge_weights(donor_model, base_model, recipient_params_cpu, args.donor_alpha, args.device)
    print("[load] low-alpha recipient model", flush=True)
    recipient_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    recipient_model.eval()
    set_linear_merge_weights(recipient_model, base_model, recipient_params_cpu, args.recipient_alpha, args.device)
    del base_model
    torch.cuda.empty_cache()

    detail_rows = []
    top_events: dict[tuple[str, str, int, str], list[dict[str, object]]] = {}
    for row_idx, row in enumerate(rows, start=1):
        print(f"[audit] row {row_idx}/{len(rows)} {row.get('model')} {row.get('split')}", flush=True)
        prompt = chat_prompt(tokenizer, str(row.get("prompt", "")))
        full_text = prompt + str(row.get("text", ""))
        prompt_len = len(tokenizer(prompt, return_tensors="pt")["input_ids"][0])
        ids, donor_feats = collect_feature_values(
            donor_model,
            tokenizer,
            sae,
            args.layer,
            args.feature_ids,
            full_text,
            args.device,
            args.output_mode,
        )
        ids2, recipient_feats = collect_feature_values(
            recipient_model,
            tokenizer,
            sae,
            args.layer,
            args.feature_ids,
            full_text,
            args.device,
            args.output_mode,
        )
        if ids != ids2:
            raise RuntimeError("tokenization mismatch between model passes")
        gen_mask = torch.zeros(len(ids), dtype=torch.bool)
        gen_mask[prompt_len:] = True
        prompt_mask = ~gen_mask
        for col, feature_id in enumerate(args.feature_ids):
            donor_vals = donor_feats[:, col]
            recipient_vals = recipient_feats[:, col]
            delta = donor_vals - recipient_vals
            abs_delta = delta.abs()
            detail = {
                "row_idx": row_idx,
                "source_model": row.get("model"),
                "split": row.get("split"),
                "prompt": row.get("prompt"),
                "feature_id": feature_id,
                "token_count": len(ids),
                "prompt_token_count": int(prompt_len),
                "generation_token_count": int(gen_mask.sum().item()),
                "donor_prompt_mean": mean_or_zero(donor_vals[prompt_mask]),
                "recipient_prompt_mean": mean_or_zero(recipient_vals[prompt_mask]),
                "abs_delta_prompt_mean": mean_or_zero(abs_delta[prompt_mask]),
                "donor_gen_mean": mean_or_zero(donor_vals[gen_mask]),
                "recipient_gen_mean": mean_or_zero(recipient_vals[gen_mask]),
                "donor_minus_recipient_gen_mean": mean_or_zero(delta[gen_mask]),
                "recipient_minus_donor_gen_mean": mean_or_zero((-delta)[gen_mask]),
                "abs_delta_gen_mean": mean_or_zero(abs_delta[gen_mask]),
                "donor_gen_max": max_or_zero(donor_vals[gen_mask]),
                "recipient_gen_max": max_or_zero(recipient_vals[gen_mask]),
                "abs_delta_gen_max": max_or_zero(abs_delta[gen_mask]),
            }
            detail_rows.append(detail)
            event_metrics = {
                "donor_activation": donor_vals,
                "recipient_activation": recipient_vals,
                "donor_minus_recipient": delta,
                "recipient_minus_donor": -delta,
                "abs_delta": abs_delta,
            }
            for metric_name, values in event_metrics.items():
                values = values.clone()
                values[~gen_mask] = 0.0
                k = min(args.event_k, int(values.numel()))
                if k <= 0:
                    continue
                vals, positions = torch.topk(values, k)
                bucket_key = (str(row.get("model")), str(row.get("split")), int(feature_id), metric_name)
                bucket = top_events.setdefault(bucket_key, [])
                for value, pos_tensor in zip(vals.tolist(), positions.tolist()):
                    if value <= 0:
                        continue
                    pos = int(pos_tensor)
                    update_top_events(
                        bucket,
                        {
                            "source_model": row.get("model"),
                            "split": row.get("split"),
                            "prompt": row.get("prompt"),
                            "feature_id": feature_id,
                            "metric": metric_name,
                            "value": float(value),
                            "token_position": pos,
                            "token_id": ids[pos],
                            "token": one_token(tokenizer, ids[pos]),
                            "context": token_context(tokenizer, ids, pos, args.context_radius),
                        },
                        args.event_k,
                    )

    event_rows = []
    for key in sorted(top_events):
        event_rows.extend(top_events[key])
    aggregate_rows = summarize_rows(detail_rows)

    detail_path = args.result_dir / "gemma2_2b_linear_merge_sae_feature_event_detail.csv"
    aggregate_path = args.result_dir / "gemma2_2b_linear_merge_sae_feature_event_aggregate.csv"
    event_path = args.result_dir / "gemma2_2b_linear_merge_sae_feature_event_top_events.jsonl"
    summary_path = args.result_dir / "GEMMA2_2B_LINEAR_MERGE_SAE_FEATURE_EVENT_AUDIT_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(detail_path, detail_rows)
    write_csv(aggregate_path, aggregate_rows)
    write_jsonl(event_path, event_rows)
    write_summary(summary_path, aggregate_rows, event_rows, args)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_file": sae_file,
                "records": str(args.records),
                "layer": args.layer,
                "feature_ids": args.feature_ids,
                "donor_alpha": args.donor_alpha,
                "recipient_alpha": args.recipient_alpha,
                "source_models": args.source_models,
                "split": args.split,
                "max_records": args.max_records,
                "outputs": {
                    "detail": str(detail_path),
                    "aggregate": str(aggregate_path),
                    "events": str(event_path),
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
