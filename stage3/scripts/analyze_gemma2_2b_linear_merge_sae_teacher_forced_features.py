#!/usr/bin/env python3
"""Teacher-forced SAE feature activations along the Gemma linear merge line."""

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


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_teacher_forced_features_v0"


def parse_features(raw: str) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        left, right = item.split(":", 1)
        out.setdefault(int(left), []).append(int(right))
    return {layer: sorted(set(vals)) for layer, vals in out.items()}


def one_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")


def load_targets(path: Path, target_alphas: tuple[float, ...], max_records: int) -> list[dict[str, object]]:
    keep = {round(float(x), 8) for x in target_alphas}
    rows: list[dict[str, object]] = []
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if round(float(row.get("alpha", -999.0)), 8) not in keep:
                continue
            if str(row.get("split", "")) not in {"harmful", "benign"}:
                raise ValueError(f"{path}:{line_no}: split must be harmful or benign")
            if not str(row.get("prompt", "")).strip() or not str(row.get("text", "")).strip():
                raise ValueError(f"{path}:{line_no}: prompt/text is empty")
            rows.append(row)
            if max_records > 0 and len(rows) >= max_records:
                break
    if not rows:
        raise ValueError(f"no target rows found in {path} for alphas {target_alphas}")
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
def forward_features(
    model,
    tokenizer,
    saes,
    features_by_layer: dict[int, list[int]],
    *,
    model_alpha: float,
    target: dict[str, object],
    target_index: int,
    device: str,
    output_mode: str,
    max_length: int,
) -> list[dict[str, object]]:
    cache: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            cache[layer] = tensor.detach()

        return hook

    for layer in features_by_layer:
        handles.append(module_mlp(model, layer, output_mode).register_forward_hook(make_hook(layer)))

    user = str(target["prompt"])
    assistant_text = str(target["text"])
    prompt = chat_prompt(tokenizer, user)
    prompt_len = int(tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)["input_ids"].shape[1])
    enc = tokenizer(prompt + assistant_text, return_tensors="pt", truncation=True, max_length=max_length).to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    rows: list[dict[str, object]] = []
    try:
        _ = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        valid_len = int(attention_mask[0].sum().item())
        start = min(prompt_len, valid_len)
        ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
        for pos in range(start, valid_len):
            token = one_token(tokenizer, ids[pos])
            for layer, feature_ids in features_by_layer.items():
                acts = saes[layer].encode(cache[layer][0, pos : pos + 1]).float()[0]
                for feature_id in feature_ids:
                    rows.append(
                        {
                            "model_alpha": model_alpha,
                            "target_alpha": float(target["alpha"]),
                            "target_model": str(target.get("model", "")),
                            "split": str(target["split"]),
                            "target_ok": bool(target.get("ok")),
                            "target_unsafe": bool(target.get("harmful_unsafe_continuation")),
                            "target_over_refusal": bool(target.get("benign_over_refusal")),
                            "target_index": target_index,
                            "prompt": user,
                            "position": pos,
                            "assistant_token_index": pos - start,
                            "token": token,
                            "token_id": ids[pos],
                            "layer": layer,
                            "feature_id": feature_id,
                            "activation": float(acts[feature_id].item()),
                        }
                    )
    finally:
        remove_hooks(handles)
    return rows


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[float, float, str, int, int], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            float(row["model_alpha"]),
            float(row["target_alpha"]),
            str(row["split"]),
            int(row["layer"]),
            int(row["feature_id"]),
        )
        buckets.setdefault(key, []).append(row)
    out = []
    for (model_alpha, target_alpha, split, layer, feature_id), vals in sorted(buckets.items()):
        acts = [float(x["activation"]) for x in vals]
        out.append(
            {
                "model_alpha": model_alpha,
                "target_alpha": target_alpha,
                "split": split,
                "layer": layer,
                "feature_id": feature_id,
                "n": len(vals),
                "mean_activation": sum(acts) / len(acts),
                "max_activation": max(acts),
                "nonzero_fraction": sum(x > 0.0 for x in acts) / len(acts),
            }
        )
    return out


def write_summary(path: Path, summary_rows: list[dict[str, object]], args) -> None:
    lines = [
        "# Gemma-2-2B Linear Merge Teacher-Forced SAE Features",
        "",
        f"Merge line: `abliterated + alpha * (base - abliterated)`.",
        f"Model alphas: `{','.join(f'{x:g}' for x in args.alphas)}`.",
        f"Target alphas: `{','.join(f'{x:g}' for x in args.target_alphas)}`.",
        f"Target records: `{args.target_records}`.",
        f"Features: `{args.features}`.",
        "",
        "## Feature Summary",
        "",
        "| model alpha | target alpha | split | layer | feature | n | mean activation | max activation | nonzero fraction |",
        "|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['model_alpha']:.3f} | {row['target_alpha']:.3f} | `{row['split']}` | "
            f"{row['layer']} | {row['feature_id']} | {row['n']} | {row['mean_activation']:.4f} | "
            f"{row['max_activation']:.4f} | {row['nonzero_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- These are fixed assistant continuations sampled from prior model generations.",
            "- The result separates representation changes from generated-token changes, but it does not say whether the fixed continuation is likely under each model.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--alphas", type=parse_alphas, default=parse_alphas("0,0.25,0.5,0.75,1"))
    ap.add_argument("--target-alphas", type=parse_alphas, default=parse_alphas("0,0.75"))
    ap.add_argument("--target-records", type=Path, required=True)
    ap.add_argument("--max-records", type=int, default=0)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--features", default="19:16048,12:40,12:12075")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")
    features_by_layer = parse_features(args.features)
    layers = parse_ints(",".join(str(x) for x in sorted(features_by_layer)))
    targets = load_targets(args.target_records, args.target_alphas, args.max_records)

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

    rows: list[dict[str, object]] = []
    print("[teacher] alpha sweep", flush=True)
    for model_alpha in args.alphas:
        print(f"[merge] alpha={model_alpha:g}", flush=True)
        set_linear_merge_weights(merged_model, donor_model, recipient_params_cpu, model_alpha, args.device)
        for target_index, target in enumerate(targets):
            rows.extend(
                forward_features(
                    merged_model,
                    tokenizer,
                    saes,
                    features_by_layer,
                    model_alpha=model_alpha,
                    target=target,
                    target_index=target_index,
                    device=args.device,
                    output_mode=args.output_mode,
                    max_length=args.max_length,
                )
            )

    summary_rows = summarize(rows)
    rows_path = args.result_dir / "gemma2_2b_linear_merge_sae_teacher_forced_feature_rows.csv"
    summary_csv_path = args.result_dir / "gemma2_2b_linear_merge_sae_teacher_forced_feature_summary.csv"
    summary_path = args.result_dir / "GEMMA2_2B_LINEAR_MERGE_SAE_TEACHER_FORCED_FEATURE_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(rows_path, rows)
    write_csv(summary_csv_path, summary_rows)
    write_summary(summary_path, summary_rows, args)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "merge_line": "abliterated + alpha * (base - abliterated)",
                "model_alphas": list(args.alphas),
                "target_alphas": list(args.target_alphas),
                "target_records": str(args.target_records),
                "features": features_by_layer,
                "outputs": {
                    "rows": str(rows_path),
                    "summary_csv": str(summary_csv_path),
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
