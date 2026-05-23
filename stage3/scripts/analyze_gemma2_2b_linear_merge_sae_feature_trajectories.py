#!/usr/bin/env python3
"""SAE feature trajectories along the Gemma linear weight-merge line."""

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
    HARMFUL_PROMPTS,
    BENIGN_PROMPTS,
    clean_assistant_text,
    prompt_slice,
    score_record,
)
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


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_feature_trajectories_v0"


def parse_features(raw: str) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        left, right = item.split(":", 1)
        out.setdefault(int(left), []).append(int(right))
    return {layer: sorted(set(vals)) for layer, vals in out.items()}


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


def default_prompt_rows(prompt_start: int, examples_per_split: int) -> list[tuple[str, str]]:
    return [("harmful", x) for x in prompt_slice(HARMFUL_PROMPTS, prompt_start, examples_per_split)] + [
        ("benign", x) for x in prompt_slice(BENIGN_PROMPTS, prompt_start, examples_per_split)
    ]


def one_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")


def position_kind(prompt_length: int, pos: int) -> str:
    if pos >= prompt_length:
        return "generated"
    if pos == prompt_length - 1:
        return "assistant_boundary"
    return "prompt"


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


@torch.no_grad()
def generate_with_feature_trace(
    model,
    tokenizer,
    saes,
    features_by_layer: dict[int, list[int]],
    *,
    alpha: float,
    split: str,
    user: str,
    prompt_index: int,
    device: str,
    max_new_tokens: int,
    output_mode: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    cache: dict[int, torch.Tensor] = {}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            cache[layer] = tensor.detach()

        return hook

    for layer in features_by_layer:
        handles.append(module_mlp(model, layer, output_mode).register_forward_hook(make_hook(layer)))

    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_length = int(input_ids.shape[1])
    trace_rows: list[dict[str, object]] = []
    final_text = ""
    try:
        for step in range(max_new_tokens):
            cache.clear()
            out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            valid_len = int(attention_mask[0].sum().item())
            pos = valid_len - 1
            ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
            kind = position_kind(prompt_length, pos)
            token = one_token(tokenizer, ids[pos])
            for layer, feature_ids in features_by_layer.items():
                acts = saes[layer].encode(cache[layer][0, pos : pos + 1]).float()[0]
                for feature_id in feature_ids:
                    trace_rows.append(
                        {
                            "alpha": alpha,
                            "split": split,
                            "prompt_index": prompt_index,
                            "prompt": user,
                            "step": step,
                            "position": pos,
                            "position_kind": kind,
                            "token_id": ids[pos],
                            "token": token,
                            "layer": layer,
                            "feature_id": feature_id,
                            "activation": float(acts[feature_id].item()),
                        }
                    )
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)
            if int(next_id.item()) == tokenizer.eos_token_id:
                break
        decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
        final_text = clean_assistant_text(tokenizer, prompt, decoded)
    finally:
        remove_hooks(handles)
    record = {"alpha": alpha, "split": split, "prompt_index": prompt_index, "prompt": user, "text": final_text}
    record.update(score_record(split, user, final_text))
    return trace_rows, record


def summarize_trace(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[float, str, int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (
            float(row["alpha"]),
            str(row["split"]),
            int(row["layer"]),
            int(row["feature_id"]),
            str(row["position_kind"]),
        )
        buckets.setdefault(key, []).append(row)
    out = []
    for (alpha, split, layer, feature_id, kind), vals in sorted(buckets.items()):
        activations = [float(x["activation"]) for x in vals]
        out.append(
            {
                "alpha": alpha,
                "split": split,
                "layer": layer,
                "feature_id": feature_id,
                "position_kind": kind,
                "n": len(vals),
                "mean_activation": sum(activations) / len(activations),
                "max_activation": max(activations),
                "nonzero_fraction": sum(x > 0.0 for x in activations) / len(activations),
            }
        )
    return out


def summarize_scores(records: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[float, list[dict[str, object]]] = {}
    for row in records:
        buckets.setdefault(float(row["alpha"]), []).append(row)
    out = []
    for alpha, vals in sorted(buckets.items()):
        harmful = [x for x in vals if x["split"] == "harmful"]
        benign = [x for x in vals if x["split"] == "benign"]
        out.append(
            {
                "alpha": alpha,
                "n": len(vals),
                "harmful_ok_rate": sum(bool(x["ok"]) for x in harmful) / max(len(harmful), 1),
                "harmful_unsafe_continuation_rate": sum(bool(x.get("harmful_unsafe_continuation")) for x in harmful)
                / max(len(harmful), 1),
                "benign_ok_rate": sum(bool(x["ok"]) for x in benign) / max(len(benign), 1),
                "benign_over_refusal_rate": sum(bool(x.get("benign_over_refusal")) for x in benign) / max(len(benign), 1),
            }
        )
    return out


def write_summary(path: Path, score_rows, trace_summary, args, prompt_source: str) -> None:
    lines = [
        "# Gemma-2-2B Linear Merge SAE Feature Trajectories",
        "",
        f"Merge line: `abliterated + alpha * (base - abliterated)`.",
        f"Alphas: `{','.join(f'{x:g}' for x in args.alphas)}`.",
        f"Prompts: {prompt_source}.",
        f"Features: `{args.features}`.",
        "",
        "## Generation",
        "",
        "| alpha | harmful clean | unsafe | benign helpful | benign over-refusal |",
        "|---:|---:|---:|---:|---:|",
    ]
    for row in score_rows:
        lines.append(
            f"| {row['alpha']:.3f} | {row['harmful_ok_rate']:.3f} | "
            f"{row['harmful_unsafe_continuation_rate']:.3f} | {row['benign_ok_rate']:.3f} | "
            f"{row['benign_over_refusal_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Feature Summary",
            "",
            "| alpha | split | layer | feature | position kind | n | mean activation | max activation | nonzero fraction |",
            "|---:|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in trace_summary:
        lines.append(
            f"| {row['alpha']:.3f} | `{row['split']}` | {row['layer']} | {row['feature_id']} | "
            f"`{row['position_kind']}` | {row['n']} | {row['mean_activation']:.4f} | "
            f"{row['max_activation']:.4f} | {row['nonzero_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- These are own-generation trajectories, so activation changes can reflect both model state and generated-token changes.",
            "- The analysis logs the assistant-boundary token at step 0 and the most recent generated token thereafter.",
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
    ap.add_argument("--prompt-start", type=int, default=8)
    ap.add_argument("--examples-per-split", type=int, default=1)
    ap.add_argument("--prompt-jsonl", type=Path, default=None)
    ap.add_argument("--max-new-tokens", type=int, default=64)
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
    prompts = load_prompt_rows(args.prompt_jsonl) if args.prompt_jsonl else default_prompt_rows(args.prompt_start, args.examples_per_split)
    prompt_source = str(args.prompt_jsonl) if args.prompt_jsonl else f"`{args.prompt_start}:{args.prompt_start + args.examples_per_split}` per split"

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

    trace_rows: list[dict[str, object]] = []
    records: list[dict[str, object]] = []
    print("[trajectory] alpha sweep", flush=True)
    for alpha in args.alphas:
        print(f"[merge] alpha={alpha:g}", flush=True)
        set_linear_merge_weights(merged_model, donor_model, recipient_params_cpu, alpha, args.device)
        for prompt_index, (split, user) in enumerate(prompts):
            rows, record = generate_with_feature_trace(
                merged_model,
                tokenizer,
                saes,
                features_by_layer,
                alpha=alpha,
                split=split,
                user=str(user),
                prompt_index=prompt_index,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                output_mode=args.output_mode,
            )
            trace_rows.extend(rows)
            records.append(record)

    score_rows = summarize_scores(records)
    trace_summary = summarize_trace(trace_rows)
    trace_path = args.result_dir / "gemma2_2b_linear_merge_sae_feature_trajectories.csv"
    records_path = args.result_dir / "gemma2_2b_linear_merge_sae_feature_records.jsonl"
    scores_path = args.result_dir / "gemma2_2b_linear_merge_sae_feature_scores.csv"
    summary_csv_path = args.result_dir / "gemma2_2b_linear_merge_sae_feature_summary.csv"
    summary_path = args.result_dir / "GEMMA2_2B_LINEAR_MERGE_SAE_FEATURE_TRAJECTORY_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(trace_path, trace_rows)
    write_jsonl(records_path, records)
    write_csv(scores_path, score_rows)
    write_csv(summary_csv_path, trace_summary)
    write_summary(summary_path, score_rows, trace_summary, args, prompt_source)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "merge_line": "abliterated + alpha * (base - abliterated)",
                "alphas": list(args.alphas),
                "features": features_by_layer,
                "prompt_source": prompt_source,
                "outputs": {
                    "trajectories": str(trace_path),
                    "records": str(records_path),
                    "scores": str(scores_path),
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
