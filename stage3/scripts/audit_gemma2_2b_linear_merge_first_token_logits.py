#!/usr/bin/env python3
"""Audit first assistant-token logits for Gemma-2-2B linear merge alphas."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))
from run_gemma2_2b_activation_patch_target_loss import MODEL_IDS  # noqa: E402
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import chat_prompt  # noqa: E402
from run_gemma2_2b_linear_merge_sae_bundle_patch import load_prompt_rows, write_csv  # noqa: E402
from run_gemma2_2b_linear_weight_merge_sweep import set_linear_merge_weights  # noqa: E402


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_first_token_logits_v0"


def parse_floats(raw: str) -> list[float]:
    return [float(item.strip()) for item in raw.split(",") if item.strip()]


def one_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token: {ids}")
    return int(ids[0])


@torch.no_grad()
def first_token_logits(model, tokenizer, prompt: str, device: str) -> torch.Tensor:
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc, use_cache=False)
    return out.logits[0, -1].detach().float().cpu()


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[float, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((float(row["alpha"]), str(row["split"])), []).append(row)
    out = []
    for (alpha, split), bucket in sorted(groups.items()):
        n = len(bucket)
        out.append(
            {
                "alpha": alpha,
                "split": split,
                "n": n,
                "top_i_rate": sum(1 for row in bucket if row["top_token"] == "I") / n,
                "top_it_rate": sum(1 for row in bucket if row["top_token"] == "It") / n,
                "mean_i_minus_it": sum(float(row["i_minus_it"]) for row in bucket) / n,
                "min_i_minus_it": min(float(row["i_minus_it"]) for row in bucket),
                "max_i_minus_it": max(float(row["i_minus_it"]) for row in bucket),
            }
        )
    return out


def write_summary(path: Path, rows, summary_rows, args) -> None:
    lines = [
        "# Gemma-2-2B Linear Merge First-Token Logit Audit",
        "",
        f"Prompt file: `{args.prompt_jsonl}`.",
        f"Alphas: `{args.alphas}`.",
        "",
        "| alpha | split | n | top I | top It | mean I-It | min I-It | max I-It |",
        "|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {float(row['alpha']):.3f} | `{row['split']}` | {row['n']} | "
            f"{float(row['top_i_rate']):.3f} | {float(row['top_it_rate']):.3f} | "
            f"{float(row['mean_i_minus_it']):.4f} | {float(row['min_i_minus_it']):.4f} | "
            f"{float(row['max_i_minus_it']):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The `I-It` margin measures whether the first assistant-token distribution",
            "leans toward the direct-refusal basin (`I...`) or the warning/procedure",
            "basin (`It...`) before any token is generated.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--prompt-jsonl", type=Path, required=True)
    ap.add_argument("--alphas", default="0.75,1.0")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    alphas = parse_floats(args.alphas)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    cache_dir = os.environ.get("HF_HOME")
    prompts = load_prompt_rows(args.prompt_jsonl)

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_i = one_token_id(tokenizer, "I")
    token_it = one_token_id(tokenizer, "It")

    print("[load] base donor for weight construction", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    base_model.eval()
    print("[load] abliterated endpoint", flush=True)
    endpoint = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    endpoint.eval()
    endpoint_params_cpu = {name: param.detach().cpu().clone() for name, param in endpoint.named_parameters()}

    rows = []
    for alpha in alphas:
        print(f"[alpha] {alpha:g}", flush=True)
        set_linear_merge_weights(endpoint, base_model, endpoint_params_cpu, alpha, args.device)
        for row_idx, (split, user) in enumerate(prompts, start=1):
            print(f"[eval] alpha {alpha:g} prompt {row_idx}/{len(prompts)} ({split})", flush=True)
            prompt = chat_prompt(tokenizer, str(user))
            logits = first_token_logits(endpoint, tokenizer, prompt, args.device)
            top_value, top_id = torch.max(logits, dim=0)
            top_token = tokenizer.decode([int(top_id)], skip_special_tokens=False)
            rows.append(
                {
                    "alpha": alpha,
                    "split": split,
                    "prompt": str(user),
                    "token_i_id": token_i,
                    "token_it_id": token_it,
                    "logit_i": float(logits[token_i].item()),
                    "logit_it": float(logits[token_it].item()),
                    "i_minus_it": float((logits[token_i] - logits[token_it]).item()),
                    "top_token_id": int(top_id.item()),
                    "top_token": top_token,
                    "top_logit": float(top_value.item()),
                }
            )

    detail_path = args.result_dir / "first_token_logits.csv"
    summary_path = args.result_dir / "first_token_logit_summary.csv"
    md_path = args.result_dir / "FIRST_TOKEN_LOGIT_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    summary_rows = summarize(rows)
    write_csv(detail_path, rows)
    write_csv(summary_path, summary_rows)
    write_summary(md_path, rows, summary_rows, args)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "prompt_jsonl": str(args.prompt_jsonl),
                "alphas": alphas,
                "token_i": token_i,
                "token_it": token_it,
                "outputs": {
                    "detail": str(detail_path),
                    "summary": str(summary_path),
                    "markdown": str(md_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {md_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
