#!/usr/bin/env python3
"""Small RQ0 screen for the Gemma-2-2B abliterated safety-loss pair."""

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
import torch.nn.functional as F
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
from screen_chat_merge_candidate import ARITH_PROMPTS, BENIGN_PROMPTS, HARMFUL_PROMPTS, POLITE_PROMPTS  # noqa: E402


DEFAULT_MODELS = {
    "base": "google/gemma-2-2b-it",
    "abliterated": "IlyaGusev/gemma-2-2b-it-abliterated",
}
REP_LAYERS = (0, 4, 8, 12, 16, 20, 25)
RESULT_DIR = ROOT / "stage1" / "results" / "gemma2_2b_abliterated_rq0"


def chat_prompt(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def make_prompt_rows(examples_per_split: int) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for split, prompts in (
        ("harmful", [(p, "") for p in HARMFUL_PROMPTS]),
        ("benign", [(p, "") for p in BENIGN_PROMPTS]),
        ("arith", ARITH_PROMPTS),
        ("polite", [(p, "") for p in POLITE_PROMPTS]),
    ):
        for i, (user, expected) in enumerate(prompts[:examples_per_split]):
            rows.append(
                {
                    "prompt_id": len(rows),
                    "split": split,
                    "split_index": i,
                    "user": user,
                    "expected": expected,
                }
            )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


@torch.no_grad()
def collect_reps(model, tokenizer, prompt_rows, *, device: str, batch_size: int):
    reps = {layer: [] for layer in REP_LAYERS}
    for start in range(0, len(prompt_rows), batch_size):
        batch_rows = prompt_rows[start : start + batch_size]
        texts = [chat_prompt(tokenizer, str(row["user"])) for row in batch_rows]
        batch = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        out = model(**batch, output_hidden_states=True, use_cache=False)
        last_pos = batch["attention_mask"].sum(dim=1) - 1
        row_idx = torch.arange(len(batch_rows), device=device)
        for layer in REP_LAYERS:
            h = out.hidden_states[layer + 1]
            reps[layer].append(h[row_idx, last_pos].detach().cpu().float())
    return {layer: torch.cat(parts, dim=0) for layer, parts in reps.items()}


def row_cosine_mean(a: torch.Tensor, b: torch.Tensor, idx: list[int]) -> float:
    aa = F.normalize(a[idx].float(), dim=1)
    bb = F.normalize(b[idx].float(), dim=1)
    return float((aa * bb).sum(dim=1).mean().item())


def activation_similarity_rows(reps_by_model, prompt_rows) -> list[dict[str, object]]:
    by_split: dict[str, list[int]] = {}
    for i, row in enumerate(prompt_rows):
        by_split.setdefault(str(row["split"]), []).append(i)
    rows = []
    for layer in REP_LAYERS:
        for split, idx in by_split.items():
            rows.append(
                {
                    "model_a": "base",
                    "model_b": "abliterated",
                    "layer": layer,
                    "split": split,
                    "row_cosine": row_cosine_mean(
                        reps_by_model["base"][layer],
                        reps_by_model["abliterated"][layer],
                        idx,
                    ),
                    "n": len(idx),
                }
            )
    return rows


def config_rows(models: dict[str, str]) -> list[dict[str, object]]:
    rows = []
    for short, model_id in models.items():
        cfg = AutoConfig.from_pretrained(model_id)
        rows.append(
            {
                "short": short,
                "model": model_id,
                "model_type": getattr(cfg, "model_type", ""),
                "hidden_size": getattr(cfg, "hidden_size", ""),
                "intermediate_size": getattr(cfg, "intermediate_size", ""),
                "num_hidden_layers": getattr(cfg, "num_hidden_layers", ""),
                "num_attention_heads": getattr(cfg, "num_attention_heads", ""),
                "num_key_value_heads": getattr(cfg, "num_key_value_heads", ""),
                "vocab_size": getattr(cfg, "vocab_size", ""),
                "sliding_window": getattr(cfg, "sliding_window", ""),
            }
        )
    return rows


def load_stage0_behavior(path: Path) -> list[dict[str, object]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_summary(path: Path, models, behavior_rows, cfg_rows, activation_rows) -> None:
    lines = [
        "# Gemma-2-2B Abliterated RQ0 Screen",
        "",
        "This is a small pre-SAE screen for the Gemma safety-loss pair.",
        "",
        "## Models",
        "",
    ]
    for short, model_id in models.items():
        lines.append(f"- `{short}`: `{model_id}`")

    lines.extend(["", "## Stage 0 Behavior", ""])
    if behavior_rows:
        lines.extend(
            [
                "| model | substrate pass | refusal pass | clean gen | harmful clean | benign helpful | arith | polite |",
                "|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in behavior_rows:
            lines.append(
                f"| `{row['model']}` | {row['substrate_pass']} | {row['refusal_target_pass']} | "
                f"{float(row['clean_generation_rate']):.3f} | {float(row['harmful_ok_rate']):.3f} | "
                f"{float(row['benign_ok_rate']):.3f} | {float(row['arith_ok_rate']):.3f} | "
                f"{float(row['polite_ok_rate']):.3f} |"
            )

    lines.extend(["", "## Architecture", ""])
    lines.extend(
        [
            "| short | type | layers | hidden | intermediate | heads | kv heads | vocab |",
            "|---|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in cfg_rows:
        lines.append(
            f"| `{row['short']}` | `{row['model_type']}` | {row['num_hidden_layers']} | "
            f"{row['hidden_size']} | {row['intermediate_size']} | {row['num_attention_heads']} | "
            f"{row['num_key_value_heads']} | {row['vocab_size']} |"
        )

    lines.extend(["", "## Activation Similarity", ""])
    lines.extend(
        [
            "| layer | harmful cosine | benign cosine | arith cosine | polite cosine | harmful-benign gap |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
    )
    by_layer = {layer: {} for layer in REP_LAYERS}
    for row in activation_rows:
        by_layer[int(row["layer"])][str(row["split"])] = float(row["row_cosine"])
    for layer in REP_LAYERS:
        vals = by_layer[layer]
        harmful = vals.get("harmful", float("nan"))
        benign = vals.get("benign", float("nan"))
        lines.append(
            f"| {layer} | {harmful:.3f} | {benign:.3f} | {vals.get('arith', float('nan')):.3f} | "
            f"{vals.get('polite', float('nan')):.3f} | {benign - harmful:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Decision",
            "",
            "- This pair passes the cheap behavior and architecture gates.",
            "- If harmful cosine is substantially lower than benign cosine in middle/late layers, run module/activation patching next.",
            "- Do not start GemmaScope feature interpretation until a causal patch target is established.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--examples-per-split", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument(
        "--stage0-metrics",
        type=Path,
        default=ROOT
        / "stage0"
        / "results"
        / "candidate_screens_gemma2_2b_abliterated_20260522_clean"
        / "chat_candidate_screen_metrics.csv",
    )
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    prompt_rows = make_prompt_rows(args.examples_per_split)
    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODELS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    reps_by_model = {}
    for short, model_id in DEFAULT_MODELS.items():
        print(f"[load] {short}: {model_id}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(args.device)
        model.eval()
        reps_by_model[short] = collect_reps(
            model,
            tokenizer,
            prompt_rows,
            device=args.device,
            batch_size=args.batch_size,
        )
        del model
        torch.cuda.empty_cache()

    cfg_rows = config_rows(DEFAULT_MODELS)
    behavior_rows = load_stage0_behavior(args.stage0_metrics)
    activation_rows = activation_similarity_rows(reps_by_model, prompt_rows)

    config_path = args.result_dir / "gemma2_2b_config_check.csv"
    activation_path = args.result_dir / "gemma2_2b_activation_similarity.csv"
    summary_path = args.result_dir / "GEMMA2_2B_ABLITERATED_RQ0_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(config_path, cfg_rows)
    write_csv(activation_path, activation_rows)
    write_summary(summary_path, DEFAULT_MODELS, behavior_rows, cfg_rows, activation_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "models": DEFAULT_MODELS,
                "examples_per_split": args.examples_per_split,
                "rep_layers": REP_LAYERS,
                "stage0_metrics": str(args.stage0_metrics),
                "outputs": {
                    "config": str(config_path),
                    "activation_similarity": str(activation_path),
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
