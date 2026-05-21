#!/usr/bin/env python3
"""Small RQ0 screen for public Qwen merge candidates.

This checks whether Stage 0 behavioral candidates are distinct enough to be
interesting before spending time on SAE-style analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from itertools import combinations
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
from screen_chat_merge_candidate import ARITH_PROMPTS, BENIGN_PROMPTS, HARMFUL_PROMPTS, POLITE_PROMPTS  # noqa: E402


DEFAULT_MODELS = {
    "qwen2_5_instruct": "Qwen/Qwen2.5-0.5B-Instruct",
    "sjt": "Sakalti/SJT-0.5B",
    "ko_merge": "vitus9988/Qwen2.5-0.5B-ko-merge",
    "amadeus_ptbr": "amadeusai/Amadeus-Verbo-MI-Qwen-2.5-0.5B-PT-BR-Instruct-Experimental",
}
REP_LAYERS = (0, 4, 8, 12, 16, 20, 23)


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


def sampled_delta(
    ref_state: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    *,
    max_values: int,
    seed: int,
) -> torch.Tensor:
    keys = [key for key, value in ref_state.items() if torch.is_floating_point(value)]
    per_key = max(1, max_values // max(len(keys), 1))
    parts = []
    for key_i, key in enumerate(keys):
        delta = (state[key].float() - ref_state[key].float()).reshape(-1)
        if delta.numel() > per_key:
            gen = torch.Generator()
            gen.manual_seed(seed + key_i * 9973)
            idx = torch.randperm(delta.numel(), generator=gen)[:per_key]
            delta = delta[idx]
        parts.append(delta.cpu())
    return torch.cat(parts)


def state_distance(
    a_state: dict[str, torch.Tensor],
    b_state: dict[str, torch.Tensor],
) -> dict[str, float | int]:
    sq = 0.0
    a_sq = 0.0
    max_abs = 0.0
    values = 0
    for key, a in a_state.items():
        if not torch.is_floating_point(a):
            continue
        b = b_state[key]
        diff = (a.float() - b.float()).reshape(-1)
        aa = a.float().reshape(-1)
        sq += float(torch.dot(diff, diff).item())
        a_sq += float(torch.dot(aa, aa).item())
        max_abs = max(max_abs, float(diff.abs().max().item()))
        values += int(diff.numel())
    l2 = sq**0.5
    ref_l2 = a_sq**0.5
    return {
        "l2": l2,
        "relative_to_a": l2 / ref_l2 if ref_l2 else float("nan"),
        "max_abs": max_abs,
        "n_values": values,
    }


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    denom = float(a.norm().item() * b.norm().item())
    if denom == 0.0:
        return float("nan")
    return float(torch.dot(a.float(), b.float()).item() / denom)


def sign_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    active = (a != 0) | (b != 0)
    if int(active.sum().item()) == 0:
        return float("nan")
    return float((torch.sign(a[active]) == torch.sign(b[active])).float().mean().item())


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
    for model_a, model_b in combinations(reps_by_model, 2):
        for layer in REP_LAYERS:
            for split, idx in by_split.items():
                rows.append(
                    {
                        "model_a": model_a,
                        "model_b": model_b,
                        "layer": layer,
                        "split": split,
                        "row_cosine": row_cosine_mean(
                            reps_by_model[model_a][layer],
                            reps_by_model[model_b][layer],
                            idx,
                        ),
                        "n": len(idx),
                    }
                )
    return rows


def load_stage0_behavior(stage0_metrics: Path) -> list[dict[str, object]]:
    if not stage0_metrics.exists():
        return []
    with stage0_metrics.open(newline="", encoding="utf-8") as f:
        return [dict(row) for row in csv.DictReader(f)]


def write_summary(
    path: Path,
    models: dict[str, str],
    behavior_rows: list[dict[str, object]],
    magnitude_rows: list[dict[str, object]],
    delta_rows: list[dict[str, object]],
    activation_rows: list[dict[str, object]],
) -> None:
    lines = [
        "# Qwen Public Candidate RQ0 Screen",
        "",
        "This is a small RQ0-style screen for public Qwen merge candidates that passed or nearly passed Stage 0.",
        "",
        "Models:",
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
            if row["model"] not in models.values():
                continue
            lines.append(
                f"| `{row['model']}` | {row['substrate_pass']} | {row['refusal_target_pass']} | "
                f"{float(row['clean_generation_rate']):.3f} | {float(row['harmful_ok_rate']):.3f} | "
                f"{float(row['benign_ok_rate']):.3f} | {float(row['arith_ok_rate']):.3f} | "
                f"{float(row['polite_ok_rate']):.3f} |"
            )
    else:
        lines.append("Stage 0 behavior file not found.")

    lines.extend(
        [
            "",
            "## Sampled Delta Magnitude",
            "",
            "Deltas are sampled relative to `qwen2_5_instruct` using the same sample as the cosine table.",
            "",
            "| model | sampled L2 | mean abs | max abs | n values |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in magnitude_rows:
        lines.append(
            f"| `{row['model']}` | {row['sample_l2']:.3f} | "
            f"{row['sample_mean_abs']:.6f} | {row['sample_max_abs']:.3f} | {row['n_values']} |"
        )

    lines.extend(
        [
            "",
            "## Delta Geometry",
            "",
            "Deltas are sampled relative to `qwen2_5_instruct`.",
            "",
            "| pair | cosine | sign agreement |",
            "|---|---:|---:|",
        ]
    )
    for row in sorted(delta_rows, key=lambda r: (str(r["model_a"]), str(r["model_b"]))):
        lines.append(
            f"| `{row['model_a']}` vs `{row['model_b']}` | {row['cosine']:.3f} | {row['sign_agreement']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Activation Similarity Highlights",
            "",
            "Mean row-wise cosine at layer 23 on harmful and benign prompts:",
            "",
            "| pair | harmful | benign |",
            "|---|---:|---:|",
        ]
    )
    layer23 = [row for row in activation_rows if int(row["layer"]) == 23]
    pairs = sorted({(str(row["model_a"]), str(row["model_b"])) for row in layer23})
    for model_a, model_b in pairs:
        harmful = next(
            (row for row in layer23 if row["model_a"] == model_a and row["model_b"] == model_b and row["split"] == "harmful"),
            None,
        )
        benign = next(
            (row for row in layer23 if row["model_a"] == model_a and row["model_b"] == model_b and row["split"] == "benign"),
            None,
        )
        if harmful and benign:
            lines.append(f"| `{model_a}` vs `{model_b}` | {harmful['row_cosine']:.3f} | {benign['row_cosine']:.3f} |")

    lines.extend(
        [
            "",
            "## Current Decision",
            "",
            "- Promote only candidates that are behaviorally clean and not just a near-copy of `qwen2_5_instruct`.",
            "- Use failed-but-clean candidates as contrast cases only when they expose a clear mechanism, such as refusal loss.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_model_arg(raw_models: str | None) -> dict[str, str]:
    if not raw_models:
        return dict(DEFAULT_MODELS)
    models = {}
    for item in raw_models.split(","):
        short, model_id = item.split("=", 1)
        models[short.strip()] = model_id.strip()
    return models


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--examples-per-split", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--max-delta-values", type=int, default=120_000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--models", default=None, help="Comma-separated short=model_id entries.")
    ap.add_argument(
        "--stage0-metrics",
        type=Path,
        default=ROOT / "stage0" / "results" / "candidate_screens" / "qwen_merge_candidates_round3" / "chat_candidate_screen_metrics.csv",
    )
    ap.add_argument(
        "--result-dir",
        type=Path,
        default=ROOT / "stage1" / "results" / "qwen_public_candidates_rq0",
    )
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    models = parse_model_arg(args.models)
    if "qwen2_5_instruct" not in models:
        raise ValueError("models must include qwen2_5_instruct=Qwen/Qwen2.5-0.5B-Instruct")

    prompt_rows = make_prompt_rows(args.examples_per_split)
    tokenizer = AutoTokenizer.from_pretrained(models["qwen2_5_instruct"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[load] reference state qwen2_5_instruct", flush=True)
    ref_model = AutoModelForCausalLM.from_pretrained(models["qwen2_5_instruct"], torch_dtype=torch.float16)
    ref_state = {k: v.detach().cpu().half() for k, v in ref_model.state_dict().items()}
    del ref_model

    magnitude_rows = []
    delta_vectors = {}
    reps_by_model = {}

    for short, model_id in models.items():
        print(f"[load/eval] {short}: {model_id}", flush=True)
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=torch.float16).to(args.device)
        model.eval()
        state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
        if short != "qwen2_5_instruct":
            delta = sampled_delta(ref_state, state, max_values=args.max_delta_values, seed=args.seed)
            delta_vectors[short] = delta
            magnitude_rows.append(
                {
                    "model": short,
                    "sample_l2": float(delta.float().norm().item()),
                    "sample_mean_abs": float(delta.float().abs().mean().item()),
                    "sample_max_abs": float(delta.float().abs().max().item()),
                    "n_values": int(delta.numel()),
                }
            )
        reps_by_model[short] = collect_reps(model, tokenizer, prompt_rows, device=args.device, batch_size=args.batch_size)
        del model, state
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    delta_rows = []
    for model_a, model_b in combinations(delta_vectors, 2):
        a = delta_vectors[model_a]
        b = delta_vectors[model_b]
        delta_rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "cosine": cosine(a, b),
                "sign_agreement": sign_agreement(a, b),
                "n_values": int(min(a.numel(), b.numel())),
            }
        )

    activation_rows = activation_similarity_rows(reps_by_model, prompt_rows)
    behavior_rows = load_stage0_behavior(args.stage0_metrics)

    prompt_path = args.result_dir / "qwen_public_candidate_prompts.csv"
    magnitude_path = args.result_dir / "qwen_public_candidate_delta_magnitude.csv"
    delta_path = args.result_dir / "qwen_public_candidate_delta_geometry.csv"
    activation_path = args.result_dir / "qwen_public_candidate_activation_similarity.csv"
    behavior_path = args.result_dir / "qwen_public_candidate_stage0_behavior.csv"
    summary_path = args.result_dir / "QWEN_PUBLIC_CANDIDATE_RQ0_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"

    write_csv(prompt_path, prompt_rows)
    write_csv(magnitude_path, magnitude_rows)
    write_csv(delta_path, delta_rows)
    write_csv(activation_path, activation_rows)
    write_csv(behavior_path, behavior_rows)
    write_summary(summary_path, models, behavior_rows, magnitude_rows, delta_rows, activation_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "models": models,
                "rep_layers": REP_LAYERS,
                "examples_per_split": args.examples_per_split,
                "stage0_metrics": str(args.stage0_metrics),
                "outputs": {
                    "prompts": str(prompt_path),
                    "delta_magnitude": str(magnitude_path),
                    "delta_geometry": str(delta_path),
                    "activation_similarity": str(activation_path),
                    "behavior": str(behavior_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print(f"[save] {summary_path}")
    for row in magnitude_rows:
        print(
            f"  {row['model']:<18} sample_l2={row['sample_l2']:.3f} "
            f"mean_abs={row['sample_mean_abs']:.6f} max={row['sample_max_abs']:.3f}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
