#!/usr/bin/env python3
"""Activation-direction steering test for Stage 3 refusal attempts."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage1" / "scripts"))
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))

import run_smollm2_synthetic_experts as smol  # noqa: E402
from analyze_smollm2_refusal_basis import CACHE_STAGE3, RESULT_DIR, score_stage3_refusal_quality  # noqa: E402
from analyze_smollm2_rq0 import CACHE_DIR, load_model_with_state, load_state_cache, write_csv  # noqa: E402
from analyze_smollm2_refusal_failure_modes import label_record, target_value  # noqa: E402
from apply_smollm2_refusal_assistant_audit import assistant_label, boolish, has_refusal_attempt  # noqa: E402
from evaluate_smollm2_refusal_quality import write_jsonl  # noqa: E402


PROMPT_PATH = ROOT / "stage3" / "data" / "smollm2_refusal_basis_prompts.jsonl"


def read_jsonl(path: Path) -> list[dict[str, object]]:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def write_json(path: Path, obj: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=True), encoding="utf-8")


def parse_alphas(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x)


def module_by_name(model, dotted_name: str):
    out = model
    for part in dotted_name.split("."):
        out = getattr(out, part)
    return out


def build_direction(cache: dict[str, object], rep_name: str, target: str) -> torch.Tensor:
    labeled = [label_record(row) for row in cache["records"]]
    y = torch.tensor([target_value(row, target) for row in labeled], dtype=torch.bool)
    x = cache["features"][rep_name].float()
    if int(y.sum().item()) == 0 or int((~y).sum().item()) == 0:
        raise ValueError(f"target {target} has one class only")
    direction = x[y].mean(dim=0) - x[~y].mean(dim=0)
    return direction


def steering_hook(direction: torch.Tensor, alpha: float, position: str):
    def hook(_module, _inputs, output):
        if isinstance(output, tuple):
            tensor = output[0]
            rest = output[1:]
        else:
            tensor = output
            rest = None
        delta = (alpha * direction).to(device=tensor.device, dtype=tensor.dtype)
        patched = tensor.clone()
        if position == "last":
            patched[:, -1, :] = patched[:, -1, :] + delta
        elif position == "all":
            patched = patched + delta.view(1, 1, -1)
        else:
            raise ValueError(f"unknown position {position}")
        if rest is None:
            return patched
        return (patched, *rest)

    return hook


def label_generation(text: str) -> dict[str, object]:
    row = {"generation": text}
    row.update(score_stage3_refusal_quality(text))
    label, note = assistant_label(row)
    row["failure_label"] = label
    row["failure_notes"] = note
    row["failure_refusal_attempt"] = has_refusal_attempt(row)
    return row


@torch.no_grad()
def evaluate_with_direction(model, tokenizer, prompts, direction: torch.Tensor, args, *, alpha: float):
    records = []
    handle = None
    if alpha != 0.0:
        module = module_by_name(model, f"model.layers.{args.layer}.mlp")
        handle = module.register_forward_hook(steering_hook(direction, alpha, args.position))
    try:
        for row in prompts:
            generation = smol.generate(model, tokenizer, str(row["user"]), args.device, args.max_new_tokens)
            labels = label_generation(generation)
            records.append(
                {
                    "prompt_id": row["prompt_id"],
                    "prompt_type": row["prompt_type"],
                    "user": row["user"],
                    "generation": generation,
                    **labels,
                }
            )
    finally:
        if handle is not None:
            handle.remove()
    return records


def summarize(records: list[dict[str, object]]) -> dict[str, object]:
    n = len(records)
    counts = Counter(str(row["failure_label"]) for row in records)
    return {
        "n": n,
        "clean_refusal_rate": counts["clean_refusal"] / n,
        "attempted_refusal_rate": sum(boolish(row["failure_refusal_attempt"]) for row in records) / n,
        "bad_attempt_rate": sum(boolish(row["failure_refusal_attempt"]) and row["failure_label"] != "clean_refusal" for row in records)
        / n,
        "problem_response_rate": (
            counts["messy_refusal_repetition"] + counts["messy_refusal_artifact"] + counts["contradictory_or_unsafe"]
        )
        / n,
        "repetition_rate": counts["messy_refusal_repetition"] / n,
        "artifact_rate": counts["messy_refusal_artifact"] / n,
        "unsafe_or_contradictory_rate": counts["contradictory_or_unsafe"] / n,
        "no_refusal_rate": counts["no_refusal"] / n,
    }


def write_summary_md(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# SmolLM2 Refusal Direction Steering",
        "",
        "Exploratory causal check: add the refusal-attempt centroid direction to layer-20 MLP outputs during generation.",
        "",
        "| model | alpha | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe/contradictory | no refusal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['alpha']} | {row['n']} | {row['clean_refusal_rate']:.3f} | "
            f"{row['attempted_refusal_rate']:.3f} | {row['bad_attempt_rate']:.3f} | "
            f"{row['problem_response_rate']:.3f} | {row['repetition_rate']:.3f} | "
            f"{row['artifact_rate']:.3f} | {row['unsafe_or_contradictory_rate']:.3f} | "
            f"{row['no_refusal_rate']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "This is a causal sanity check for a raw activation direction, not an SAE result. A useful direction should increase attempted refusal without simply increasing artifacts, repetition, or unsafe continuations.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", type=Path, default=CACHE_STAGE3 / "smollm2_refusal_basis_activations.pt")
    ap.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    ap.add_argument("--state-cache-dir", type=Path, default=CACHE_DIR / "smollm2_state_cache")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--model-names", default="merge_arith_polite")
    ap.add_argument("--examples", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    ap.add_argument("--rep-name", default="mlp_out_l20")
    ap.add_argument("--target", default="failure_attempted_refusal")
    ap.add_argument("--layer", type=int, default=20)
    ap.add_argument("--alphas", default="0,0.25,0.5,1,2")
    ap.add_argument("--position", choices=("last", "all"), default="last")
    args = ap.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(smol.TOKENIZER_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    direction = build_direction(cache, args.rep_name, args.target)
    prompts = read_jsonl(args.prompt_path)[: args.examples]
    merges = load_state_cache(args.state_cache_dir)
    model_names = tuple(x for x in args.model_names.split(",") if x)
    alphas = parse_alphas(args.alphas)

    runtime = AutoModelForCausalLM.from_pretrained(smol.BASE_ID, torch_dtype=torch.float16).to(args.device)
    all_records = []
    summary_rows = []
    for model_name in model_names:
        if model_name not in merges:
            raise ValueError(f"model {model_name} not found in cached merges: {sorted(merges)}")
        print(f"[model] {model_name}", flush=True)
        load_model_with_state(runtime, merges[model_name], args.device)
        for alpha in alphas:
            print(f"  [alpha] {alpha}", flush=True)
            records = evaluate_with_direction(runtime, tokenizer, prompts, direction, args, alpha=alpha)
            for rec in records:
                rec["model"] = model_name
                rec["alpha"] = alpha
                rec["rep_name"] = args.rep_name
                rec["target"] = args.target
                rec["layer"] = args.layer
                rec["position"] = args.position
            all_records.extend(records)
            row = summarize(records)
            row.update({"model": model_name, "alpha": alpha, "rep_name": args.rep_name, "target": args.target})
            summary_rows.append(row)

    del runtime
    torch.cuda.empty_cache()

    records_path = args.result_dir / "smollm2_refusal_direction_steering_records.jsonl"
    summary_csv = args.result_dir / "smollm2_refusal_direction_steering_summary.csv"
    summary_md = args.result_dir / "SMOLLM2_REFUSAL_DIRECTION_STEERING_SUMMARY.md"
    meta_path = args.result_dir / "smollm2_refusal_direction_steering_summary.json"
    write_jsonl(records_path, all_records)
    write_csv(summary_csv, summary_rows)
    write_summary_md(summary_md, summary_rows)
    write_json(
        meta_path,
        {
            "rep_name": args.rep_name,
            "target": args.target,
            "layer": args.layer,
            "position": args.position,
            "direction_norm": float(direction.norm().item()),
            "examples": args.examples,
            "alphas": alphas,
            "models": model_names,
            "outputs": {
                "records": str(records_path),
                "summary_csv": str(summary_csv),
                "summary_md": str(summary_md),
            },
        },
    )
    print(f"[save] {records_path}")
    print(f"[save] {summary_csv}")
    print(f"[save] {summary_md}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

