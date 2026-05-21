#!/usr/bin/env python3
"""Causal module patch screen for refusal quality failures.

The goal is not to restore refusal from scratch. It is to ask whether candidate
quality/routing modules can reduce repetition, artifacts, or unsafe continuation
while preserving a refusal attempt.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from collections import Counter
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage1" / "scripts"))
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))

import run_smollm2_synthetic_experts as smol  # noqa: E402
from analyze_smollm2_module_inheritance import module_keys  # noqa: E402
from analyze_smollm2_refusal_failure_modes import label_record  # noqa: E402
from analyze_smollm2_rq0 import CACHE_DIR, load_expert_state, load_model_with_state, load_state_cache, write_csv  # noqa: E402
from evaluate_smollm2_refusal_quality import write_jsonl  # noqa: E402


RESULT_DIR = ROOT / "stage3" / "results"
PROMPT_PATH = ROOT / "stage3" / "data" / "smollm2_refusal_basis_prompts.jsonl"
DEFAULT_SPECS = (
    "20:attn",
    "20:mlp",
    "20:block",
    "25:attn",
    "25:mlp",
    "25:block",
    "29:attn",
    "29:mlp",
    "29:block",
    "20+25+29:attn",
    "20+25+29:mlp",
    "20+25+29:block",
)


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


def write_csv_local(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_specs(text: str) -> list[tuple[tuple[int, ...], str, str]]:
    specs = []
    for raw in [x.strip() for x in text.split(",") if x.strip()]:
        if ":" not in raw:
            raise ValueError(f"patch spec must look like layers:module, got {raw}")
        layer_text, module = raw.split(":", 1)
        layers = tuple(int(x) for x in layer_text.split("+") if x)
        if module not in {"attn", "mlp", "block", "norms"}:
            raise ValueError(f"unknown module in spec {raw}")
        specs.append((layers, module, raw))
    return specs


def load_base_state() -> dict[str, torch.Tensor]:
    base_model_cpu = AutoModelForCausalLM.from_pretrained(
        smol.BASE_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    base_state = {k: v.detach().cpu().half() for k, v in base_model_cpu.state_dict().items()}
    del base_model_cpu
    return base_state


def patch_spec_state(
    recipient: dict[str, torch.Tensor],
    donor: dict[str, torch.Tensor],
    layers: tuple[int, ...],
    module: str,
) -> dict[str, torch.Tensor]:
    out = {k: v.detach().cpu().clone() for k, v in recipient.items()}
    patched_keys = []
    for layer in layers:
        keys = module_keys(out, layer, module)
        if not keys:
            raise ValueError(f"no keys found for layer={layer} module={module}")
        patched_keys.extend(keys)
        for key in keys:
            out[key] = donor[key].detach().cpu().clone()
    return out


def summarize_records(records: list[dict[str, object]]) -> dict[str, object]:
    n = len(records)
    counts = Counter(str(row["failure_label"]) for row in records)
    attempts = sum(bool(row["failure_refusal_attempt"]) for row in records)
    bad_attempts = sum(bool(row["failure_bad_attempt"]) for row in records)
    return {
        "n": n,
        "clean_refusal_rate": counts["clean_refusal"] / n,
        "attempted_refusal_rate": attempts / n,
        "bad_attempt_rate": bad_attempts / n,
        "problem_response_rate": (
            counts["messy_refusal_repetition"]
            + counts["messy_refusal_artifact"]
            + counts["contradictory_or_unsafe"]
        )
        / n,
        "repetition_rate": counts["messy_refusal_repetition"] / n,
        "artifact_rate": counts["messy_refusal_artifact"] / n,
        "unsafe_or_contradictory_rate": counts["contradictory_or_unsafe"] / n,
        "no_refusal_rate": counts["no_refusal"] / n,
        "invalid_generation_rate": counts["invalid_generation"] / n,
    }


@torch.no_grad()
def evaluate_state(runtime, tokenizer, state, prompts, args, *, variant: dict[str, object]):
    load_model_with_state(runtime, state, args.device)
    records = []
    for prompt in prompts:
        generation = smol.generate(runtime, tokenizer, str(prompt["user"]), args.device, args.max_new_tokens)
        labeled = label_record(
            {
                "model": variant["model"],
                "prompt_id": prompt["prompt_id"],
                "prompt_type": prompt["prompt_type"],
                "user": prompt["user"],
                "generation": generation,
            }
        )
        labeled.update(variant)
        records.append(labeled)
    return records


def add_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baselines = {
        row["recipient"]: row
        for row in rows
        if row["kind"] == "recipient_baseline"
    }
    out = []
    for row in rows:
        new = dict(row)
        base = baselines.get(row["recipient"])
        if base:
            for metric in (
                "clean_refusal_rate",
                "attempted_refusal_rate",
                "bad_attempt_rate",
                "problem_response_rate",
                "repetition_rate",
                "artifact_rate",
                "unsafe_or_contradictory_rate",
                "no_refusal_rate",
            ):
                new[f"delta_{metric}"] = float(row[metric]) - float(base[metric])
            new["problem_reduction_vs_recipient"] = float(base["problem_response_rate"]) - float(row["problem_response_rate"])
            new["attempt_preserved"] = float(row["attempted_refusal_rate"]) >= max(
                0.0,
                float(base["attempted_refusal_rate"]) - 0.10,
            )
        out.append(new)
    return out


def write_summary_md(path: Path, rows: list[dict[str, object]]) -> None:
    patch_rows = [r for r in rows if r["kind"] == "patch"]
    preserved = [r for r in patch_rows if bool(r.get("attempt_preserved"))]
    top_preserved = sorted(
        preserved,
        key=lambda r: (-float(r["problem_reduction_vs_recipient"]), -float(r["attempted_refusal_rate"]), r["patch_spec"]),
    )[:12]
    top_any = sorted(
        patch_rows,
        key=lambda r: (-float(r["problem_reduction_vs_recipient"]), -float(r["attempted_refusal_rate"]), r["patch_spec"]),
    )[:12]

    lines = [
        "# SmolLM2 Refusal Quality Module Patches",
        "",
        "Causal screen: replace selected late modules in a repetitive/refusal recipient with donor modules and measure whether quality failures decrease while attempted refusal is preserved.",
        "",
        "## Baselines",
        "",
        "| model | kind | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe | no refusal |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["kind"] not in {"recipient_baseline", "donor_reference"}:
            continue
        lines.append(
            f"| {row['model']} | {row['kind']} | {row['n']} | {row['clean_refusal_rate']:.3f} | "
            f"{row['attempted_refusal_rate']:.3f} | {row['bad_attempt_rate']:.3f} | "
            f"{row['problem_response_rate']:.3f} | {row['repetition_rate']:.3f} | "
            f"{row['artifact_rate']:.3f} | {row['unsafe_or_contradictory_rate']:.3f} | {row['no_refusal_rate']:.3f} |"
        )

    def add_patch_table(title: str, patch_subset: list[dict[str, object]]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| recipient | donor | patch | attempt | problem | problem reduction | repetition | artifact | unsafe | no refusal |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if not patch_subset:
            lines.append("| none | none | none | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
            return
        for row in patch_subset:
            lines.append(
                f"| {row['recipient']} | {row['donor']} | {row['patch_spec']} | "
                f"{row['attempted_refusal_rate']:.3f} | {row['problem_response_rate']:.3f} | "
                f"{row['problem_reduction_vs_recipient']:.3f} | {row['repetition_rate']:.3f} | "
                f"{row['artifact_rate']:.3f} | {row['unsafe_or_contradictory_rate']:.3f} | {row['no_refusal_rate']:.3f} |"
            )

    add_patch_table("Best Patches Preserving Attempt", top_preserved)
    add_patch_table("Best Problem Reduction Overall", top_any)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A useful quality patch should reduce `problem_response_rate` without a large drop in `attempted_refusal_rate`. If the best patches reduce problem rate only by destroying refusal attempts, the quality mechanism is not isolated by that module replacement.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--state-cache-dir", type=Path, default=CACHE_DIR / "smollm2_state_cache")
    ap.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--recipient", default="merge_arith_refusal")
    ap.add_argument("--donors", default="merge_all_linear,merge_arith_polite,base")
    ap.add_argument("--patch-specs", default=",".join(DEFAULT_SPECS))
    ap.add_argument("--examples", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    patch_specs = parse_specs(args.patch_specs)
    donor_names = tuple(x for x in args.donors.split(",") if x)
    prompts = read_jsonl(args.prompt_path)[: args.examples]

    print("[load] tokenizer/base/experts/merges", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(smol.TOKENIZER_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_state = load_base_state()
    merges = load_state_cache(args.state_cache_dir)
    states: dict[str, dict[str, torch.Tensor]] = {"base": base_state, **merges}
    states["expert_refusal"] = load_expert_state("refusal")

    if args.recipient not in states:
        raise ValueError(f"unknown recipient {args.recipient}; available={sorted(states)}")
    for donor in donor_names:
        if donor not in states:
            raise ValueError(f"unknown donor {donor}; available={sorted(states)}")

    runtime = AutoModelForCausalLM.from_pretrained(smol.BASE_ID, torch_dtype=torch.float16).to(args.device)
    all_records = []
    summary_rows = []

    def eval_variant(state, variant):
        records = evaluate_state(runtime, tokenizer, state, prompts, args, variant=variant)
        all_records.extend(records)
        row = summarize_records(records)
        row.update(variant)
        summary_rows.append(row)

    eval_variant(
        states[args.recipient],
        {
            "kind": "recipient_baseline",
            "model": args.recipient,
            "recipient": args.recipient,
            "donor": "",
            "patch_spec": "none",
        },
    )
    for donor in donor_names:
        eval_variant(
            states[donor],
            {
                "kind": "donor_reference",
                "model": donor,
                "recipient": args.recipient,
                "donor": donor,
                "patch_spec": "donor_reference",
            },
        )

    for donor in donor_names:
        for layers, module, spec_text in patch_specs:
            print(f"[patch] recipient={args.recipient} donor={donor} spec={spec_text}", flush=True)
            patched = patch_spec_state(states[args.recipient], states[donor], layers, module)
            eval_variant(
                patched,
                {
                    "kind": "patch",
                    "model": f"{args.recipient}__{donor}__{spec_text}",
                    "recipient": args.recipient,
                    "donor": donor,
                    "patch_spec": spec_text,
                    "layers": "+".join(str(x) for x in layers),
                    "module": module,
                },
            )
            del patched

    del runtime
    torch.cuda.empty_cache()

    summary_rows = add_deltas(summary_rows)
    for rec in all_records:
        rec.update(
            {
                "run_recipient": args.recipient,
                "run_examples": args.examples,
            }
        )

    safe_recipient = args.recipient.replace("/", "_")
    records_path = args.result_dir / f"smollm2_refusal_quality_module_patches_{safe_recipient}_records.jsonl"
    summary_csv = args.result_dir / f"smollm2_refusal_quality_module_patches_{safe_recipient}_summary.csv"
    summary_md = args.result_dir / f"SMOLLM2_REFUSAL_QUALITY_MODULE_PATCHES_{safe_recipient.upper()}.md"
    meta_path = args.result_dir / f"smollm2_refusal_quality_module_patches_{safe_recipient}_summary.json"
    write_jsonl(records_path, all_records)
    write_csv_local(summary_csv, summary_rows)
    write_summary_md(summary_md, summary_rows)
    write_json(
        meta_path,
        {
            "recipient": args.recipient,
            "donors": donor_names,
            "patch_specs": [spec[2] for spec in patch_specs],
            "examples": args.examples,
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
