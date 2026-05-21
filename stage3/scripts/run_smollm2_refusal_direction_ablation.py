#!/usr/bin/env python3
"""Direction ablation for refusal failure modes.

This is a finer-grained follow-up to module and activation replacement. Instead
of replacing a whole module activation with a donor activation, it computes a
failure-mode direction from the Stage 3 activation cache and removes that
direction during generation.
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
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage1" / "scripts"))
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))

import run_smollm2_synthetic_experts as smol  # noqa: E402
from analyze_smollm2_refusal_basis import CACHE_STAGE3, RESULT_DIR  # noqa: E402
from analyze_smollm2_refusal_failure_modes import label_record, target_value  # noqa: E402
from analyze_smollm2_rq0 import CACHE_DIR, load_model_with_state, load_state_cache, write_csv  # noqa: E402
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


def write_csv_local(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def parse_alphas(text: str) -> tuple[float, ...]:
    return tuple(float(x) for x in text.split(",") if x)


def parse_direction_specs(text: str) -> list[tuple[str, str]]:
    specs = []
    for raw in [x.strip() for x in text.split(",") if x.strip()]:
        if ":" not in raw:
            raise ValueError(f"direction spec must look like representation:target, got {raw}")
        rep, target = raw.split(":", 1)
        specs.append((rep, target))
    return specs


def rep_to_layer_kind(rep_name: str) -> tuple[int, str]:
    prefix, layer_text = rep_name.rsplit("_l", 1)
    layer = int(layer_text)
    if prefix == "mlp_out":
        return layer, "mlp"
    if prefix == "resid":
        return layer, "block"
    raise ValueError(f"unsupported representation for dynamic intervention: {rep_name}")


def module_for(model, layer: int, kind: str):
    block = model.model.layers[layer]
    if kind == "block":
        return block
    if kind == "mlp":
        return block.mlp
    raise ValueError(kind)


def first_tensor(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def replace_first_tensor(output, tensor):
    _original, rest = first_tensor(output)
    if rest is None:
        return tensor
    return (tensor, *rest)


def build_direction(cache: dict[str, object], rep_name: str, target: str) -> torch.Tensor:
    labeled = [label_record(row) for row in cache["records"]]
    y = torch.tensor([target_value(row, target) for row in labeled], dtype=torch.bool)
    x = cache["features"][rep_name].float()
    if int(y.sum().item()) == 0 or int((~y).sum().item()) == 0:
        raise ValueError(f"target {target} has one class only")
    direction = x[y].mean(dim=0) - x[~y].mean(dim=0)
    return direction


def make_direction_hook(direction: torch.Tensor, alpha: float, position: str, mode: str):
    unit = F.normalize(direction.float(), dim=0)

    def patch_slice(values: torch.Tensor) -> torch.Tensor:
        coeff = values.float() @ unit.to(values.device)
        if mode == "positive_project":
            coeff = coeff.clamp_min(0.0)
        elif mode == "project":
            pass
        elif mode == "constant":
            coeff = torch.ones_like(coeff) * direction.norm().to(values.device)
        else:
            raise ValueError(f"unknown mode {mode}")
        delta = coeff.unsqueeze(-1).to(values.dtype) * unit.to(device=values.device, dtype=values.dtype)
        return values - alpha * delta

    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        patched = tensor.clone()
        if position == "last":
            patched[:, -1, :] = patch_slice(patched[:, -1, :])
        elif position == "all":
            flat = patched.reshape(-1, patched.shape[-1])
            patched = patch_slice(flat).reshape_as(patched)
        elif position == "prompt":
            if patched.shape[1] > 1:
                flat = patched[:, :-1, :].reshape(-1, patched.shape[-1])
                patched[:, :-1, :] = patch_slice(flat).reshape_as(patched[:, :-1, :])
        else:
            raise ValueError(f"unknown position {position}")
        return replace_first_tensor(output, patched)

    return hook


@torch.no_grad()
def greedy_generate(model, tokenizer, user: str, device: str, max_new_tokens: int) -> str:
    prompt = smol.chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    for _ in range(max_new_tokens):
        out = model(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
        next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        input_ids = torch.cat([input_ids, next_id], dim=1)
        attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)
        if int(next_id.item()) == tokenizer.eos_token_id:
            break
    text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[1]
    text = text.replace("<|im_end|>", "").strip()
    return text


@torch.no_grad()
def hf_generate(model, tokenizer, user: str, device: str, max_new_tokens: int, *, use_cache: bool) -> str:
    prompt = smol.chat_prompt(tokenizer, user)
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
        use_cache=use_cache,
    )
    text = tokenizer.decode(out[0], skip_special_tokens=False)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[1]
    text = text.replace("<|im_end|>", "").strip()
    return text


@torch.no_grad()
def evaluate_variant(model, tokenizer, prompts, args, variant: dict[str, object], *, hook=None):
    handle = None
    if hook is not None:
        layer, kind = rep_to_layer_kind(str(variant["rep_name"]))
        handle = module_for(model, layer, kind).register_forward_hook(hook)
    try:
        records = []
        for prompt in prompts:
            if args.generation_mode == "custom":
                generation = greedy_generate(model, tokenizer, str(prompt["user"]), args.device, args.max_new_tokens)
            elif args.generation_mode == "hf":
                generation = hf_generate(
                    model,
                    tokenizer,
                    str(prompt["user"]),
                    args.device,
                    args.max_new_tokens,
                    use_cache=not args.hf_no_cache,
                )
            else:
                raise ValueError(f"unknown generation mode {args.generation_mode}")
            row = label_record(
                {
                    "model": variant["model"],
                    "prompt_id": prompt["prompt_id"],
                    "prompt_type": prompt["prompt_type"],
                    "user": prompt["user"],
                    "generation": generation,
                }
            )
            row.update(variant)
            records.append(row)
        return records
    finally:
        if handle is not None:
            handle.remove()


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


def add_deltas(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    baseline = next(row for row in rows if row["kind"] == "recipient_baseline")
    out = []
    for row in rows:
        new = dict(row)
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
            new[f"delta_{metric}"] = float(row[metric]) - float(baseline[metric])
        new["problem_reduction_vs_recipient"] = float(baseline["problem_response_rate"]) - float(row["problem_response_rate"])
        new["attempt_preserved"] = float(row["attempted_refusal_rate"]) >= max(0.0, float(baseline["attempted_refusal_rate"]) - 0.10)
        out.append(new)
    return out


def write_summary_md(path: Path, rows: list[dict[str, object]]) -> None:
    interventions = [row for row in rows if row["kind"] == "direction_ablation"]
    top_preserved = sorted(
        [row for row in interventions if bool(row["attempt_preserved"])],
        key=lambda row: (-float(row["problem_reduction_vs_recipient"]), row["rep_name"], row["target"], float(row["alpha"])),
    )[:12]
    top_any = sorted(
        interventions,
        key=lambda row: (-float(row["problem_reduction_vs_recipient"]), -float(row["attempted_refusal_rate"]), row["rep_name"]),
    )[:12]

    lines = [
        "# SmolLM2 Refusal Direction Ablation",
        "",
        "Failure-mode directions are computed from the Stage 3 prompt-final activation cache, then projected out during dynamic greedy generation.",
        "",
        "## Baseline",
        "",
        "| model | n | clean | attempt | bad attempt | problem | repetition | artifact | unsafe | no refusal |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        if row["kind"] != "recipient_baseline":
            continue
        lines.append(
            f"| {row['model']} | {row['n']} | {row['clean_refusal_rate']:.3f} | "
            f"{row['attempted_refusal_rate']:.3f} | {row['bad_attempt_rate']:.3f} | "
            f"{row['problem_response_rate']:.3f} | {row['repetition_rate']:.3f} | "
            f"{row['artifact_rate']:.3f} | {row['unsafe_or_contradictory_rate']:.3f} | {row['no_refusal_rate']:.3f} |"
        )

    def table(title: str, subset: list[dict[str, object]]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| rep | target | alpha | position | mode | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |",
                "|---|---|---:|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if not subset:
            lines.append("| none | none | 0 | none | none | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
            return
        for row in subset:
            lines.append(
                f"| {row['rep_name']} | {row['target']} | {row['alpha']} | {row['position']} | {row['mode']} | "
                f"{row['attempted_refusal_rate']:.3f} | {row['problem_response_rate']:.3f} | "
                f"{row['problem_reduction_vs_recipient']:.3f} | {row['repetition_rate']:.3f} | "
                f"{row['artifact_rate']:.3f} | {row['unsafe_or_contradictory_rate']:.3f} | {row['no_refusal_rate']:.3f} |"
            )

    table("Best Interventions Preserving Attempt", top_preserved)
    table("Best Problem Reduction Overall", top_any)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A successful subspace intervention should reduce problem responses while preserving attempted refusal. If the best interventions either do nothing or delete the refusal attempt, the failure direction is predictive but not a clean causal control.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--cache", type=Path, default=CACHE_STAGE3 / "smollm2_refusal_basis_activations.pt")
    ap.add_argument("--state-cache-dir", type=Path, default=CACHE_DIR / "smollm2_state_cache")
    ap.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--recipient", default="merge_arith_refusal")
    ap.add_argument("--direction-specs", default="mlp_out_l20:failure_repetition,mlp_out_l20:failure_bad_attempt")
    ap.add_argument("--alphas", default="0.25,0.5,1,2")
    ap.add_argument("--position", choices=("last", "prompt", "all"), default="last")
    ap.add_argument("--mode", choices=("positive_project", "project", "constant"), default="positive_project")
    ap.add_argument("--generation-mode", choices=("custom", "hf"), default="custom")
    ap.add_argument("--hf-no-cache", action="store_true")
    ap.add_argument("--examples", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    prompts = read_jsonl(args.prompt_path)[: args.examples]
    direction_specs = parse_direction_specs(args.direction_specs)
    alphas = parse_alphas(args.alphas)

    print("[load] tokenizer/cache/states/model", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(smol.TOKENIZER_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    cache = torch.load(args.cache, map_location="cpu", weights_only=False)
    directions = {(rep, target): build_direction(cache, rep, target) for rep, target in direction_specs}
    merges = load_state_cache(args.state_cache_dir)
    if args.recipient not in merges:
        raise ValueError(f"recipient {args.recipient} not in cached merges: {sorted(merges)}")
    model = AutoModelForCausalLM.from_pretrained(smol.BASE_ID, torch_dtype=torch.float16).to(args.device)
    load_model_with_state(model, merges[args.recipient], args.device)

    all_records = []
    summary_rows = []
    baseline_variant = {
        "kind": "recipient_baseline",
        "model": args.recipient,
        "recipient": args.recipient,
        "rep_name": "",
        "target": "",
        "alpha": 0.0,
                "position": args.position,
                "mode": args.mode,
                "generation_mode": args.generation_mode,
            }
    baseline_records = evaluate_variant(model, tokenizer, prompts, args, baseline_variant)
    all_records.extend(baseline_records)
    row = summarize_records(baseline_records)
    row.update(baseline_variant)
    summary_rows.append(row)

    for rep_name, target in direction_specs:
        direction = directions[(rep_name, target)]
        layer, kind = rep_to_layer_kind(rep_name)
        for alpha in alphas:
            print(f"[ablate] rep={rep_name} target={target} alpha={alpha} position={args.position}", flush=True)
            variant = {
                "kind": "direction_ablation",
                "model": f"{args.recipient}__ablate_{rep_name}_{target}_a{alpha:g}_{args.position}",
                "recipient": args.recipient,
                "rep_name": rep_name,
                "target": target,
                "alpha": alpha,
                "position": args.position,
                "mode": args.mode,
                "generation_mode": args.generation_mode,
                "layer": layer,
                "activation_kind": kind,
                "direction_norm": float(direction.norm().item()),
            }
            hook = make_direction_hook(direction, alpha, args.position, args.mode)
            records = evaluate_variant(model, tokenizer, prompts, args, variant, hook=hook)
            all_records.extend(records)
            row = summarize_records(records)
            row.update(variant)
            summary_rows.append(row)

    del model
    torch.cuda.empty_cache()
    summary_rows = add_deltas(summary_rows)

    safe_recipient = args.recipient.replace("/", "_")
    mode_label = f"{args.generation_mode}{'_nocache' if args.hf_no_cache else ''}"
    records_path = args.result_dir / f"smollm2_refusal_direction_ablation_{safe_recipient}_{args.position}_{args.mode}_{mode_label}_records.jsonl"
    summary_csv = args.result_dir / f"smollm2_refusal_direction_ablation_{safe_recipient}_{args.position}_{args.mode}_{mode_label}_summary.csv"
    summary_md = args.result_dir / f"SMOLLM2_REFUSAL_DIRECTION_ABLATION_{safe_recipient.upper()}_{args.position.upper()}_{args.mode.upper()}_{mode_label.upper()}.md"
    meta_path = args.result_dir / f"smollm2_refusal_direction_ablation_{safe_recipient}_{args.position}_{args.mode}_{mode_label}_summary.json"
    write_jsonl(records_path, all_records)
    write_csv_local(summary_csv, summary_rows)
    write_summary_md(summary_md, summary_rows)
    write_json(
        meta_path,
        {
            "recipient": args.recipient,
            "direction_specs": direction_specs,
            "alphas": alphas,
            "position": args.position,
            "mode": args.mode,
            "generation_mode": args.generation_mode,
            "hf_no_cache": args.hf_no_cache,
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
