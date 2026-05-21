#!/usr/bin/env python3
"""Dynamic activation patching for SmolLM2 refusal-quality failures.

At each greedy decoding step, the donor model is run on the current prefix, and
selected donor activations are patched into the recipient model before choosing
the next token. This is slower than normal generation, but it tests whether
activation-level quality repair is possible without replacing weights.
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
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))

import run_smollm2_synthetic_experts as smol  # noqa: E402
from analyze_smollm2_refusal_failure_modes import label_record  # noqa: E402
from analyze_smollm2_rq0 import CACHE_DIR, load_model_with_state, load_state_cache, write_csv  # noqa: E402
from evaluate_smollm2_refusal_quality import write_jsonl  # noqa: E402


RESULT_DIR = ROOT / "stage3" / "results"
PROMPT_PATH = ROOT / "stage3" / "data" / "smollm2_refusal_basis_prompts.jsonl"
DEFAULT_PATCH_SPECS = (
    "20:mlp",
    "20:block",
    "25:mlp",
    "25:block",
    "29:mlp",
    "29:block",
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
            raise ValueError(f"patch spec must look like layers:kind, got {raw}")
        layer_text, kind = raw.split(":", 1)
        layers = tuple(int(x) for x in layer_text.split("+") if x)
        if kind not in {"mlp", "attn", "block"}:
            raise ValueError(f"unknown activation kind {kind}; expected mlp, attn, or block")
        specs.append((layers, kind, raw))
    return specs


def load_base_state() -> dict[str, torch.Tensor]:
    base_model_cpu = AutoModelForCausalLM.from_pretrained(
        smol.BASE_ID,
        torch_dtype=torch.float16,
        device_map="cpu",
    )
    state = {k: v.detach().cpu().half() for k, v in base_model_cpu.state_dict().items()}
    del base_model_cpu
    return state


def module_for(model, layer: int, kind: str):
    block = model.model.layers[layer]
    if kind == "block":
        return block
    if kind == "mlp":
        return block.mlp
    if kind == "attn":
        return block.self_attn
    raise ValueError(kind)


def first_tensor(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def replace_first_tensor(output, tensor):
    original, rest = first_tensor(output)
    if rest is None:
        return tensor
    return (tensor, *rest)


def make_donor_hook(cache: dict[tuple[int, str], torch.Tensor], key: tuple[int, str]):
    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        cache[key] = tensor.detach()

    return hook


def make_recipient_hook(cache: dict[tuple[int, str], torch.Tensor], key: tuple[int, str], position: str):
    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        donor = cache[key].to(device=tensor.device, dtype=tensor.dtype)
        patched = tensor.clone()
        if position == "all":
            patched = donor
        elif position == "last":
            patched[:, -1:, :] = donor[:, -1:, :]
        elif position == "prompt":
            # During greedy decoding the current prefix is prompt + generated.
            # The hook receives only the current prefix, so patching all but the
            # final token approximates prompt/history patching while preserving
            # the current generation position.
            if patched.shape[1] > 1:
                patched[:, :-1, :] = donor[:, :-1, :]
        else:
            raise ValueError(f"unknown position {position}")
        return replace_first_tensor(output, patched)

    return hook


def install_hooks(donor, recipient, patch_points: tuple[tuple[int, str], ...], position: str):
    cache: dict[tuple[int, str], torch.Tensor] = {}
    donor_handles = []
    recipient_handles = []
    for key in patch_points:
        layer, kind = key
        donor_handles.append(module_for(donor, layer, kind).register_forward_hook(make_donor_hook(cache, key)))
        recipient_handles.append(module_for(recipient, layer, kind).register_forward_hook(make_recipient_hook(cache, key, position)))
    return cache, donor_handles, recipient_handles


def remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


@torch.no_grad()
def greedy_generate(
    recipient,
    tokenizer,
    user: str,
    device: str,
    max_new_tokens: int,
    *,
    donor=None,
    patch_points: tuple[tuple[int, str], ...] = (),
    position: str = "last",
) -> str:
    prompt = smol.chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    cache = {}
    donor_handles = []
    recipient_handles = []
    if donor is not None and patch_points:
        cache, donor_handles, recipient_handles = install_hooks(donor, recipient, patch_points, position)
    try:
        for _ in range(max_new_tokens):
            if donor is not None and patch_points:
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

    text = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    if "<|im_start|>assistant" in text:
        text = text.split("<|im_start|>assistant", 1)[1]
    text = text.replace("<|im_end|>", "").strip()
    return text


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
    patch_rows = [r for r in rows if r["kind"] == "activation_patch"]
    top_preserved = sorted(
        [r for r in patch_rows if bool(r["attempt_preserved"])],
        key=lambda r: (-float(r["problem_reduction_vs_recipient"]), -float(r["attempted_refusal_rate"]), r["patch_spec"]),
    )[:12]
    top_any = sorted(
        patch_rows,
        key=lambda r: (-float(r["problem_reduction_vs_recipient"]), -float(r["attempted_refusal_rate"]), r["patch_spec"]),
    )[:12]

    lines = [
        "# SmolLM2 Refusal Activation Patches",
        "",
        "Dynamic activation patching during greedy generation. The donor is run on the same current prefix, selected donor activations are inserted into the recipient, and the recipient chooses the next token.",
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

    def table(title: str, subset: list[dict[str, object]]) -> None:
        lines.extend(
            [
                "",
                f"## {title}",
                "",
                "| donor | patch | position | attempt | problem | reduction | repetition | artifact | unsafe | no refusal |",
                "|---|---|---|---:|---:|---:|---:|---:|---:|---:|",
            ]
        )
        if not subset:
            lines.append("| none | none | none | 0 | 0 | 0 | 0 | 0 | 0 | 0 |")
            return
        for row in subset:
            lines.append(
                f"| {row['donor']} | {row['patch_spec']} | {row['position']} | "
                f"{row['attempted_refusal_rate']:.3f} | {row['problem_response_rate']:.3f} | "
                f"{row['problem_reduction_vs_recipient']:.3f} | {row['repetition_rate']:.3f} | "
                f"{row['artifact_rate']:.3f} | {row['unsafe_or_contradictory_rate']:.3f} | {row['no_refusal_rate']:.3f} |"
            )

    table("Best Patches Preserving Attempt", top_preserved)
    table("Best Problem Reduction Overall", top_any)
    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "A useful activation patch should reduce problem responses without merely deleting the refusal attempt. If activation patching has the same tradeoff as weight patching, the quality components are still not isolated.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


@torch.no_grad()
def eval_variant(recipient_model, donor_model, tokenizer, prompts, args, *, variant, patch_points=()):
    records = []
    for prompt in prompts:
        generation = greedy_generate(
            recipient_model,
            tokenizer,
            str(prompt["user"]),
            args.device,
            args.max_new_tokens,
            donor=donor_model,
            patch_points=patch_points,
            position=args.position,
        )
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
    row = summarize_records(records)
    row.update(variant)
    return row, records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--state-cache-dir", type=Path, default=CACHE_DIR / "smollm2_state_cache")
    ap.add_argument("--prompt-path", type=Path, default=PROMPT_PATH)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    ap.add_argument("--recipient", default="merge_all_linear")
    ap.add_argument("--donors", default="base")
    ap.add_argument("--patch-specs", default=",".join(DEFAULT_PATCH_SPECS))
    ap.add_argument("--position", choices=("last", "prompt", "all"), default="last")
    ap.add_argument("--examples", type=int, default=16)
    ap.add_argument("--max-new-tokens", type=int, default=32)
    args = ap.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    prompts = read_jsonl(args.prompt_path)[: args.examples]
    patch_specs = parse_specs(args.patch_specs)
    donor_names = tuple(x for x in args.donors.split(",") if x)

    print("[load] tokenizer/states/models", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(smol.TOKENIZER_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_state = load_base_state()
    merges = load_state_cache(args.state_cache_dir)
    states: dict[str, dict[str, torch.Tensor]] = {"base": base_state, **merges}
    if args.recipient not in states:
        raise ValueError(f"unknown recipient {args.recipient}; available={sorted(states)}")
    for donor in donor_names:
        if donor not in states:
            raise ValueError(f"unknown donor {donor}; available={sorted(states)}")

    recipient_model = AutoModelForCausalLM.from_pretrained(smol.BASE_ID, torch_dtype=torch.float16).to(args.device)
    donor_model = AutoModelForCausalLM.from_pretrained(smol.BASE_ID, torch_dtype=torch.float16).to(args.device)
    load_model_with_state(recipient_model, states[args.recipient], args.device)

    all_records = []
    summary_rows = []
    row, records = eval_variant(
        recipient_model,
        None,
        tokenizer,
        prompts,
        args,
        variant={
            "kind": "recipient_baseline",
            "model": args.recipient,
            "recipient": args.recipient,
            "donor": "",
            "patch_spec": "none",
            "position": args.position,
        },
    )
    summary_rows.append(row)
    all_records.extend(records)

    for donor in donor_names:
        print(f"[donor] {donor}", flush=True)
        load_model_with_state(donor_model, states[donor], args.device)
        row, records = eval_variant(
            donor_model,
            None,
            tokenizer,
            prompts,
            args,
            variant={
                "kind": "donor_reference",
                "model": donor,
                "recipient": args.recipient,
                "donor": donor,
                "patch_spec": "donor_reference",
                "position": args.position,
            },
        )
        summary_rows.append(row)
        all_records.extend(records)
        for layers, kind, spec_text in patch_specs:
            print(f"[patch] donor={donor} spec={spec_text} position={args.position}", flush=True)
            patch_points = tuple((layer, kind) for layer in layers)
            row, records = eval_variant(
                recipient_model,
                donor_model,
                tokenizer,
                prompts,
                args,
                variant={
                    "kind": "activation_patch",
                    "model": f"{args.recipient}__{donor}__act_{spec_text}_{args.position}",
                    "recipient": args.recipient,
                    "donor": donor,
                    "patch_spec": spec_text,
                    "position": args.position,
                    "layers": "+".join(str(x) for x in layers),
                    "activation_kind": kind,
                },
                patch_points=patch_points,
            )
            summary_rows.append(row)
            all_records.extend(records)

    del recipient_model
    del donor_model
    torch.cuda.empty_cache()

    summary_rows = add_deltas(summary_rows)
    safe_recipient = args.recipient.replace("/", "_")
    safe_pos = args.position
    records_path = args.result_dir / f"smollm2_refusal_activation_patches_{safe_recipient}_{safe_pos}_records.jsonl"
    summary_csv = args.result_dir / f"smollm2_refusal_activation_patches_{safe_recipient}_{safe_pos}_summary.csv"
    summary_md = args.result_dir / f"SMOLLM2_REFUSAL_ACTIVATION_PATCHES_{safe_recipient.upper()}_{safe_pos.upper()}.md"
    meta_path = args.result_dir / f"smollm2_refusal_activation_patches_{safe_recipient}_{safe_pos}_summary.json"
    write_jsonl(records_path, all_records)
    write_csv_local(summary_csv, summary_rows)
    write_summary_md(summary_md, summary_rows)
    write_json(
        meta_path,
        {
            "recipient": args.recipient,
            "donors": donor_names,
            "patch_specs": [spec[2] for spec in patch_specs],
            "position": args.position,
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

