#!/usr/bin/env python3
"""Audit first-token logits for SAE bundle patch files."""

from __future__ import annotations

import argparse
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
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import (  # noqa: E402
    apply_feature_patch,
    chat_prompt,
    patch_position_mask,
)
from run_gemma2_2b_linear_merge_sae_bundle_patch import (  # noqa: E402
    load_prompt_rows,
    make_selected,
    parse_bundles,
    write_csv,
)
from run_gemma2_2b_linear_weight_merge_sweep import set_linear_merge_weights  # noqa: E402
from validate_gemma2_2b_gemmascope_mlp_sae import (  # noqa: E402
    SAE_REPO,
    first_tensor,
    load_sae,
    module_mlp,
    parse_ints,
    remove_hooks,
    replace_first_tensor,
    select_sae_file,
)


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_linear_merge_sae_bundle_first_token_logits_v0"


def one_token_id(tokenizer, text: str) -> int:
    ids = tokenizer(text, add_special_tokens=False)["input_ids"]
    if len(ids) != 1:
        raise ValueError(f"{text!r} is not one token: {ids}")
    return int(ids[0])


@torch.no_grad()
def unpatched_logits(model, tokenizer, user: str, device: str) -> torch.Tensor:
    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    out = model(**enc, use_cache=False)
    return out.logits[0, -1].detach().float().cpu()


@torch.no_grad()
def patched_logits(donor, recipient, tokenizer, saes, variant, selected, user: str, args) -> torch.Tensor:
    donor_cache = {}
    donor_handles = []
    recipient_handles = []
    current_patch_mask = None

    def make_donor_hook(layer):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            donor_cache[layer] = tensor.detach()

        return hook

    def make_recipient_hook(layer):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            patched = apply_feature_patch(saes[layer], tensor, donor_cache[layer], variant, selected[str(variant["label"])][layer])
            if current_patch_mask is not None:
                mask = current_patch_mask.to(device=tensor.device).unsqueeze(-1)
                patched = torch.where(mask, patched, tensor)
            return replace_first_tensor(output, patched.to(device=tensor.device, dtype=tensor.dtype))

        return hook

    for layer in args.layers:
        donor_handles.append(module_mlp(donor, layer, args.output_mode).register_forward_hook(make_donor_hook(layer)))
        recipient_handles.append(module_mlp(recipient, layer, args.output_mode).register_forward_hook(make_recipient_hook(layer)))

    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(args.device)
    prompt_length = int(enc["input_ids"].shape[1])
    try:
        donor(**enc, use_cache=False)
        current_patch_mask = patch_position_mask(
            tokenizer,
            enc["input_ids"],
            enc["attention_mask"],
            args.patch_token_filter,
            prompt_length=prompt_length,
        )
        out = recipient(**enc, use_cache=False)
        return out.logits[0, -1].detach().float().cpu()
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)


def build_row(tokenizer, token_i: int, token_it: int, row: dict[str, object]) -> dict[str, object]:
    logits = row.pop("logits")
    top_value, top_id = torch.max(logits, dim=0)
    return {
        **row,
        "token_i_id": token_i,
        "token_it_id": token_it,
        "logit_i": float(logits[token_i].item()),
        "logit_it": float(logits[token_it].item()),
        "i_minus_it": float((logits[token_i] - logits[token_it]).item()),
        "top_token_id": int(top_id.item()),
        "top_token": tokenizer.decode([int(top_id)], skip_special_tokens=False),
        "top_logit": float(top_value.item()),
    }


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    groups: dict[tuple[str, str], list[dict[str, object]]] = {}
    for row in rows:
        groups.setdefault((str(row["condition"]), str(row["split"])), []).append(row)
    out = []
    for (condition, split), bucket in sorted(groups.items()):
        n = len(bucket)
        margins = [float(row["i_minus_it"]) for row in bucket]
        out.append(
            {
                "condition": condition,
                "split": split,
                "n": n,
                "top_i_rate": sum(1 for row in bucket if row["top_token"] == "I") / n,
                "top_it_rate": sum(1 for row in bucket if row["top_token"] == "It") / n,
                "mean_i_minus_it": sum(margins) / n,
                "min_i_minus_it": min(margins),
                "max_i_minus_it": max(margins),
            }
        )
    return out


def write_summary(path: Path, args, summary_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Gemma-2-2B Linear Merge SAE Bundle First-Token Logit Audit",
        "",
        f"Prompt file: `{args.prompt_jsonl}`.",
        f"Bundles file: `{args.bundles_file}`.",
        f"Donor alpha: `{args.donor_alpha:g}`.",
        f"Recipient alpha: `{args.recipient_alpha:g}`.",
        f"Patch mode: `{args.patch_mode}`.",
        f"Patch token filter: `{args.patch_token_filter}`.",
        "",
        "| condition | split | n | top I | top It | mean I-It | min I-It | max I-It |",
        "|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| `{row['condition']}` | `{row['split']}` | {row['n']} | "
            f"{float(row['top_i_rate']):.3f} | {float(row['top_it_rate']):.3f} | "
            f"{float(row['mean_i_minus_it']):.4f} | {float(row['min_i_minus_it']):.4f} | "
            f"{float(row['max_i_minus_it']):.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float32")
    ap.add_argument("--layers", default="20")
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--donor-alpha", type=float, default=1.0)
    ap.add_argument("--recipient-alpha", type=float, default=0.75)
    ap.add_argument("--prompt-jsonl", type=Path, required=True)
    ap.add_argument("--patch-token-filter", default="assistant_boundary_final_newline")
    ap.add_argument(
        "--patch-mode",
        choices=("full_decode", "donor_subset_decode", "recipient_recon", "delta_add_all", "mix_decode", "delta_add", "feature_subtract"),
        default="delta_add",
    )
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--bundles-file", type=Path, required=True)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    args = ap.parse_args()
    args.layers = parse_ints(args.layers)
    return args


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")

    prompts = load_prompt_rows(args.prompt_jsonl)
    bundles = parse_bundles(args.bundles_file.read_text(encoding="utf-8").strip())

    print("[load] tokenizer", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"], cache_dir=cache_dir)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    token_i = one_token_id(tokenizer, "I")
    token_it = one_token_id(tokenizer, "It")

    print("[files] selecting GemmaScope MLP SAEs", flush=True)
    files = {layer: select_sae_file(layer, args.l0_target) for layer in args.layers}
    for layer, filename in files.items():
        print(f"[files] L{layer}: {filename}", flush=True)
    print("[load] SAEs", flush=True)
    saes = {layer: load_sae(filename, device=args.device, dtype=sae_dtype, cache_dir=cache_dir) for layer, filename in files.items()}

    print("[load] base donor for weight construction", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    base_model.eval()
    print("[load] donor model", flush=True)
    donor_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    donor_model.eval()
    recipient_params_cpu = {name: param.detach().cpu().clone() for name, param in donor_model.named_parameters()}
    set_linear_merge_weights(donor_model, base_model, recipient_params_cpu, args.donor_alpha, args.device)
    print("[load] recipient model", flush=True)
    recipient_model = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype, cache_dir=cache_dir).to(args.device)
    recipient_model.eval()
    set_linear_merge_weights(recipient_model, base_model, recipient_params_cpu, args.recipient_alpha, args.device)
    del base_model
    torch.cuda.empty_cache()

    variants, selected, count_rows = make_selected(bundles, args.layers, args.device, args.patch_mode)

    raw_rows = []
    if not args.skip_baselines:
        for condition, model in ((f"linear_alpha_{args.recipient_alpha:g}", recipient_model), (f"linear_alpha_{args.donor_alpha:g}", donor_model)):
            for prompt_index, (split, user) in enumerate(prompts, start=1):
                print(f"[eval] {condition} prompt {prompt_index}/{len(prompts)} ({split})", flush=True)
                raw_rows.append(
                    {
                        "prompt_index": prompt_index,
                        "split": split,
                        "prompt": user,
                        "condition": condition,
                        "logits": unpatched_logits(model, tokenizer, user, args.device),
                    }
                )

    for variant in variants:
        label = str(variant["label"])
        condition = f"bundle_patch_{label}"
        for prompt_index, (split, user) in enumerate(prompts, start=1):
            print(f"[eval] {condition} prompt {prompt_index}/{len(prompts)} ({split})", flush=True)
            raw_rows.append(
                {
                    "prompt_index": prompt_index,
                    "split": split,
                    "prompt": user,
                    "condition": condition,
                    "patch_mode": args.patch_mode,
                    "patch_token_filter": args.patch_token_filter,
                    "logits": patched_logits(donor_model, recipient_model, tokenizer, saes, variant, selected, user, args),
                }
            )

    detail_rows = [build_row(tokenizer, token_i, token_it, row) for row in raw_rows]
    summary_rows = summarize(detail_rows)
    detail_path = args.result_dir / "first_token_logits.csv"
    summary_csv_path = args.result_dir / "first_token_logit_summary.csv"
    counts_path = args.result_dir / "selected_feature_counts.csv"
    summary_path = args.result_dir / "BUNDLE_FIRST_TOKEN_LOGIT_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(detail_path, detail_rows)
    write_csv(summary_csv_path, summary_rows)
    write_csv(counts_path, count_rows)
    write_summary(summary_path, args, summary_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "prompt_jsonl": str(args.prompt_jsonl),
                "bundles_file": str(args.bundles_file),
                "donor_alpha": args.donor_alpha,
                "recipient_alpha": args.recipient_alpha,
                "patch_mode": args.patch_mode,
                "patch_token_filter": args.patch_token_filter,
                "outputs": {
                    "detail": str(detail_path),
                    "summary_csv": str(summary_csv_path),
                    "counts": str(counts_path),
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
