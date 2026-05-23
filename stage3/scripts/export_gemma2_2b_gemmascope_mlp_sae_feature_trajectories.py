#!/usr/bin/env python3
"""Export signed GemmaScope feature trajectories during patched generation."""

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
from run_gemma2_2b_gemmascope_mlp_sae_feature_specific_timing import (  # noqa: E402
    apply_grouped_mix_decode,
    grouped_specs_for_layer,
    make_variant_specs,
    resolve_group_indices,
)
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import (  # noqa: E402
    HARMFUL_PROMPTS,
    clean_assistant_text,
    collect_feature_stats,
    patch_position_mask,
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
    replace_first_tensor,
    select_sae_file,
)


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_gemmascope_mlp_sae_feature16048_signed_trajectories_v0"


def parse_features(raw: str) -> dict[int, list[int]]:
    out: dict[int, list[int]] = {}
    for item in raw.split(","):
        if not item.strip():
            continue
        left, right = item.split(":", 1)
        layer = int(left)
        feature_id = int(right)
        out.setdefault(layer, []).append(feature_id)
    return {layer: sorted(set(vals)) for layer, vals in out.items()}


def one_token(tokenizer, token_id: int) -> str:
    return tokenizer.decode([int(token_id)], skip_special_tokens=False).replace("\n", "\\n")


def position_kind(tokenizer, input_ids, attention_mask, prompt_length: int, pos: int) -> str:
    if pos >= prompt_length:
        return "generated"
    token = tokenizer.decode([int(input_ids[0, pos].item())], skip_special_tokens=False).strip()
    if token in {"", "<end_of_turn>", "<start_of_turn>", "model"}:
        return "assistant_boundary_or_template"
    if any(ch.isalnum() for ch in token) and token not in {"user", "model"} and not (token.startswith("<") and token.endswith(">")):
        return "contentish"
    return "prompt_template"


def feature_is_patched(variant, resolved, layer: int, feature_id: int, masks, pos: int) -> bool:
    if variant is None:
        return False
    for group in variant.groups:
        key = (group.kind, group.layer, group.value, group.mask)
        idx = resolved[key].get(layer)
        if idx is None or idx.numel() == 0:
            continue
        if feature_id not in {int(x) for x in idx.detach().cpu().tolist()}:
            continue
        if bool(masks[group.mask][0, pos].item()):
            return True
    return False


def selected_positions(tokenizer, input_ids, attention_mask, prompt_length: int, masks, step: int) -> list[int]:
    valid_len = int(attention_mask[0].sum().item())
    positions = {valid_len - 1}
    if step == 0:
        for mask in masks.values():
            for pos in torch.nonzero(mask[0, :prompt_length], as_tuple=False).flatten().detach().cpu().tolist():
                positions.add(int(pos))
    return sorted(pos for pos in positions if 0 <= pos < valid_len)


@torch.no_grad()
def run_condition(
    donor,
    recipient,
    tokenizer,
    saes,
    *,
    condition_name: str,
    variant,
    resolved,
    features_by_layer: dict[int, list[int]],
    patch_layers: tuple[int, ...],
    user: str,
    prompt_index: int,
    device: str,
    max_new_tokens: int,
    output_mode: str,
):
    layers = patch_layers
    donor_cache = {}
    recipient_cache = {}
    patched_cache = {}
    donor_handles = []
    recipient_handles = []
    current_masks = None
    if variant is None:
        needed_masks = ("assistant_boundary", "generated", "assistant_boundary_or_generated", "prompt_template", "prompt_template_or_generated")
    else:
        needed_masks = tuple(sorted({group.mask for group in variant.groups} | {"assistant_boundary", "generated", "prompt_template"}))

    def make_donor_hook(layer):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            donor_cache[layer] = tensor.detach()

        return hook

    def make_recipient_hook(layer):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            recipient_cache[layer] = tensor.detach()
            if variant is None:
                patched = tensor
            else:
                layer_groups = grouped_specs_for_layer(variant, resolved, layer)
                patched = apply_grouped_mix_decode(saes[layer], tensor, donor_cache[layer], layer_groups, current_masks)
            patched_cache[layer] = patched.detach()
            return replace_first_tensor(output, patched.to(device=tensor.device, dtype=tensor.dtype))

        return hook

    for layer in layers:
        donor_handles.append(module_mlp(donor, layer, output_mode).register_forward_hook(make_donor_hook(layer)))
        recipient_handles.append(module_mlp(recipient, layer, output_mode).register_forward_hook(make_recipient_hook(layer)))

    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_length = int(input_ids.shape[1])
    trajectory_rows = []
    final_text = ""
    try:
        for step in range(max_new_tokens):
            donor_cache.clear()
            recipient_cache.clear()
            patched_cache.clear()
            _ = donor(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            current_masks = {
                mask: patch_position_mask(tokenizer, input_ids, attention_mask, mask, prompt_length=prompt_length).to(device)
                for mask in needed_masks
            }
            out = recipient(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            positions = selected_positions(tokenizer, input_ids, attention_mask, prompt_length, current_masks, step)
            ids = [int(x) for x in input_ids[0].detach().cpu().tolist()]
            for pos in positions:
                kind = position_kind(tokenizer, input_ids, attention_mask, prompt_length, pos)
                for layer, feature_ids in features_by_layer.items():
                    donor_f = saes[layer].encode(donor_cache[layer][0, pos : pos + 1]).float()[0]
                    recipient_f = saes[layer].encode(recipient_cache[layer][0, pos : pos + 1]).float()[0]
                    patched_f = saes[layer].encode(patched_cache[layer][0, pos : pos + 1]).float()[0]
                    for feature_id in feature_ids:
                        donor_val = float(donor_f[feature_id].item())
                        recipient_val = float(recipient_f[feature_id].item())
                        patched_val = float(patched_f[feature_id].item())
                        trajectory_rows.append(
                            {
                                "condition": condition_name,
                                "prompt_index": prompt_index,
                                "prompt": user,
                                "step": step,
                                "position": pos,
                                "position_kind": kind,
                                "token_id": ids[pos],
                                "token": one_token(tokenizer, ids[pos]),
                                "layer": layer,
                                "feature_id": feature_id,
                                "donor_activation": donor_val,
                                "recipient_activation": recipient_val,
                                "patched_activation": patched_val,
                                "signed_delta": donor_val - recipient_val,
                                "abs_delta": abs(donor_val - recipient_val),
                                "patch_applied_to_feature": feature_is_patched(variant, resolved, layer, feature_id, current_masks, pos),
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
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)
    record = {"condition": condition_name, "split": "harmful", "prompt": user, "text": final_text}
    record.update(score_record("harmful", user, final_text))
    return trajectory_rows, record


def summarize(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    buckets: dict[tuple[str, int, int, str], list[dict[str, object]]] = {}
    for row in rows:
        key = (str(row["condition"]), int(row["layer"]), int(row["feature_id"]), str(row["position_kind"]))
        buckets.setdefault(key, []).append(row)
    out = []
    for (condition, layer, feature_id, kind), vals in sorted(buckets.items()):
        out.append(
            {
                "condition": condition,
                "layer": layer,
                "feature_id": feature_id,
                "position_kind": kind,
                "n": len(vals),
                "mean_signed_delta": sum(float(x["signed_delta"]) for x in vals) / len(vals),
                "mean_abs_delta": sum(float(x["abs_delta"]) for x in vals) / len(vals),
                "mean_donor_activation": sum(float(x["donor_activation"]) for x in vals) / len(vals),
                "mean_recipient_activation": sum(float(x["recipient_activation"]) for x in vals) / len(vals),
                "mean_patched_activation": sum(float(x["patched_activation"]) for x in vals) / len(vals),
                "patched_fraction": sum(bool(x["patch_applied_to_feature"]) for x in vals) / len(vals),
            }
        )
    return out


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


def write_summary(path: Path, score_rows, summary_rows, args) -> None:
    lines = [
        "# Gemma-2-2B GemmaScope Signed Feature Trajectories",
        "",
        f"Feature-selection prompts: `{args.basis_start}:{args.basis_start + args.basis_examples_per_split}` per split.",
        f"Trajectory prompts: harmful `{args.prompt_start}:{args.prompt_start + args.examples}`.",
        f"Variant set: `{args.variant_set}`.",
        f"Prefix k: `{args.prefix_k}`.",
        f"Features: `{args.features}`.",
        "",
        "## Generation Scores",
        "",
        "| condition | harmful clean | unsafe | fake-ID prompt |",
        "|---|---:|---:|---|",
    ]
    for row in score_rows:
        fake = "pass" if bool(row.get("ok")) and "fake ID" in str(row["prompt"]) else "fail"
        lines.append(
            f"| `{row['condition']}` | {1.0 if row.get('ok') else 0.0:.3f} | "
            f"{1.0 if row.get('harmful_unsafe_continuation') else 0.0:.3f} | {fake} |"
        )
    lines.extend(
        [
            "",
            "## Signed Delta Summary",
            "",
            "| condition | layer | feature | position kind | n | mean signed delta | mean abs delta | patched fraction |",
            "|---|---:|---:|---|---:|---:|---:|---:|",
        ]
    )
    for row in summary_rows:
        lines.append(
            f"| `{row['condition']}` | {row['layer']} | {row['feature_id']} | `{row['position_kind']}` | "
            f"{row['n']} | {row['mean_signed_delta']:.4f} | {row['mean_abs_delta']:.4f} | {row['patched_fraction']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Caveat",
            "",
            "- Rows summarize only selected prompt/template positions at step 0 and the autoregressive last-token position at each generation step.",
            "- This is a trajectory diagnostic, not a standalone causal intervention result.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--layers", default="12,13,14,15,16,17,18,19,20")
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--basis-start", type=int, default=0)
    ap.add_argument("--basis-examples-per-split", type=int, default=4)
    ap.add_argument("--prompt-start", type=int, default=8)
    ap.add_argument("--examples", type=int, default=1)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--feature-token-filter", choices=("all", "contentish"), default="all")
    ap.add_argument("--prefix-k", type=int, default=896)
    ap.add_argument("--variant-set", choices=("l19_narrow", "l19_broad", "l19_both", "narrow", "broad", "both"), default="l19_narrow")
    ap.add_argument("--variant-labels", default="narrow_prefix_only,narrow_f16048_generated,narrow_f16048_abog")
    ap.add_argument("--features", default="19:16048,12:40,12:12075")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_ints(args.layers)
    features_by_layer = parse_features(args.features)
    all_layers = tuple(sorted(set(layers) | set(features_by_layer)))
    specs = make_variant_specs(args.prefix_k, args.variant_set)
    if args.variant_labels.strip():
        keep = {x.strip() for x in args.variant_labels.split(",") if x.strip()}
        specs = [spec for spec in specs if spec.label in keep]
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[files] selecting GemmaScope MLP SAEs", flush=True)
    files = {layer: select_sae_file(layer, args.l0_target) for layer in all_layers}
    for layer, filename in files.items():
        print(f"[files] L{layer}: {filename}", flush=True)
    print("[load] SAEs", flush=True)
    saes = {layer: load_sae(filename, device=args.device, dtype=sae_dtype, cache_dir=cache_dir) for layer, filename in files.items()}
    print("[load] base donor", flush=True)
    donor = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=model_dtype).to(args.device)
    donor.eval()
    print("[load] abliterated recipient", flush=True)
    recipient = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=model_dtype).to(args.device)
    recipient.eval()

    print("[features] collecting calibration feature stats", flush=True)
    stats = collect_feature_stats(
        donor,
        recipient,
        tokenizer,
        saes,
        layers=all_layers,
        examples_per_split=args.basis_examples_per_split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        prompt_start=args.basis_start,
        feature_token_filter=args.feature_token_filter,
        device=args.device,
        output_mode=args.output_mode,
    )
    resolved, group_rows = resolve_group_indices(stats, all_layers, specs, args.device)

    prompts = list(enumerate(prompt_slice(HARMFUL_PROMPTS, args.prompt_start, args.examples), start=args.prompt_start))
    trajectory_rows = []
    score_rows = []
    print("[trajectory] running conditions", flush=True)
    for prompt_index, user in prompts:
        for spec in specs:
            print(f"[eval] {spec.label} prompt={prompt_index}", flush=True)
            rows, score = run_condition(
                donor,
                recipient,
                tokenizer,
                saes,
                condition_name=spec.label,
                variant=spec,
                resolved=resolved,
                features_by_layer=features_by_layer,
                patch_layers=all_layers,
                user=str(user),
                prompt_index=prompt_index,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                output_mode=args.output_mode,
            )
            trajectory_rows.extend(rows)
            score_rows.append(score)
    summary_rows = summarize(trajectory_rows)

    trajectory_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_trajectories.csv"
    score_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_trajectory_scores.csv"
    summary_csv_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_trajectory_summary.csv"
    groups_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_trajectory_groups.csv"
    summary_path = args.result_dir / "GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE_TRAJECTORY_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(trajectory_path, trajectory_rows)
    write_csv(score_path, score_rows)
    write_csv(summary_csv_path, summary_rows)
    write_csv(groups_path, group_rows)
    write_summary(summary_path, score_rows, summary_rows, args)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "layers": list(all_layers),
                "features": features_by_layer,
                "basis_start": args.basis_start,
                "basis_examples_per_split": args.basis_examples_per_split,
                "prompt_start": args.prompt_start,
                "examples": args.examples,
                "prefix_k": args.prefix_k,
                "variant_set": args.variant_set,
                "variant_labels": [spec.label for spec in specs],
                "outputs": {
                    "trajectories": str(trajectory_path),
                    "scores": str(score_path),
                    "summary_csv": str(summary_csv_path),
                    "groups": str(groups_path),
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
