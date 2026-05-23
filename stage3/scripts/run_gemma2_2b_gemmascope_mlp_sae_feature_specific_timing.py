#!/usr/bin/env python3
"""Feature-specific timing interventions for the GemmaScope MLP-SAE repair."""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
sys.path.insert(0, str(ROOT / "stage3" / "scripts"))

from run_gemma2_2b_activation_patch_target_loss import MODEL_IDS  # noqa: E402
from run_gemma2_2b_gemmascope_mlp_sae_feature_subsets import (  # noqa: E402
    BENIGN_PROMPTS,
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


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_gemmascope_mlp_sae_feature16048_feature_specific_timing_v0"
DEFAULT_LAYERS = tuple(range(12, 21))
MASK_ALIASES = {
    "ab": "assistant_boundary",
    "gen": "generated",
    "abog": "assistant_boundary_or_generated",
    "pt": "prompt_template",
    "ptog": "prompt_template_or_generated",
}


@dataclass(frozen=True)
class GroupSpec:
    name: str
    layer: int | None
    kind: str
    value: int
    mask: str


@dataclass(frozen=True)
class VariantSpec:
    label: str
    groups: tuple[GroupSpec, ...]


def normalize_mask(mask: str) -> str:
    return MASK_ALIASES.get(mask, mask)


def make_variant_specs(prefix_k: int, variant_set: str) -> list[VariantSpec]:
    specs: list[VariantSpec] = []
    include_narrow = variant_set in {"narrow", "both", "l19_narrow", "l19_both"}
    include_broad = variant_set in {"broad", "both", "l19_broad", "l19_both"}
    include_l12_narrow = variant_set in {"narrow", "both"}
    include_l12_broad = variant_set in {"broad", "both"}

    def prefix(name: str, mask: str) -> GroupSpec:
        return GroupSpec(name=name, layer=None, kind="prefix", value=prefix_k, mask=normalize_mask(mask))

    def feature(name: str, layer: int, feature_id: int, mask: str) -> GroupSpec:
        return GroupSpec(name=name, layer=layer, kind="feature_id", value=feature_id, mask=normalize_mask(mask))

    def rank(name: str, layer: int, rank_id: int, mask: str) -> GroupSpec:
        return GroupSpec(name=name, layer=layer, kind="rank", value=rank_id, mask=normalize_mask(mask))

    if include_narrow:
        base = prefix(f"prefix_k{prefix_k}", "abog")
        f_abog = feature("l19_f16048", 19, 16048, "abog")
        specs.extend(
            [
                VariantSpec("narrow_prefix_only", (base,)),
                VariantSpec("narrow_f16048_assistant_boundary", (base, feature("l19_f16048", 19, 16048, "ab"))),
                VariantSpec("narrow_f16048_generated", (base, feature("l19_f16048", 19, 16048, "gen"))),
                VariantSpec("narrow_f16048_abog", (base, f_abog)),
            ]
        )
        if include_l12_narrow:
            specs.extend(
                [
                    VariantSpec("narrow_f16048_abog_l12r274_assistant_boundary", (base, f_abog, rank("l12_r274", 12, 274, "ab"))),
                    VariantSpec("narrow_f16048_abog_l12r274_generated", (base, f_abog, rank("l12_r274", 12, 274, "gen"))),
                    VariantSpec("narrow_f16048_abog_l12r274_abog", (base, f_abog, rank("l12_r274", 12, 274, "abog"))),
                    VariantSpec("narrow_f16048_abog_l12r295_assistant_boundary", (base, f_abog, rank("l12_r295", 12, 295, "ab"))),
                    VariantSpec("narrow_f16048_abog_l12r295_generated", (base, f_abog, rank("l12_r295", 12, 295, "gen"))),
                    VariantSpec("narrow_f16048_abog_l12r295_abog", (base, f_abog, rank("l12_r295", 12, 295, "abog"))),
                ]
            )

    if include_broad:
        base = prefix(f"prefix_k{prefix_k}", "ptog")
        f_ptog = feature("l19_f16048", 19, 16048, "ptog")
        specs.extend(
            [
                VariantSpec("broad_prefix_only", (base,)),
                VariantSpec("broad_f16048_prompt_template", (base, feature("l19_f16048", 19, 16048, "pt"))),
                VariantSpec("broad_f16048_generated", (base, feature("l19_f16048", 19, 16048, "gen"))),
                VariantSpec("broad_f16048_ptog", (base, f_ptog)),
            ]
        )
        if include_l12_broad:
            specs.extend(
                [
                    VariantSpec("broad_f16048_ptog_l12r274_prompt_template", (base, f_ptog, rank("l12_r274", 12, 274, "pt"))),
                    VariantSpec("broad_f16048_ptog_l12r274_generated", (base, f_ptog, rank("l12_r274", 12, 274, "gen"))),
                    VariantSpec("broad_f16048_ptog_l12r274_ptog", (base, f_ptog, rank("l12_r274", 12, 274, "ptog"))),
                    VariantSpec("broad_f16048_ptog_l12r295_prompt_template", (base, f_ptog, rank("l12_r295", 12, 295, "pt"))),
                    VariantSpec("broad_f16048_ptog_l12r295_generated", (base, f_ptog, rank("l12_r295", 12, 295, "gen"))),
                    VariantSpec("broad_f16048_ptog_l12r295_ptog", (base, f_ptog, rank("l12_r295", 12, 295, "ptog"))),
                ]
            )

    return specs


def resolve_group_indices(stats, layers: tuple[int, ...], specs: list[VariantSpec], device: str):
    cache: dict[tuple[str, int | None, int, str], dict[int, torch.Tensor]] = {}
    rows = []

    for variant in specs:
        for group in variant.groups:
            key = (group.kind, group.layer, group.value, group.mask)
            if key in cache:
                continue
            by_layer: dict[int, torch.Tensor] = {}
            if group.kind == "prefix":
                for layer in layers:
                    score = stats[layer]["harm_delta_abs"]
                    k = min(group.value, int(score.numel()))
                    idx = torch.topk(score, k).indices if float(score.abs().sum().item()) else torch.empty(0, dtype=torch.long)
                    by_layer[layer] = idx.to(device=device, dtype=torch.long)
                    rows.append(
                        {
                            "group": group.name,
                            "kind": group.kind,
                            "layer": layer,
                            "value": group.value,
                            "mask": group.mask,
                            "selected_features": int(idx.numel()),
                        }
                    )
            elif group.kind == "feature_id":
                if group.layer is None:
                    raise ValueError("feature_id group needs a layer")
                score = stats[group.layer]["harm_delta_abs"]
                if group.value >= int(score.numel()):
                    raise ValueError(f"feature {group.value} outside layer {group.layer} dimension {score.numel()}")
                by_layer[group.layer] = torch.tensor([group.value], device=device, dtype=torch.long)
                rows.append(
                    {
                        "group": group.name,
                        "kind": group.kind,
                        "layer": group.layer,
                        "value": group.value,
                        "mask": group.mask,
                        "selected_features": 1,
                    }
                )
            elif group.kind == "rank":
                if group.layer is None:
                    raise ValueError("rank group needs a layer")
                score = stats[group.layer]["harm_delta_abs"]
                if group.value < 1 or group.value > int(score.numel()):
                    raise ValueError(f"rank {group.value} outside layer {group.layer} dimension {score.numel()}")
                idx = torch.topk(score, group.value).indices[group.value - 1 : group.value]
                by_layer[group.layer] = idx.to(device=device, dtype=torch.long)
                rows.append(
                    {
                        "group": group.name,
                        "kind": group.kind,
                        "layer": group.layer,
                        "value": group.value,
                        "feature_id": int(idx.item()),
                        "mask": group.mask,
                        "selected_features": 1,
                    }
                )
            else:
                raise ValueError(group.kind)
            cache[key] = by_layer
    return cache, rows


def grouped_specs_for_layer(
    variant: VariantSpec,
    resolved: dict[tuple[str, int | None, int, str], dict[int, torch.Tensor]],
    layer: int,
) -> list[tuple[torch.Tensor, str]]:
    groups = []
    for group in variant.groups:
        key = (group.kind, group.layer, group.value, group.mask)
        idx = resolved[key].get(layer)
        if idx is not None and idx.numel() > 0:
            groups.append((idx, group.mask))
    return groups


def apply_grouped_mix_decode(sae, recipient_out, donor_out, groups, masks):
    if not groups:
        return recipient_out
    donor_f = sae.encode(donor_out)
    recipient_f = sae.encode(recipient_out)
    mixed = recipient_f.clone()
    feature_dim = mixed.shape[-1]
    flat_mixed = mixed.reshape(-1, feature_dim)
    flat_donor = donor_f.reshape(-1, feature_dim)
    union_mask = torch.zeros(recipient_out.shape[:-1], device=recipient_out.device, dtype=torch.bool)
    for idx, mask_name in groups:
        position_mask = masks[mask_name].to(device=recipient_out.device, dtype=torch.bool)
        union_mask |= position_mask
        mask = position_mask.reshape(-1)
        rows = torch.nonzero(mask, as_tuple=False).flatten()
        if rows.numel() == 0:
            continue
        row_idx = rows.unsqueeze(1)
        feat_idx = idx.to(device=mixed.device).unsqueeze(0)
        flat_mixed[row_idx, feat_idx] = flat_donor[row_idx, feat_idx]
    decoded = sae.decode(mixed)
    return torch.where(union_mask.unsqueeze(-1), decoded, recipient_out)


@torch.no_grad()
def feature_timing_generate(
    donor,
    recipient,
    tokenizer,
    saes,
    user: str,
    *,
    layers: tuple[int, ...],
    variant: VariantSpec,
    resolved,
    device: str,
    max_new_tokens: int,
    output_mode: str,
) -> str:
    donor_cache = {}
    donor_handles = []
    recipient_handles = []
    current_masks = None
    needed_masks = sorted({group.mask for group in variant.groups})

    def make_donor_hook(layer):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            donor_cache[layer] = tensor.detach()

        return hook

    def make_recipient_hook(layer):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            layer_groups = grouped_specs_for_layer(variant, resolved, layer)
            patched = apply_grouped_mix_decode(saes[layer], tensor, donor_cache[layer], layer_groups, current_masks)
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
    try:
        for _ in range(max_new_tokens):
            donor_cache.clear()
            _ = donor(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            current_masks = {
                mask: patch_position_mask(tokenizer, input_ids, attention_mask, mask, prompt_length=prompt_length).to(device)
                for mask in needed_masks
            }
            out = recipient(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            next_id = out.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            input_ids = torch.cat([input_ids, next_id], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_id)], dim=1)
            if int(next_id.item()) == tokenizer.eos_token_id:
                break
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)
    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    return clean_assistant_text(tokenizer, prompt, decoded)


def summarize_generation(model_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    fake_id = [row for row in harmful if "fake ID" in str(row["prompt"])]
    return {
        "model": model_name,
        "n": len(rows),
        "harmful_ok_rate": sum(bool(row["ok"]) for row in harmful) / max(len(harmful), 1),
        "harmful_attempted_refusal_rate": sum(bool(row.get("harmful_attempted_refusal")) for row in harmful) / max(len(harmful), 1),
        "harmful_bad_attempt_rate": sum(bool(row.get("harmful_bad_refusal_attempt")) for row in harmful) / max(len(harmful), 1),
        "harmful_unsafe_continuation_rate": sum(bool(row.get("harmful_unsafe_continuation")) for row in harmful) / max(len(harmful), 1),
        "benign_ok_rate": sum(bool(row["ok"]) for row in benign) / max(len(benign), 1),
        "benign_over_refusal_rate": sum(bool(row.get("benign_over_refusal")) for row in benign) / max(len(benign), 1),
        "fake_id_ok": bool(fake_id and fake_id[0].get("ok")),
    }


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


def write_summary(path: Path, metrics, group_rows, specs, args) -> None:
    lines = [
        "# Gemma-2-2B GemmaScope Feature-Specific Timing",
        "",
        f"Layers: `{','.join(str(x) for x in parse_ints(args.layers))}`.",
        f"Feature-selection prompts: `{args.basis_start}:{args.basis_start + args.basis_examples_per_split}` per split.",
        f"Evaluation prompts: `{args.eval_start}:{args.eval_start + args.examples_per_split}` per split.",
        f"Variant set: `{args.variant_set}`.",
        f"Prefix k: `{args.prefix_k}`.",
        "",
        "## Generation",
        "",
        "| model | harmful clean | harmful attempt | harmful bad | unsafe | benign helpful | fake-ID |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in metrics:
        fake = "pass" if row["fake_id_ok"] else "fail"
        lines.append(
            f"| `{row['model']}` | {row['harmful_ok_rate']:.3f} | "
            f"{row['harmful_attempted_refusal_rate']:.3f} | "
            f"{row['harmful_bad_attempt_rate']:.3f} | "
            f"{row['harmful_unsafe_continuation_rate']:.3f} | "
            f"{row['benign_ok_rate']:.3f} | {fake} |"
        )
    lines.extend(["", "## Variant Groups", ""])
    for spec in specs:
        parts = []
        for group in spec.groups:
            layer = "all" if group.layer is None else f"L{group.layer}"
            parts.append(f"{group.name}:{group.kind}:{layer}:{group.value}@{group.mask}")
        lines.append(f"- `{spec.label}`: " + "; ".join(parts))
    lines.extend(["", "## Resolved Rank Features", ""])
    seen = set()
    for row in group_rows:
        if row.get("kind") != "rank":
            continue
        key = (row["layer"], row["value"], row.get("feature_id"))
        if key in seen:
            continue
        seen.add(key)
        lines.append(f"- L{row['layer']} rank {row['value']} resolved to feature ID `{row['feature_id']}`.")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--basis-start", type=int, default=0)
    ap.add_argument("--basis-examples-per-split", type=int, default=8)
    ap.add_argument("--eval-start", type=int, default=8)
    ap.add_argument("--examples-per-split", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--feature-token-filter", choices=("all", "contentish"), default="all")
    ap.add_argument("--prefix-k", type=int, default=256)
    ap.add_argument(
        "--variant-set",
        choices=("narrow", "broad", "both", "l19_narrow", "l19_broad", "l19_both"),
        default="both",
    )
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_ints(args.layers)
    specs = make_variant_specs(args.prefix_k, args.variant_set)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    sae_dtype = {"float16": torch.float16, "bfloat16": torch.bfloat16, "float32": torch.float32}[args.sae_dtype]
    cache_dir = os.environ.get("HF_HOME")

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    print("[files] selecting GemmaScope MLP SAEs", flush=True)
    files = {layer: select_sae_file(layer, args.l0_target) for layer in layers}
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
        layers=layers,
        examples_per_split=args.basis_examples_per_split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        prompt_start=args.basis_start,
        feature_token_filter=args.feature_token_filter,
        device=args.device,
        output_mode=args.output_mode,
    )
    resolved, group_rows = resolve_group_indices(stats, layers, specs, args.device)

    prompts = [("harmful", x) for x in prompt_slice(HARMFUL_PROMPTS, args.eval_start, args.examples_per_split)] + [
        ("benign", x) for x in prompt_slice(BENIGN_PROMPTS, args.eval_start, args.examples_per_split)
    ]
    all_records = []
    metrics = []
    print("[generation] running feature-specific timing variants", flush=True)
    for spec in specs:
        rows = []
        model_name = f"feature_timing_{spec.label}"
        print(f"[eval] {model_name}", flush=True)
        for split, user in prompts:
            text = feature_timing_generate(
                donor,
                recipient,
                tokenizer,
                saes,
                str(user),
                layers=layers,
                variant=spec,
                resolved=resolved,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                output_mode=args.output_mode,
            )
            record = {"model": model_name, "split": split, "prompt": str(user), "text": text}
            record.update(score_record(split, str(user), text))
            rows.append(record)
        all_records.extend(rows)
        metrics.append(summarize_generation(model_name, rows))

    metrics_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_specific_timing_metrics.csv"
    records_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_specific_timing_records.jsonl"
    group_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_specific_timing_groups.csv"
    summary_path = args.result_dir / "GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE_SPECIFIC_TIMING_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, metrics)
    write_jsonl(records_path, all_records)
    write_csv(group_path, group_rows)
    write_summary(summary_path, metrics, group_rows, specs, args)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "layers": list(layers),
                "output_mode": args.output_mode,
                "feature_token_filter": args.feature_token_filter,
                "basis_start": args.basis_start,
                "basis_examples_per_split": args.basis_examples_per_split,
                "eval_start": args.eval_start,
                "examples_per_split": args.examples_per_split,
                "prefix_k": args.prefix_k,
                "variant_set": args.variant_set,
                "variants": [spec.label for spec in specs],
                "outputs": {
                    "metrics": str(metrics_path),
                    "records": str(records_path),
                    "groups": str(group_path),
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
