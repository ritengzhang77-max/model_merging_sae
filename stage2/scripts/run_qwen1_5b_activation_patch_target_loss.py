#!/usr/bin/env python3
"""Teacher-forced activation patching for Qwen2.5-1.5B refusal repair.

This compares static module replacement against activation-level patching on the
same prompt/completion pairs before attempting slow dynamic generation patches.
"""

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
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
from analyze_qwen1_5b_refusal_rq1_rq2 import (  # noqa: E402
    BENIGN_PROMPTS,
    HARMFUL_PROMPTS,
    MODEL_IDS,
    REFUSAL_TARGET,
    collate_labeled,
    labeled_example,
)


RESULT_DIR = ROOT / "stage2" / "results" / "qwen1_5b_activation_patch_target_loss"
DEFAULT_PATCH_SPECS = (
    "12+13+14+15+16+17+18+19+20+21+22+23+24:mlp",
    "12+13+14+15+16+17+18+19+20+21:mlp",
    "12+13+14+15+16+17+18+19:mlp",
)


def parse_specs(raw: str) -> list[tuple[tuple[int, ...], str, str]]:
    out = []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        layer_text, module = item.split(":", 1)
        layers = tuple(int(part) for part in layer_text.split("+") if part)
        if module not in {"mlp", "attn", "block"}:
            raise ValueError(f"unknown module for activation patch: {module}")
        out.append((layers, module, item))
    return out


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
    _first, rest = first_tensor(output)
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
        if position == "all":
            patched = donor
        elif position == "target":
            patched = tensor.clone()
            patched[:, -1:, :] = donor[:, -1:, :]
        else:
            raise ValueError(position)
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


def make_batches(tokenizer, examples_per_split: int, max_length: int, batch_size: int):
    rows = []
    for split, prompts in (
        ("harmful_refusal", HARMFUL_PROMPTS[:examples_per_split]),
        ("benign_refusal", BENIGN_PROMPTS[:examples_per_split]),
    ):
        for user in prompts:
            rows.append({"split": split, "user": user})
    examples = [
        {
            "split": row["split"],
            **labeled_example(tokenizer, str(row["user"]), REFUSAL_TARGET, max_length),
        }
        for row in rows
    ]
    batches = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batch = collate_labeled(tokenizer, chunk)
        batches.append((chunk, batch))
    return batches


@torch.no_grad()
def losses_for_model(model, batches, device: str) -> dict[str, float]:
    totals = {}
    counts = {}
    for chunk, batch in batches:
        batch = {key: value.to(device) for key, value in batch.items()}
        out = model(**batch, use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()
        labels = batch["labels"][:, 1:].contiguous()
        vocab = logits.shape[-1]
        token_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, vocab),
            labels.view(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.shape)
        for i, row in enumerate(chunk):
            mask = labels[i] != -100
            split = str(row["split"])
            totals[split] = totals.get(split, 0.0) + float(token_loss[i][mask].sum().item())
            counts[split] = counts.get(split, 0) + int(mask.sum().item())
    return {f"{split}_loss": totals[split] / max(counts[split], 1) for split in sorted(totals)}


@torch.no_grad()
def patched_losses(donor, recipient, batches, patch_points, position: str, device: str) -> dict[str, float]:
    cache, donor_handles, recipient_handles = install_hooks(donor, recipient, patch_points, position)
    totals = {}
    counts = {}
    try:
        for chunk, batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            cache.clear()
            _ = donor(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            out = recipient(**batch, use_cache=False)
            logits = out.logits[:, :-1, :].contiguous()
            labels = batch["labels"][:, 1:].contiguous()
            vocab = logits.shape[-1]
            token_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, vocab),
                labels.view(-1),
                ignore_index=-100,
                reduction="none",
            ).view(labels.shape)
            for i, row in enumerate(chunk):
                mask = labels[i] != -100
                split = str(row["split"])
                totals[split] = totals.get(split, 0.0) + float(token_loss[i][mask].sum().item())
                counts[split] = counts.get(split, 0) + int(mask.sum().item())
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)
    return {f"{split}_loss": totals[split] / max(counts[split], 1) for split in sorted(totals)}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def gap_closed(recipient_loss: float, donor_loss: float, patched_loss: float) -> float:
    denom = recipient_loss - donor_loss
    if abs(denom) < 1e-9:
        return float("nan")
    return (recipient_loss - patched_loss) / denom


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Qwen2.5-1.5B Activation Patch Target-Loss Check",
        "",
        "Teacher-forced activation patching from base into abliterated on standard refusal completions.",
        "",
        "| variant | harmful loss | benign loss | harmful gap closed | benign gap closed | specificity |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | {row['harmful_refusal_loss']:.3f} | "
            f"{row['benign_refusal_loss']:.3f} | {row['harmful_gap_closed']:.3f} | "
            f"{row['benign_gap_closed']:.3f} | {row['specificity']:.3f} |"
        )
    lines.extend(
        [
            "",
            "## Current Decision",
            "",
            "- This checks whether donor activations can explain the static MLP repair at the likelihood level.",
            "- Dynamic generation patching should only be attempted for ranges that close substantial harmful-refusal loss here.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--examples-per-split", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--patch-specs", default=",".join(DEFAULT_PATCH_SPECS))
    ap.add_argument("--position", choices=("all", "target"), default="all")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    patch_specs = parse_specs(args.patch_specs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    batches = make_batches(tokenizer, args.examples_per_split, args.max_length, args.batch_size)

    print("[load] base donor", flush=True)
    donor = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=torch.float16).to(args.device)
    donor.eval()
    print("[load] abliterated recipient", flush=True)
    recipient = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=torch.float16).to(args.device)
    recipient.eval()

    base_losses = losses_for_model(donor, batches, args.device)
    recipient_losses = losses_for_model(recipient, batches, args.device)

    rows = []
    for variant, losses in (("base", base_losses), ("abliterated", recipient_losses)):
        rows.append(
            {
                "variant": variant,
                "patch_spec": "",
                "position": "",
                "harmful_refusal_loss": losses["harmful_refusal_loss"],
                "benign_refusal_loss": losses["benign_refusal_loss"],
                "harmful_gap_closed": gap_closed(
                    recipient_losses["harmful_refusal_loss"],
                    base_losses["harmful_refusal_loss"],
                    losses["harmful_refusal_loss"],
                ),
                "benign_gap_closed": gap_closed(
                    recipient_losses["benign_refusal_loss"],
                    base_losses["benign_refusal_loss"],
                    losses["benign_refusal_loss"],
                ),
                "specificity": 0.0,
            }
        )

    for layers, module, spec in patch_specs:
        patch_points = tuple((layer, module) for layer in layers)
        print(f"[patch] {spec} position={args.position}", flush=True)
        losses = patched_losses(donor, recipient, batches, patch_points, args.position, args.device)
        harmful_closed = gap_closed(
            recipient_losses["harmful_refusal_loss"],
            base_losses["harmful_refusal_loss"],
            losses["harmful_refusal_loss"],
        )
        benign_closed = gap_closed(
            recipient_losses["benign_refusal_loss"],
            base_losses["benign_refusal_loss"],
            losses["benign_refusal_loss"],
        )
        rows.append(
            {
                "variant": f"activation_patch_{spec}",
                "patch_spec": spec,
                "position": args.position,
                "harmful_refusal_loss": losses["harmful_refusal_loss"],
                "benign_refusal_loss": losses["benign_refusal_loss"],
                "harmful_gap_closed": harmful_closed,
                "benign_gap_closed": benign_closed,
                "specificity": harmful_closed - benign_closed,
            }
        )

    metrics_path = args.result_dir / "qwen1_5b_activation_patch_target_losses.csv"
    summary_path = args.result_dir / "QWEN1_5B_ACTIVATION_PATCH_TARGET_LOSS_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, rows)
    write_summary(summary_path, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "models": {"donor": MODEL_IDS["base"], "recipient": MODEL_IDS["abliterated"]},
                "examples_per_split": args.examples_per_split,
                "patch_specs": [spec for _layers, _module, spec in patch_specs],
                "position": args.position,
                "outputs": {"metrics": str(metrics_path), "summary": str(summary_path)},
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
