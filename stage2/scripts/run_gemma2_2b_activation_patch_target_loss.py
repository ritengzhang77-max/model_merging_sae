#!/usr/bin/env python3
"""Teacher-forced activation patching for the Gemma-2-2B abliterated pair."""

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
from screen_chat_merge_candidate import BENIGN_PROMPTS, HARMFUL_PROMPTS  # noqa: E402


MODEL_IDS = {
    "base": "google/gemma-2-2b-it",
    "abliterated": "IlyaGusev/gemma-2-2b-it-abliterated",
}
REFUSAL_TARGET = "I cannot help with that request."
RESULT_DIR = ROOT / "stage2" / "results" / "gemma2_2b_activation_patch_target_loss"
DEFAULT_PATCH_SPECS = (
    "16:mlp",
    "20:mlp",
    "25:mlp",
    "16+17+18+19+20:mlp",
    "20+21+22+23+24+25:mlp",
    "12+13+14+15+16+17+18+19+20:mlp",
    "16+17+18+19+20+21+22+23+24+25:mlp",
)


def chat_prompt(tokenizer, user: str) -> str:
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def parse_specs(raw: str) -> list[tuple[tuple[int, ...], str, str]]:
    out = []
    for item in [x.strip() for x in raw.split(",") if x.strip()]:
        layer_text, module = item.split(":", 1)
        layers = tuple(int(part) for part in layer_text.split("+") if part)
        if module not in {"mlp", "post_ff", "attn", "block"}:
            raise ValueError(f"unknown module for activation patch: {module}")
        out.append((layers, module, item))
    return out


def module_for(model, layer: int, kind: str):
    block = model.model.layers[layer]
    if kind == "block":
        return block
    if kind == "mlp":
        return block.mlp
    if kind == "post_ff":
        return block.post_feedforward_layernorm
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


def labeled_example(tokenizer, user: str, target: str, max_length: int) -> dict[str, list[int]]:
    prompt = chat_prompt(tokenizer, user)
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    suffix = target + (tokenizer.eos_token or "")
    target_ids = tokenizer(suffix, add_special_tokens=False).input_ids
    input_ids = (prompt_ids + target_ids)[-max_length:]
    n_target = min(len(target_ids), len(input_ids))
    labels = [-100] * (len(input_ids) - n_target) + input_ids[-n_target:]
    attention_mask = [1] * len(input_ids)
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention_mask}


def collate_labeled(tokenizer, examples: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
    max_len = max(len(ex["input_ids"]) for ex in examples)
    input_ids = []
    labels = []
    attention_mask = []
    for ex in examples:
        pad = max_len - len(ex["input_ids"])
        input_ids.append(ex["input_ids"] + [tokenizer.pad_token_id] * pad)
        labels.append(ex["labels"] + [-100] * pad)
        attention_mask.append(ex["attention_mask"] + [0] * pad)
    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
    }


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
        batches.append((chunk, collate_labeled(tokenizer, chunk)))
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
        "# Gemma-2-2B Activation Patch Target-Loss Check",
        "",
        "Teacher-forced activation patching from Gemma base into abliterated on a fixed refusal completion.",
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
            "- This checks whether base activations can restore the refusal target at the likelihood level.",
            "- Dynamic generation patching should be attempted for ranges that close substantial harmful-refusal loss.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--examples-per-split", type=int, default=8)
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

    metrics_path = args.result_dir / "gemma2_2b_activation_patch_target_losses.csv"
    summary_path = args.result_dir / "GEMMA2_2B_ACTIVATION_PATCH_TARGET_LOSS_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, rows)
    write_summary(summary_path, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "target_refusal": REFUSAL_TARGET,
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
