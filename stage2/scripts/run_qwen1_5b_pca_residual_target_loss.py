#!/usr/bin/env python3
"""Fast target-loss diagnostic for the residual missing from PCA64 patches."""

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
from run_qwen1_5b_lowdim_activation_patch_generation import (  # noqa: E402
    HELDOUT_BENIGN_PROMPTS,
    HELDOUT_HARMFUL_PROMPTS,
)
from run_qwen1_5b_lowdim_activation_patch_target_loss import (  # noqa: E402
    DEFAULT_LAYERS,
    build_bases,
    collect_basis_stats,
    first_tensor,
    make_batches as make_basis_batches,
    module_for,
    parse_ints,
    remove_hooks,
    replace_first_tensor,
)
from run_qwen1_5b_pca_residual_patch_generation import (  # noqa: E402
    HELDOUT_FAILURE_BENIGN_PROMPTS,
    HELDOUT_FAILURE_HARMFUL_PROMPTS,
    STRESS_BENIGN_PROMPTS,
    STRESS_HARMFUL_PROMPTS,
    STRESS_PERMISSION_HARMFUL_PROMPTS,
    label_for,
    parse_layer_spec,
)


RESULT_DIR = ROOT / "stage2" / "results" / "qwen1_5b_pca_residual_target_loss"


def prompts_for_mode(mode: str, n: int) -> list[dict[str, str]]:
    if mode == "train":
        harmful = HARMFUL_PROMPTS
        benign = BENIGN_PROMPTS
    elif mode == "heldout_failures":
        harmful = HELDOUT_FAILURE_HARMFUL_PROMPTS
        benign = HELDOUT_FAILURE_BENIGN_PROMPTS
    elif mode == "heldout_failures_harmful_only":
        harmful = HELDOUT_FAILURE_HARMFUL_PROMPTS
        benign = ()
    elif mode == "heldout_all":
        harmful = HELDOUT_HARMFUL_PROMPTS
        benign = HELDOUT_BENIGN_PROMPTS
    elif mode == "stress":
        harmful = STRESS_HARMFUL_PROMPTS
        benign = STRESS_BENIGN_PROMPTS
    elif mode == "stress_harmful_only":
        harmful = STRESS_HARMFUL_PROMPTS
        benign = ()
    elif mode == "stress_permission_harmful_only":
        harmful = STRESS_PERMISSION_HARMFUL_PROMPTS
        benign = ()
    else:
        raise ValueError(mode)
    rows = []
    rows.extend({"split": "harmful_refusal", "user": prompt} for prompt in harmful[:n])
    rows.extend({"split": "benign_refusal", "user": prompt} for prompt in benign[:n])
    return rows


def make_eval_batches(tokenizer, mode: str, n: int, max_length: int, batch_size: int):
    rows = prompts_for_mode(mode, n)
    examples = [
        {
            "split": row["split"],
            "user": row["user"],
            **labeled_example(tokenizer, str(row["user"]), REFUSAL_TARGET, max_length),
        }
        for row in rows
    ]
    batches = []
    for start in range(0, len(examples), batch_size):
        chunk = examples[start : start + batch_size]
        batches.append((chunk, collate_labeled(tokenizer, chunk)))
    return batches


def make_donor_hook(cache: dict[int, torch.Tensor], layer: int):
    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        cache[layer] = tensor.detach()

    return hook


def make_residual_hook(cache: dict[int, torch.Tensor], layer: int, *, full_layers: set[int], bases, pca_rank: int):
    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        donor = cache[layer].to(device=tensor.device, dtype=tensor.dtype)
        if layer in full_layers:
            patched = donor
        else:
            delta = donor - tensor
            basis = bases["pca"][(layer, pca_rank)].to(device=tensor.device, dtype=tensor.dtype)
            coeff = torch.matmul(delta, basis)
            patched = tensor + torch.matmul(coeff, basis.T)
        return replace_first_tensor(output, patched)

    return hook


def install_residual_hooks(donor, recipient, layers: tuple[int, ...], full_layers: tuple[int, ...], bases, pca_rank: int):
    cache: dict[int, torch.Tensor] = {}
    full_set = set(full_layers)
    donor_handles = []
    recipient_handles = []
    for layer in layers:
        donor_handles.append(module_for(donor, layer).register_forward_hook(make_donor_hook(cache, layer)))
        recipient_handles.append(
            module_for(recipient, layer).register_forward_hook(
                make_residual_hook(cache, layer, full_layers=full_set, bases=bases, pca_rank=pca_rank)
            )
        )
    return cache, donor_handles, recipient_handles


@torch.no_grad()
def losses_for_model(model, batches, device: str) -> dict[str, float]:
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    for chunk, batch in batches:
        batch = {key: value.to(device) for key, value in batch.items()}
        out = model(**batch, use_cache=False)
        logits = out.logits[:, :-1, :].contiguous()
        labels = batch["labels"][:, 1:].contiguous()
        token_loss = torch.nn.functional.cross_entropy(
            logits.view(-1, logits.shape[-1]),
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
def residual_losses(donor, recipient, batches, layers, full_layers, bases, pca_rank: int, device: str):
    cache, donor_handles, recipient_handles = install_residual_hooks(
        donor,
        recipient,
        layers,
        full_layers,
        bases,
        pca_rank,
    )
    totals: dict[str, float] = {}
    counts: dict[str, int] = {}
    try:
        for chunk, batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            cache.clear()
            _ = donor(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            out = recipient(**batch, use_cache=False)
            logits = out.logits[:, :-1, :].contiguous()
            labels = batch["labels"][:, 1:].contiguous()
            token_loss = torch.nn.functional.cross_entropy(
                logits.view(-1, logits.shape[-1]),
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


def gap_closed(recipient_loss: float, donor_loss: float, patched_loss: float) -> float:
    denom = recipient_loss - donor_loss
    if abs(denom) < 1e-9:
        return float("nan")
    return (recipient_loss - patched_loss) / denom


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_summary(path: Path, rows: list[dict[str, object]], prompt_mode: str) -> None:
    lines = [
        "# Qwen2.5-1.5B PCA64 Residual Target-Loss Diagnostic",
        "",
        f"Prompt mode: `{prompt_mode}`",
        "",
        "Teacher-forced refusal-target loss with PCA64 on all layers and selected layers upgraded to full donor MLP activations.",
        "",
        "| variant | harmful loss | harmful gap closed vs ablated | gap closed vs PCA64 | benign loss | benign gap closed |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | {row.get('harmful_refusal_loss', 0.0):.3f} | "
            f"{row.get('harmful_gap_closed', 0.0):.3f} | {row.get('harmful_gap_closed_vs_pca64', 0.0):.3f} | "
            f"{row.get('benign_refusal_loss', 0.0):.3f} | {row.get('benign_gap_closed', 0.0):.3f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument(
        "--prompt-mode",
        choices=(
            "train",
            "heldout_failures",
            "heldout_failures_harmful_only",
            "heldout_all",
            "stress",
            "stress_harmful_only",
            "stress_permission_harmful_only",
        ),
        default="heldout_failures",
    )
    ap.add_argument("--examples-per-split", type=int, default=12)
    ap.add_argument("--basis-examples-per-split", type=int, default=8)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--pca-rank", type=int, default=64)
    ap.add_argument("--full-layer-specs", default="none,20,21,22,23,24,20-21,22-24,20-22,23-24,20-24,12-24")
    ap.add_argument("--max-pca-rows-per-layer", type=int, default=512)
    ap.add_argument("--pca-oversample", type=int, default=8)
    ap.add_argument("--pca-n-iter", type=int, default=1)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_ints(args.layers)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    basis_batches = make_basis_batches(tokenizer, args.basis_examples_per_split, args.max_length, args.batch_size)
    eval_batches = make_eval_batches(tokenizer, args.prompt_mode, args.examples_per_split, args.max_length, args.batch_size)

    print("[load] base donor", flush=True)
    donor = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=torch.float16).to(args.device)
    donor.eval()
    print("[load] abliterated recipient", flush=True)
    recipient = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=torch.float16).to(args.device)
    recipient.eval()

    print("[basis] collect activations", flush=True)
    stats = collect_basis_stats(
        donor,
        recipient,
        basis_batches,
        layers,
        device=args.device,
        max_rows_per_layer=args.max_pca_rows_per_layer,
    )
    bases = build_bases(
        stats,
        layers,
        (),
        (args.pca_rank,),
        (),
        random_seed=args.random_seed,
        pca_oversample=args.pca_oversample,
        pca_n_iter=args.pca_n_iter,
    )

    print("[eval] base", flush=True)
    base_losses = losses_for_model(donor, eval_batches, args.device)
    print("[eval] abliterated", flush=True)
    recipient_losses = losses_for_model(recipient, eval_batches, args.device)

    raw_specs = [item.strip() for item in args.full_layer_specs.split(",") if item.strip()]
    variants: list[tuple[str, tuple[int, ...]]] = []
    for raw in raw_specs:
        full_layers = () if raw == "none" else parse_layer_spec(raw)
        variants.append((label_for(full_layers), full_layers))

    pca_losses: dict[str, float] | None = None
    rows: list[dict[str, object]] = []

    def add_row(label: str, full_layers: tuple[int, ...] | str, losses: dict[str, float]) -> None:
        nonlocal pca_losses
        harmful = losses.get("harmful_refusal_loss", float("nan"))
        benign = losses.get("benign_refusal_loss", float("nan"))
        harmful_closed = gap_closed(
            recipient_losses.get("harmful_refusal_loss", float("nan")),
            base_losses.get("harmful_refusal_loss", float("nan")),
            harmful,
        )
        benign_closed = gap_closed(
            recipient_losses.get("benign_refusal_loss", float("nan")),
            base_losses.get("benign_refusal_loss", float("nan")),
            benign,
        )
        if pca_losses is None:
            vs_pca = 0.0
        else:
            pca_harm = pca_losses.get("harmful_refusal_loss", float("nan"))
            full_harm = base_losses.get("harmful_refusal_loss", float("nan"))
            denom = pca_harm - full_harm
            vs_pca = float("nan") if abs(denom) < 1e-9 else (pca_harm - harmful) / denom
        rows.append(
            {
                "variant": label,
                "full_layers": full_layers if isinstance(full_layers, str) else "+".join(str(x) for x in full_layers),
                "harmful_refusal_loss": harmful,
                "benign_refusal_loss": benign,
                "harmful_gap_closed": harmful_closed,
                "benign_gap_closed": benign_closed,
                "harmful_gap_closed_vs_pca64": vs_pca,
            }
        )

    add_row("base", "base", base_losses)
    add_row("abliterated", "abliterated", recipient_losses)
    for label, full_layers in variants:
        print(f"[eval] {label}", flush=True)
        losses = residual_losses(donor, recipient, eval_batches, layers, full_layers, bases, args.pca_rank, args.device)
        if not full_layers:
            pca_losses = losses
        add_row(label, full_layers, losses)

    metrics_path = args.result_dir / "qwen1_5b_pca_residual_target_losses.csv"
    summary_path = args.result_dir / "QWEN1_5B_PCA_RESIDUAL_TARGET_LOSS_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, rows)
    write_summary(summary_path, rows, args.prompt_mode)
    manifest_path.write_text(
        json.dumps(
            {
                "models": {"donor": MODEL_IDS["base"], "recipient": MODEL_IDS["abliterated"]},
                "prompt_mode": args.prompt_mode,
                "examples_per_split": args.examples_per_split,
                "basis_examples_per_split": args.basis_examples_per_split,
                "layers": layers,
                "pca_rank": args.pca_rank,
                "full_layer_specs": args.full_layer_specs,
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
