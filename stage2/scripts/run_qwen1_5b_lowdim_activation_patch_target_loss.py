#!/usr/bin/env python3
"""Low-dimensional activation patch baselines for Qwen2.5-1.5B refusal repair."""

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
import torch.nn.functional as F
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


RESULT_DIR = ROOT / "stage2" / "results" / "qwen1_5b_lowdim_activation_patch_target_loss"
DEFAULT_LAYERS = tuple(range(12, 25))


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


def module_for(model, layer: int):
    return model.model.layers[layer].mlp


def first_tensor(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def replace_first_tensor(output, tensor):
    _first, rest = first_tensor(output)
    if rest is None:
        return tensor
    return (tensor, *rest)


def make_cache_hook(cache: dict[int, torch.Tensor], layer: int):
    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        cache[layer] = tensor.detach()

    return hook


def remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


@torch.no_grad()
def collect_basis_stats(donor, recipient, batches, layers: tuple[int, ...], *, device: str, max_rows_per_layer: int):
    donor_cache: dict[int, torch.Tensor] = {}
    recipient_cache: dict[int, torch.Tensor] = {}
    donor_handles = [module_for(donor, layer).register_forward_hook(make_cache_hook(donor_cache, layer)) for layer in layers]
    recipient_handles = [
        module_for(recipient, layer).register_forward_hook(make_cache_hook(recipient_cache, layer)) for layer in layers
    ]

    stats = {
        layer: {
            "abs_delta_sum": None,
            "delta_sum": None,
            "donor_harm_sum": None,
            "donor_benign_sum": None,
            "n_delta": 0,
            "n_harm": 0,
            "n_benign": 0,
            "delta_rows": [],
        }
        for layer in layers
    }
    try:
        for chunk, batch in batches:
            batch = {key: value.to(device) for key, value in batch.items()}
            donor_cache.clear()
            recipient_cache.clear()
            _ = donor(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            _ = recipient(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"], use_cache=False)
            mask = batch["attention_mask"].bool()
            harmful_idx = torch.tensor([row["split"] == "harmful_refusal" for row in chunk], device=device, dtype=torch.bool)
            benign_idx = torch.tensor([row["split"] == "benign_refusal" for row in chunk], device=device, dtype=torch.bool)
            for layer in layers:
                donor_out = donor_cache[layer].float()
                recipient_out = recipient_cache[layer].float()
                delta = donor_out - recipient_out
                harmful_mask = mask & harmful_idx[:, None]
                benign_mask = mask & benign_idx[:, None]
                harm_delta = delta[harmful_mask]
                donor_harm = donor_out[harmful_mask]
                donor_benign = donor_out[benign_mask]
                row = stats[layer]
                if harm_delta.numel():
                    if row["abs_delta_sum"] is None:
                        hidden = harm_delta.shape[-1]
                        row["abs_delta_sum"] = torch.zeros(hidden, dtype=torch.float32)
                        row["delta_sum"] = torch.zeros(hidden, dtype=torch.float32)
                    row["abs_delta_sum"] += harm_delta.abs().sum(dim=0).cpu()
                    row["delta_sum"] += harm_delta.sum(dim=0).cpu()
                    row["n_delta"] += int(harm_delta.shape[0])
                    remaining = max_rows_per_layer - sum(part.shape[0] for part in row["delta_rows"])
                    if remaining > 0:
                        row["delta_rows"].append(harm_delta[:remaining].detach().cpu())
                if donor_harm.numel():
                    if row["donor_harm_sum"] is None:
                        row["donor_harm_sum"] = torch.zeros(donor_harm.shape[-1], dtype=torch.float32)
                    row["donor_harm_sum"] += donor_harm.sum(dim=0).cpu()
                    row["n_harm"] += int(donor_harm.shape[0])
                if donor_benign.numel():
                    if row["donor_benign_sum"] is None:
                        row["donor_benign_sum"] = torch.zeros(donor_benign.shape[-1], dtype=torch.float32)
                    row["donor_benign_sum"] += donor_benign.sum(dim=0).cpu()
                    row["n_benign"] += int(donor_benign.shape[0])
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)
    return stats


def orthonormal_cols(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.ndim == 1:
        matrix = matrix[:, None]
    if matrix.numel() == 0:
        return matrix
    q, _r = torch.linalg.qr(matrix.float(), mode="reduced")
    return q


def randomized_pca_cols(mat: torch.Tensor, rank: int, *, oversample: int, n_iter: int, generator) -> torch.Tensor:
    q = min(rank + oversample, mat.shape[0], mat.shape[1])
    omega = torch.randn(mat.shape[1], q, generator=generator, dtype=mat.dtype)
    y = mat @ omega
    for _ in range(n_iter):
        y = mat @ (mat.T @ y)
    q_mat, _r = torch.linalg.qr(y, mode="reduced")
    sketch = q_mat.T @ mat
    _u, _s, vh = torch.linalg.svd(sketch, full_matrices=False)
    return vh[:rank].T.contiguous()


def build_bases(
    stats,
    layers: tuple[int, ...],
    neuron_ks: tuple[int, ...],
    pca_rs: tuple[int, ...],
    random_rs: tuple[int, ...],
    *,
    random_seed: int,
    pca_oversample: int,
    pca_n_iter: int,
):
    bases = {"full": {}, "top_neuron": {}, "mean_delta": {}, "refusal_dir": {}, "pca": {}, "random": {}}
    generator = torch.Generator(device="cpu")
    generator.manual_seed(random_seed)
    for layer in layers:
        row = stats[layer]
        abs_delta = row["abs_delta_sum"]
        delta_sum = row["delta_sum"]
        if abs_delta is None or delta_sum is None or row["n_delta"] == 0:
            continue
        hidden = abs_delta.numel()
        for k in neuron_ks:
            kk = min(k, abs_delta.numel())
            bases["top_neuron"][(layer, k)] = torch.topk(abs_delta, kk).indices
        for rank in random_rs:
            rr = min(rank, hidden)
            bases["random"][(layer, rank)] = orthonormal_cols(
                torch.randn(hidden, rr, generator=generator, dtype=torch.float32)
            )
        mean_delta = delta_sum / max(int(row["n_delta"]), 1)
        if float(mean_delta.norm().item()) > 0:
            bases["mean_delta"][layer] = F.normalize(mean_delta, dim=0)
        if row["donor_harm_sum"] is not None and row["donor_benign_sum"] is not None:
            harm_mean = row["donor_harm_sum"] / max(int(row["n_harm"]), 1)
            benign_mean = row["donor_benign_sum"] / max(int(row["n_benign"]), 1)
            refusal_dir = harm_mean - benign_mean
            if float(refusal_dir.norm().item()) > 0:
                bases["refusal_dir"][layer] = F.normalize(refusal_dir, dim=0)
        if pca_rs and row["delta_rows"]:
            mat = torch.cat(row["delta_rows"], dim=0).float()
            mat = mat - mat.mean(dim=0, keepdim=True)
            if mat.shape[0] > 1:
                max_rank = min(max(pca_rs), mat.shape[0], mat.shape[1])
                vh_t = randomized_pca_cols(
                    mat,
                    max_rank,
                    oversample=pca_oversample,
                    n_iter=pca_n_iter,
                    generator=generator,
                )
                for rank in pca_rs:
                    rr = min(rank, vh_t.shape[1])
                    bases["pca"][(layer, rank)] = vh_t[:, :rr].contiguous()
    return bases


def make_donor_hook(cache: dict[int, torch.Tensor], layer: int):
    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        cache[layer] = tensor.detach()

    return hook


def make_patch_hook(
    cache: dict[int, torch.Tensor],
    layer: int,
    *,
    variant_kind: str,
    variant_param,
    bases,
):
    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        donor = cache[layer].to(device=tensor.device, dtype=tensor.dtype)
        delta = donor - tensor
        if variant_kind == "full":
            patched = donor
        elif variant_kind == "top_neuron":
            idx = bases["top_neuron"][(layer, variant_param)].to(device=tensor.device)
            patched = tensor.clone()
            patched[:, :, idx] = donor[:, :, idx]
        elif variant_kind in {"mean_delta", "refusal_dir"}:
            direction = bases[variant_kind][layer].to(device=tensor.device, dtype=tensor.dtype)
            coeff = torch.einsum("bsh,h->bs", delta, direction)
            patched = tensor + coeff[:, :, None] * direction[None, None, :]
        elif variant_kind in {"pca", "random"}:
            basis = bases[variant_kind][(layer, variant_param)].to(device=tensor.device, dtype=tensor.dtype)
            coeff = torch.matmul(delta, basis)
            patched = tensor + torch.matmul(coeff, basis.T)
        else:
            raise ValueError(variant_kind)
        return replace_first_tensor(output, patched)

    return hook


def install_patch_hooks(donor, recipient, layers: tuple[int, ...], variant_kind: str, variant_param, bases):
    cache: dict[int, torch.Tensor] = {}
    donor_handles = []
    recipient_handles = []
    active_layers = []
    for layer in layers:
        has_basis = (
            variant_kind == "full"
            or (variant_kind == "top_neuron" and (layer, variant_param) in bases["top_neuron"])
            or (variant_kind in {"mean_delta", "refusal_dir"} and layer in bases[variant_kind])
            or (variant_kind in {"pca", "random"} and (layer, variant_param) in bases[variant_kind])
        )
        if not has_basis:
            continue
        active_layers.append(layer)
        donor_handles.append(module_for(donor, layer).register_forward_hook(make_donor_hook(cache, layer)))
        recipient_handles.append(
            module_for(recipient, layer).register_forward_hook(
                make_patch_hook(cache, layer, variant_kind=variant_kind, variant_param=variant_param, bases=bases)
            )
        )
    return cache, donor_handles, recipient_handles, active_layers


@torch.no_grad()
def losses_for_model(model, batches, device: str) -> dict[str, float]:
    totals = {}
    counts = {}
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
def patched_losses(donor, recipient, batches, layers, variant_kind: str, variant_param, bases, device: str):
    cache, donor_handles, recipient_handles, active_layers = install_patch_hooks(
        donor,
        recipient,
        layers,
        variant_kind,
        variant_param,
        bases,
    )
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
    return {f"{split}_loss": totals[split] / max(counts[split], 1) for split in sorted(totals)}, active_layers


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


def write_summary(path: Path, rows: list[dict[str, object]]) -> None:
    lines = [
        "# Qwen2.5-1.5B Low-Dimensional Activation Patch Baselines",
        "",
        "Teacher-forced refusal-target loss for lower-dimensional approximations to the successful `12-24:mlp` activation patch.",
        "",
        "| variant | dim per layer | harmful gap closed | benign gap closed | specificity | harmful loss | benign loss |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['variant']}` | {row['dim_per_layer']} | {row['harmful_gap_closed']:.3f} | "
            f"{row['benign_gap_closed']:.3f} | {row['specificity']:.3f} | "
            f"{row['harmful_refusal_loss']:.3f} | {row['benign_refusal_loss']:.3f} |"
        )
    compressed_rows = [row for row in rows if row["variant"] not in {"base", "abliterated", "full_12_24_mlp"}]
    best = max(compressed_rows, key=lambda r: r["harmful_gap_closed"])
    lines.extend(
        [
            "",
            "## Current Decision",
            "",
            f"- Best compressed baseline by harmful gap closure: `{best['variant']}` with `{best['harmful_gap_closed']:.3f}` harmful gap closed.",
            "- Compare these baselines to the full MLP patch before starting SAE/transcoder training.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x) for x in raw.split(",") if x)


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--examples-per-split", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--neuron-ks", default="16,64,256")
    ap.add_argument("--pca-ranks", default="1,4,16,64")
    ap.add_argument("--random-ranks", default="")
    ap.add_argument("--max-pca-rows-per-layer", type=int, default=2048)
    ap.add_argument("--pca-oversample", type=int, default=8)
    ap.add_argument("--pca-n-iter", type=int, default=1)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_ints(args.layers)
    neuron_ks = parse_ints(args.neuron_ks)
    pca_rs = parse_ints(args.pca_ranks)
    random_rs = parse_ints(args.random_ranks)

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

    print("[basis] collect activations", flush=True)
    stats = collect_basis_stats(
        donor,
        recipient,
        batches,
        layers,
        device=args.device,
        max_rows_per_layer=args.max_pca_rows_per_layer,
    )
    bases = build_bases(
        stats,
        layers,
        neuron_ks,
        pca_rs,
        random_rs,
        random_seed=args.random_seed,
        pca_oversample=args.pca_oversample,
        pca_n_iter=args.pca_n_iter,
    )

    base_losses = losses_for_model(donor, batches, args.device)
    recipient_losses = losses_for_model(recipient, batches, args.device)
    rows = []

    def add_row(variant: str, losses: dict[str, float], dim_per_layer: int | str, active_layers: list[int] | tuple[int, ...]):
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
                "variant": variant,
                "dim_per_layer": dim_per_layer,
                "n_layers": len(active_layers),
                "layers": "+".join(str(x) for x in active_layers),
                "harmful_refusal_loss": losses["harmful_refusal_loss"],
                "benign_refusal_loss": losses["benign_refusal_loss"],
                "harmful_gap_closed": harmful_closed,
                "benign_gap_closed": benign_closed,
                "specificity": harmful_closed - benign_closed,
            }
        )

    add_row("base", base_losses, "full", layers)
    add_row("abliterated", recipient_losses, "0", layers)

    variants: list[tuple[str, object, str, int | str]] = [("full", None, "full_12_24_mlp", "full")]
    variants.extend(("top_neuron", k, f"top_neuron_k{k}", k) for k in neuron_ks)
    variants.extend(("pca", r, f"pca_rank{r}", r) for r in pca_rs)
    variants.extend(("random", r, f"random_rank{r}", r) for r in random_rs)
    variants.extend(
        [
            ("mean_delta", None, "mean_delta_rank1", 1),
            ("refusal_dir", None, "refusal_direction_rank1", 1),
        ]
    )

    for kind, param, label, dim in variants:
        print(f"[eval] {label}", flush=True)
        losses, active_layers = patched_losses(donor, recipient, batches, layers, kind, param, bases, args.device)
        add_row(label, losses, dim, active_layers)

    rows = sorted(rows, key=lambda row: (-float(row["harmful_gap_closed"]), str(row["variant"])))
    metrics_path = args.result_dir / "qwen1_5b_lowdim_activation_patch_target_losses.csv"
    summary_path = args.result_dir / "QWEN1_5B_LOWDIM_ACTIVATION_PATCH_TARGET_LOSS_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, rows)
    write_summary(summary_path, rows)
    manifest_path.write_text(
        json.dumps(
            {
                "models": {"donor": MODEL_IDS["base"], "recipient": MODEL_IDS["abliterated"]},
                "examples_per_split": args.examples_per_split,
                "layers": layers,
                "neuron_ks": neuron_ks,
                "pca_ranks": pca_rs,
                "random_ranks": random_rs,
                "random_seed": args.random_seed,
                "pca_oversample": args.pca_oversample,
                "pca_n_iter": args.pca_n_iter,
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
