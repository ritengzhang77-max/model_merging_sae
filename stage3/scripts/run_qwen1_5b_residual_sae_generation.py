#!/usr/bin/env python3
"""Sparse autoencoder smoke test for the Qwen2.5-1.5B residual repair.

This is deliberately a basis-validation script, not a feature-interpretation
script. It asks whether a learned sparse residual basis can reproduce the
behavioral repair that currently requires a very broad native-coordinate patch.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import re
import sys
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("HF_HOME", "/data/gavin/model_merging/hf_cache")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import torch
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
from analyze_qwen1_5b_refusal_rq1_rq2 import MODEL_IDS  # noqa: E402
from run_qwen1_5b_dynamic_activation_patch_generation import chat_prompt  # noqa: E402
from run_qwen1_5b_lowdim_activation_patch_target_loss import (  # noqa: E402
    DEFAULT_LAYERS,
    build_bases,
    collect_basis_stats,
    first_tensor,
    make_batches as make_primary_basis_batches,
    parse_ints,
    remove_hooks,
    replace_first_tensor,
)
from run_qwen1_5b_pca_residual_patch_generation import (  # noqa: E402
    parse_layer_spec,
    read_prompt_jsonl,
)
from run_qwen1_5b_second_stage_residual_pca import (  # noqa: E402
    build_residual_bases,
    collect_residual_rows,
    make_cache_hook,
    make_harmful_batches,
    residual_basis_prompts,
)
from screen_chat_merge_candidate import clean_assistant_text, score_record  # noqa: E402


RESULT_DIR = ROOT / "stage3" / "results" / "qwen1_5b_residual_sae_v0_generation"
DEFAULT_PROMPT_JSONL = ROOT / "stage3" / "data" / "qwen1_5b_residual_benchmark" / "qwen1_5b_residual_benchmark_v0.jsonl"


@dataclass(frozen=True)
class SaeConfig:
    label: str
    dict_size: int
    l1: float


class ResidualSAE(nn.Module):
    def __init__(self, hidden_size: int, dict_size: int, mean: torch.Tensor):
        super().__init__()
        self.register_buffer("mean", mean.float().clone())
        self.encoder = nn.Linear(hidden_size, dict_size)
        self.decoder = nn.Linear(dict_size, hidden_size, bias=False)
        nn.init.kaiming_uniform_(self.encoder.weight, a=math.sqrt(5))
        nn.init.zeros_(self.encoder.bias)
        nn.init.normal_(self.decoder.weight, std=0.02)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return torch.relu(self.encoder(x - self.mean))

    def decode(self, acts: torch.Tensor) -> torch.Tensor:
        return self.decoder(acts) + self.mean

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        acts = self.encode(x)
        return self.decode(acts), acts


def parse_sae_configs(raw: str) -> list[SaeConfig]:
    configs: list[SaeConfig] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        match = re.fullmatch(r"d(\d+)_l1_([0-9.eE+-]+)", item)
        if not match:
            raise ValueError(f"SAE config must look like d512_l1_0.001, got {item!r}")
        dict_size = int(match.group(1))
        l1 = float(match.group(2))
        configs.append(SaeConfig(label=item, dict_size=dict_size, l1=l1))
    if not configs:
        raise ValueError("at least one SAE config is required")
    return configs


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")


def summarize(model_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    out = {"model": model_name, "n": len(rows)}
    for split in ("harmful", "benign"):
        group = [row for row in rows if row["split"] == split]
        out[f"{split}_ok_rate"] = sum(bool(row["ok"]) for row in group) / max(len(group), 1)
        out[f"{split}_clean_generation_rate"] = sum(bool(row["clean_generation"]) for row in group) / max(len(group), 1)
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    out["harmful_attempted_refusal_rate"] = sum(bool(row.get("harmful_attempted_refusal")) for row in harmful) / max(
        len(harmful), 1
    )
    out["harmful_bad_attempt_rate"] = sum(bool(row.get("harmful_bad_refusal_attempt")) for row in harmful) / max(
        len(harmful), 1
    )
    out["harmful_unsafe_continuation_rate"] = sum(
        bool(row.get("harmful_unsafe_continuation")) for row in harmful
    ) / max(len(harmful), 1)
    out["benign_over_refusal_rate"] = sum(bool(row.get("benign_over_refusal")) for row in benign) / max(len(benign), 1)
    return out


def write_summary(path: Path, rows: list[dict[str, object]], train_rows: list[dict[str, object]]) -> None:
    lines = [
        "# Qwen2.5-1.5B Residual SAE Generation",
        "",
        "Dynamic generation check for learned sparse autoencoder reconstructions of the post-PCA64 residual.",
        "",
        "| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| `{row['model']}` | {row.get('harmful_ok_rate', 0.0):.3f} | "
            f"{row.get('harmful_attempted_refusal_rate', 0.0):.3f} | "
            f"{row.get('harmful_bad_attempt_rate', 0.0):.3f} | "
            f"{row.get('harmful_unsafe_continuation_rate', 0.0):.3f} | "
            f"{row.get('benign_ok_rate', 0.0):.3f} | {row.get('benign_over_refusal_rate', 0.0):.3f} |"
        )
    lines.extend(["", "## SAE Training Metrics", ""])
    for label in sorted({str(row["config"]) for row in train_rows}):
        subset = [row for row in train_rows if str(row["config"]) == label]
        mean_ev = sum(float(row["explained_variance"]) for row in subset) / max(len(subset), 1)
        mean_l0 = sum(float(row["l0"]) for row in subset) / max(len(subset), 1)
        mean_mse = sum(float(row["mse"]) for row in subset) / max(len(subset), 1)
        lines.append(f"- `{label}`: mean EV `{mean_ev:.3f}`, mean L0 `{mean_l0:.1f}`, mean MSE `{mean_mse:.5f}`")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def train_sae_for_layer(
    rows: torch.Tensor,
    config: SaeConfig,
    *,
    device: str,
    steps: int,
    batch_size: int,
    lr: float,
    seed: int,
) -> tuple[ResidualSAE, dict[str, object]]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    data = rows.float().to(device)
    mean = data.mean(dim=0).detach().cpu()
    model = ResidualSAE(data.shape[1], config.dict_size, mean).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    n = data.shape[0]
    for _step in range(steps):
        idx = torch.randint(0, n, (min(batch_size, n),), generator=generator).to(device)
        batch = data[idx]
        recon, acts = model(batch)
        mse = torch.mean((recon - batch) ** 2)
        l1 = acts.abs().mean()
        loss = mse + config.l1 * l1
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        recon, acts = model(data)
        mse = torch.mean((recon - data) ** 2).item()
        centered = data - data.mean(dim=0, keepdim=True)
        var = torch.mean(centered**2).item()
        ev = 1.0 - mse / max(var, 1e-12)
        l0 = (acts > 1e-6).float().sum(dim=1).mean().item()
        l1_val = acts.abs().mean().item()
    return model, {
        "config": config.label,
        "dict_size": config.dict_size,
        "l1_coeff": config.l1,
        "n_rows": int(n),
        "hidden_size": int(data.shape[1]),
        "mse": mse,
        "explained_variance": ev,
        "l0": l0,
        "mean_abs_activation": l1_val,
    }


def make_patch_hook(
    cache: dict[int, torch.Tensor],
    layer: int,
    *,
    primary_bases,
    pca_rank: int,
    patch_kind: str,
    full_layers: set[int],
    topk_bases,
    topk: int | None,
    sae_by_layer: dict[int, ResidualSAE] | None,
):
    def hook(_module, _inputs, output):
        tensor, _rest = first_tensor(output)
        donor = cache[layer].to(device=tensor.device, dtype=tensor.dtype)
        if layer in full_layers:
            return replace_first_tensor(output, donor)

        delta = donor - tensor
        primary = primary_bases["pca"][(layer, pca_rank)].to(device=tensor.device, dtype=tensor.dtype)
        coeff = torch.matmul(delta, primary)
        projected = torch.matmul(coeff, primary.T)
        patched = tensor + projected

        residual = delta - projected
        if patch_kind == "topk" and topk is not None:
            key = (layer, topk)
            if key in topk_bases["top_neuron"]:
                idx = topk_bases["top_neuron"][key].to(device=tensor.device)
                patched = patched.clone()
                patched[:, :, idx] = patched[:, :, idx] + residual[:, :, idx]
        elif patch_kind == "sae":
            if sae_by_layer is not None and layer in sae_by_layer:
                sae = sae_by_layer[layer]
                flat = residual.reshape(-1, residual.shape[-1]).float()
                recon_flat, _acts = sae(flat)
                recon = recon_flat.to(device=tensor.device, dtype=tensor.dtype).reshape_as(residual)
                patched = patched + recon
        elif patch_kind == "pca64":
            pass
        else:
            raise ValueError(patch_kind)
        return replace_first_tensor(output, patched)

    return hook


def install_patch_hooks(
    donor,
    recipient,
    layers: tuple[int, ...],
    *,
    primary_bases,
    pca_rank: int,
    patch_kind: str,
    full_layers: tuple[int, ...],
    topk_bases,
    topk: int | None,
    sae_by_layer: dict[int, ResidualSAE] | None,
):
    cache: dict[int, torch.Tensor] = {}
    full_set = set(full_layers)
    donor_handles = []
    recipient_handles = []
    for layer in layers:
        donor_handles.append(donor.model.layers[layer].mlp.register_forward_hook(make_cache_hook(cache, layer)))
        recipient_handles.append(
            recipient.model.layers[layer].mlp.register_forward_hook(
                make_patch_hook(
                    cache,
                    layer,
                    primary_bases=primary_bases,
                    pca_rank=pca_rank,
                    patch_kind=patch_kind,
                    full_layers=full_set,
                    topk_bases=topk_bases,
                    topk=topk,
                    sae_by_layer=sae_by_layer,
                )
            )
        )
    return cache, donor_handles, recipient_handles


@torch.no_grad()
def patched_generate(
    donor,
    recipient,
    tokenizer,
    user: str,
    *,
    device: str,
    max_new_tokens: int,
    layers: tuple[int, ...],
    primary_bases,
    pca_rank: int,
    patch_kind: str,
    full_layers: tuple[int, ...],
    topk_bases,
    topk: int | None,
    sae_by_layer: dict[int, ResidualSAE] | None,
) -> str:
    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    cache, donor_handles, recipient_handles = install_patch_hooks(
        donor,
        recipient,
        layers,
        primary_bases=primary_bases,
        pca_rank=pca_rank,
        patch_kind=patch_kind,
        full_layers=full_layers,
        topk_bases=topk_bases,
        topk=topk,
        sae_by_layer=sae_by_layer,
    )
    try:
        for _ in range(max_new_tokens):
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
    decoded = tokenizer.decode(input_ids[0], skip_special_tokens=False)
    return clean_assistant_text(tokenizer, prompt, decoded)


@torch.no_grad()
def patched_generate_input_ids(
    donor,
    recipient,
    tokenizer,
    user: str,
    *,
    device: str,
    max_new_tokens: int,
    layers: tuple[int, ...],
    primary_bases,
    pca_rank: int,
    full_layers: tuple[int, ...],
) -> tuple[torch.Tensor, int]:
    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    prompt_len = int(input_ids.shape[1])
    cache, donor_handles, recipient_handles = install_patch_hooks(
        donor,
        recipient,
        layers,
        primary_bases=primary_bases,
        pca_rank=pca_rank,
        patch_kind="pca64",
        full_layers=full_layers,
        topk_bases={"top_neuron": {}},
        topk=None,
        sae_by_layer=None,
    )
    try:
        for _ in range(max_new_tokens):
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
    return input_ids.detach().cpu(), prompt_len


def position_mask(attention_mask: torch.Tensor, prompt_len: int, mode: str) -> torch.Tensor:
    mask = attention_mask.bool().clone()
    seq_len = int(mask.shape[1])
    positions = torch.arange(seq_len, device=mask.device)[None, :]
    if mode == "all":
        return mask
    if mode == "prompt":
        return mask & (positions < prompt_len)
    if mode == "generated":
        return mask & (positions >= prompt_len)
    if mode == "last":
        out = torch.zeros_like(mask)
        out[:, -1:] = mask[:, -1:]
        return out
    if mode == "generated_last":
        out = torch.zeros_like(mask)
        if seq_len > prompt_len:
            out[:, -1:] = mask[:, -1:]
        return out
    raise ValueError(mode)


@torch.no_grad()
def collect_generated_residual_rows(
    donor,
    recipient,
    tokenizer,
    prompts: tuple[str, ...],
    layers: tuple[int, ...],
    residual_layers: tuple[int, ...],
    primary_bases,
    pca_rank: int,
    *,
    device: str,
    max_rows_per_layer: int,
    max_new_tokens: int,
    source: str,
    position_mode: str,
):
    if source == "generated_full":
        full_layers = residual_layers
    elif source == "generated_pca64":
        full_layers = ()
    else:
        raise ValueError(source)

    generated_sequences: list[tuple[torch.Tensor, int]] = []
    for user in prompts:
        input_ids, prompt_len = patched_generate_input_ids(
            donor,
            recipient,
            tokenizer,
            user,
            device=device,
            max_new_tokens=max_new_tokens,
            layers=layers,
            primary_bases=primary_bases,
            pca_rank=pca_rank,
            full_layers=full_layers,
        )
        generated_sequences.append((input_ids, prompt_len))

    donor_cache: dict[int, torch.Tensor] = {}
    recipient_cache: dict[int, torch.Tensor] = {}
    donor_handles = [donor.model.layers[layer].mlp.register_forward_hook(make_cache_hook(donor_cache, layer)) for layer in residual_layers]
    recipient_handles = [
        recipient.model.layers[layer].mlp.register_forward_hook(make_cache_hook(recipient_cache, layer))
        for layer in residual_layers
    ]
    rows = {layer: [] for layer in residual_layers}
    raw_energy = {layer: 0.0 for layer in residual_layers}
    n_rows = {layer: 0 for layer in residual_layers}
    try:
        for input_ids_cpu, prompt_len in generated_sequences:
            input_ids = input_ids_cpu.to(device)
            attention_mask = torch.ones_like(input_ids)
            mask = position_mask(attention_mask, prompt_len, position_mode)
            if not bool(mask.any().item()):
                continue
            donor_cache.clear()
            recipient_cache.clear()
            _ = donor(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            _ = recipient(input_ids=input_ids, attention_mask=attention_mask, use_cache=False)
            for layer in residual_layers:
                delta = donor_cache[layer].float() - recipient_cache[layer].float()
                basis = primary_bases["pca"][(layer, pca_rank)].to(device=delta.device, dtype=delta.dtype)
                coeff = torch.matmul(delta, basis)
                projected = torch.matmul(coeff, basis.T)
                residual = delta - projected
                selected = residual[mask]
                if not selected.numel():
                    continue
                raw_energy[layer] += float((selected * selected).sum().item())
                n_rows[layer] += int(selected.shape[0])
                used = sum(part.shape[0] for part in rows[layer])
                remaining = max_rows_per_layer - used
                if remaining > 0:
                    rows[layer].append(selected[:remaining].detach().cpu())
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)
    return rows, raw_energy, n_rows


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--prompt-jsonl", type=Path, default=DEFAULT_PROMPT_JSONL)
    ap.add_argument("--sae-configs", default="d512_l1_0.001,d1024_l1_0.001")
    ap.add_argument("--sae-steps", type=int, default=1200)
    ap.add_argument("--sae-batch-size", type=int, default=256)
    ap.add_argument("--sae-lr", type=float, default=1e-3)
    ap.add_argument("--basis-examples-per-split", type=int, default=8)
    ap.add_argument("--examples-per-split", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--pca-rank", type=int, default=64)
    ap.add_argument("--residual-layers", default="16-23")
    ap.add_argument("--residual-basis-mode", default="residual_targets")
    ap.add_argument("--residual-basis-mask", choices=("all", "target"), default="all")
    ap.add_argument(
        "--residual-row-source",
        choices=("teacher_forced", "generated_full", "generated_pca64"),
        default="teacher_forced",
    )
    ap.add_argument(
        "--generated-row-position",
        choices=("all", "prompt", "generated", "last", "generated_last"),
        default="generated",
    )
    ap.add_argument("--generated-basis-max-new-tokens", type=int)
    ap.add_argument("--topk-baseline", type=int, default=1344)
    ap.add_argument("--max-pca-rows-per-layer", type=int, default=512)
    ap.add_argument("--max-residual-rows-per-layer", type=int, default=512)
    ap.add_argument("--pca-oversample", type=int, default=8)
    ap.add_argument("--pca-n-iter", type=int, default=1)
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_ints(args.layers)
    residual_layers = parse_layer_spec(args.residual_layers)
    sae_configs = parse_sae_configs(args.sae_configs)

    tokenizer = AutoTokenizer.from_pretrained(MODEL_IDS["base"])
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    primary_batches = make_primary_basis_batches(tokenizer, args.basis_examples_per_split, args.max_length, args.batch_size)
    residual_prompts = residual_basis_prompts(args.residual_basis_mode, args.examples_per_split)
    residual_batches = make_harmful_batches(tokenizer, residual_prompts, args.max_length, args.batch_size)
    prompt_rows = read_prompt_jsonl(args.prompt_jsonl)

    print("[load] base donor", flush=True)
    donor = AutoModelForCausalLM.from_pretrained(MODEL_IDS["base"], torch_dtype=torch.float16).to(args.device)
    donor.eval()
    print("[load] abliterated recipient", flush=True)
    recipient = AutoModelForCausalLM.from_pretrained(MODEL_IDS["abliterated"], torch_dtype=torch.float16).to(args.device)
    recipient.eval()

    print("[basis] collect PCA64 calibration activations", flush=True)
    primary_stats = collect_basis_stats(
        donor,
        recipient,
        primary_batches,
        layers,
        device=args.device,
        max_rows_per_layer=args.max_pca_rows_per_layer,
    )
    primary_bases = build_bases(
        primary_stats,
        layers,
        (),
        (args.pca_rank,),
        (),
        random_seed=args.random_seed,
        pca_oversample=args.pca_oversample,
        pca_n_iter=args.pca_n_iter,
    )

    print("[basis] collect post-PCA64 residual rows", flush=True)
    if args.residual_row_source == "teacher_forced":
        residual_rows, _energy, _n_rows = collect_residual_rows(
            donor,
            recipient,
            residual_batches,
            residual_layers,
            primary_bases,
            args.pca_rank,
            device=args.device,
            max_rows_per_layer=args.max_residual_rows_per_layer,
            basis_mask=args.residual_basis_mask,
        )
    else:
        residual_rows, _energy, _n_rows = collect_generated_residual_rows(
            donor,
            recipient,
            tokenizer,
            residual_prompts,
            layers,
            residual_layers,
            primary_bases,
            args.pca_rank,
            device=args.device,
            max_rows_per_layer=args.max_residual_rows_per_layer,
            max_new_tokens=args.generated_basis_max_new_tokens or args.max_new_tokens,
            source=args.residual_row_source,
            position_mode=args.generated_row_position,
        )
    topk_bases, _explained = build_residual_bases(
        residual_rows,
        residual_layers,
        (64,),
        (args.topk_baseline,),
        random_seed=args.random_seed + 1009,
        pca_oversample=args.pca_oversample,
        pca_n_iter=args.pca_n_iter,
    )

    train_rows: list[dict[str, object]] = []
    sae_models_by_config: dict[str, dict[int, ResidualSAE]] = {}
    for config in sae_configs:
        print(f"[sae] train {config.label}", flush=True)
        layer_models: dict[int, ResidualSAE] = {}
        for layer in residual_layers:
            if not residual_rows[layer]:
                continue
            matrix = torch.cat(residual_rows[layer], dim=0).float()
            model, row = train_sae_for_layer(
                matrix,
                config,
                device=args.device,
                steps=args.sae_steps,
                batch_size=args.sae_batch_size,
                lr=args.sae_lr,
                seed=args.random_seed + 1000 * layer + config.dict_size,
            )
            row["layer"] = layer
            train_rows.append(row)
            layer_models[layer] = model
        sae_models_by_config[config.label] = layer_models

    variants: list[tuple[str, str, tuple[int, ...], int | None, dict[int, ResidualSAE] | None]] = [
        ("pca64", "pca64", (), None, None),
        (f"residual_topk{args.topk_baseline}_{args.residual_layers}", "topk", (), args.topk_baseline, None),
        (f"full_{args.residual_layers}", "pca64", residual_layers, None, None),
    ]
    for config in sae_configs:
        variants.append((f"residual_sae_{config.label}_{args.residual_layers}", "sae", (), None, sae_models_by_config[config.label]))

    all_records: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for model_name, patch_kind, full_layers, topk, sae_by_layer in variants:
        print(f"[eval] {model_name}", flush=True)
        records = []
        for prompt_row in prompt_rows:
            split = str(prompt_row["split"])
            user = str(prompt_row["user"])
            expected = str(prompt_row.get("expected", ""))
            text = patched_generate(
                donor,
                recipient,
                tokenizer,
                user,
                device=args.device,
                max_new_tokens=args.max_new_tokens,
                layers=layers,
                primary_bases=primary_bases,
                pca_rank=args.pca_rank,
                patch_kind=patch_kind,
                full_layers=full_layers,
                topk_bases=topk_bases,
                topk=topk,
                sae_by_layer=sae_by_layer,
            )
            scores = score_record(split, user, text, expected)
            record = {
                "model": model_name,
                **prompt_row,
                "split": split,
                "user": user,
                "expected": expected,
                "generation": text,
                **scores,
            }
            records.append(record)
            all_records.append(record)
        summary_rows.append(summarize(model_name, records))

    metrics_path = args.result_dir / "qwen1_5b_residual_sae_generation_metrics.csv"
    records_path = args.result_dir / "qwen1_5b_residual_sae_generation_records.jsonl"
    train_path = args.result_dir / "qwen1_5b_residual_sae_training_metrics.csv"
    summary_path = args.result_dir / "QWEN1_5B_RESIDUAL_SAE_GENERATION_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, summary_rows)
    write_jsonl(records_path, all_records)
    write_csv(train_path, train_rows)
    write_summary(summary_path, summary_rows, train_rows)
    manifest_path.write_text(
        json.dumps(
            {
                "models": {"donor": MODEL_IDS["base"], "recipient": MODEL_IDS["abliterated"]},
                "prompt_jsonl": str(args.prompt_jsonl),
                "sae_configs": [config.__dict__ for config in sae_configs],
                "sae_steps": args.sae_steps,
                "sae_batch_size": args.sae_batch_size,
                "sae_lr": args.sae_lr,
                "residual_basis_mode": args.residual_basis_mode,
                "residual_row_source": args.residual_row_source,
                "residual_basis_mask": args.residual_basis_mask,
                "generated_row_position": args.generated_row_position,
                "generated_basis_max_new_tokens": args.generated_basis_max_new_tokens or args.max_new_tokens,
                "residual_basis_prompts": residual_prompts,
                "layers": layers,
                "residual_layers": residual_layers,
                "pca_rank": args.pca_rank,
                "topk_baseline": args.topk_baseline,
                "outputs": {
                    "metrics": str(metrics_path),
                    "records": str(records_path),
                    "training_metrics": str(train_path),
                    "summary": str(summary_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
