#!/usr/bin/env python3
"""Feature-subset interventions for the GemmaScope MLP SAE repair target."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
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
from screen_chat_merge_candidate import (  # noqa: E402
    BENIGN_PROMPTS,
    HARMFUL_PROMPTS,
    clean_assistant_text,
    generate,
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


RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_gemmascope_mlp_sae_feature_subsets"
DEFAULT_LAYERS = tuple(range(12, 21))
DEFAULT_VARIANTS = (
    "full_decode",
    "delta_add_all",
    "delta_add_delta_abs_k256",
    "delta_add_delta_abs_k512",
    "delta_add_delta_specific_k512",
    "delta_add_random_active_k512",
    "mix_decode_delta_abs_k256",
    "mix_decode_delta_abs_k512",
    "mix_decode_delta_specific_k512",
    "mix_decode_random_active_k512",
)
PATCH_TOKEN_FILTERS = (
    "all",
    "prompt_all",
    "prompt_template",
    "contentish",
    "assistant_boundary",
    "generated",
    "assistant_boundary_or_generated",
    "contentish_or_generated",
    "prompt_template_or_generated",
    "last_token",
    "prompt_or_last",
    "none",
)


def prompt_slice(prompts, start: int, count: int):
    if start < 0:
        raise ValueError(f"prompt start must be non-negative, got {start}")
    return prompts[start : start + count]


def make_prompt_rows(tokenizer, examples_per_split: int, max_length: int, prompt_start: int):
    rows = []
    for split, prompts in (("harmful", HARMFUL_PROMPTS), ("benign", BENIGN_PROMPTS)):
        for user in prompt_slice(prompts, prompt_start, examples_per_split):
            prompt = chat_prompt(tokenizer, str(user))
            enc = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=max_length)
            rows.append({"split": split, "user": str(user), "prompt": prompt, "enc": enc})
    return rows


def collate_prompt_rows(tokenizer, rows):
    ids = [row["enc"]["input_ids"][0] for row in rows]
    masks = [row["enc"]["attention_mask"][0] for row in rows]
    max_len = max(x.shape[0] for x in ids)
    out_ids = []
    out_masks = []
    for one_ids, one_mask in zip(ids, masks):
        pad = max_len - one_ids.shape[0]
        out_ids.append(torch.cat([one_ids, torch.full((pad,), tokenizer.pad_token_id, dtype=torch.long)]))
        out_masks.append(torch.cat([one_mask, torch.zeros(pad, dtype=torch.long)]))
    return {"input_ids": torch.stack(out_ids), "attention_mask": torch.stack(out_masks)}


def make_mlp_cache_hooks(model, layers: tuple[int, ...], output_mode: str):
    cache = {layer: [] for layer in layers}
    handles = []

    def make_hook(layer):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            cache[layer].append(tensor.detach())

        return hook

    for layer in layers:
        handles.append(module_mlp(model, layer, output_mode).register_forward_hook(make_hook(layer)))
    return cache, handles


def flatten_valid(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return tensor[mask.bool()]


def token_filter_mask(tokenizer, input_ids: torch.Tensor, attention_mask: torch.Tensor, mode: str) -> torch.Tensor:
    if mode == "all":
        return attention_mask.bool()
    if mode != "contentish":
        raise ValueError(f"unknown feature token filter: {mode}")
    rows = []
    for seq_ids, seq_mask in zip(input_ids.detach().cpu().tolist(), attention_mask.detach().cpu().tolist()):
        row = []
        for token_id, keep in zip(seq_ids, seq_mask):
            if not keep:
                row.append(False)
                continue
            text = tokenizer.decode([int(token_id)], skip_special_tokens=False)
            stripped = text.strip()
            has_alnum = any(ch.isalnum() for ch in stripped)
            is_role = stripped in {"user", "model"}
            is_special = stripped.startswith("<") and stripped.endswith(">")
            row.append(bool(has_alnum and not is_role and not is_special))
        rows.append(row)
    return torch.tensor(rows, device=input_ids.device, dtype=torch.bool)


def patch_position_mask(
    tokenizer,
    input_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    mode: str,
    *,
    prompt_length: int | None,
) -> torch.Tensor:
    if mode == "all":
        return attention_mask.bool()
    if mode == "none":
        return torch.zeros_like(attention_mask, dtype=torch.bool)
    if mode == "prompt_all":
        mask = attention_mask.bool()
        if prompt_length is not None and prompt_length < mask.shape[1]:
            mask[:, prompt_length:] = False
        return mask
    if mode == "contentish":
        mask = token_filter_mask(tokenizer, input_ids, attention_mask, "contentish")
        if prompt_length is not None and prompt_length < mask.shape[1]:
            mask[:, prompt_length:] = False
        return mask
    if mode == "prompt_template":
        content_mask = token_filter_mask(tokenizer, input_ids, attention_mask, "contentish")
        mask = attention_mask.bool() & ~content_mask
        if prompt_length is not None and prompt_length < mask.shape[1]:
            mask[:, prompt_length:] = False
        return mask
    if mode == "generated":
        mask = attention_mask.bool()
        if prompt_length is not None:
            mask[:, :prompt_length] = False
        return mask
    if mode == "assistant_boundary_or_generated":
        return patch_position_mask(tokenizer, input_ids, attention_mask, "assistant_boundary", prompt_length=prompt_length) | patch_position_mask(
            tokenizer, input_ids, attention_mask, "generated", prompt_length=prompt_length
        )
    if mode == "contentish_or_generated":
        return patch_position_mask(tokenizer, input_ids, attention_mask, "contentish", prompt_length=prompt_length) | patch_position_mask(
            tokenizer, input_ids, attention_mask, "generated", prompt_length=prompt_length
        )
    if mode == "prompt_template_or_generated":
        return patch_position_mask(tokenizer, input_ids, attention_mask, "prompt_template", prompt_length=prompt_length) | patch_position_mask(
            tokenizer, input_ids, attention_mask, "generated", prompt_length=prompt_length
        )
    if mode in {"last_token", "prompt_or_last"}:
        mask = torch.zeros_like(attention_mask, dtype=torch.bool)
        for row_idx, seq_mask in enumerate(attention_mask.detach().cpu().tolist()):
            valid_len = sum(int(x) for x in seq_mask)
            if valid_len > 0:
                mask[row_idx, valid_len - 1] = True
        if mode == "prompt_or_last":
            prompt_mask = patch_position_mask(tokenizer, input_ids, attention_mask, "prompt_all", prompt_length=prompt_length)
            mask = mask | prompt_mask
        return mask
    if mode != "assistant_boundary":
        raise ValueError(f"unknown patch token filter: {mode}")

    rows = []
    boundary_tokens = {"", "<end_of_turn>", "<start_of_turn>", "model"}
    for seq_ids, seq_mask in zip(input_ids.detach().cpu().tolist(), attention_mask.detach().cpu().tolist()):
        valid_len = sum(int(x) for x in seq_mask)
        limit = valid_len if prompt_length is None else min(prompt_length, valid_len)
        decoded = [tokenizer.decode([int(token_id)], skip_special_tokens=False).strip() for token_id in seq_ids[:limit]]
        starts = [idx for idx, text in enumerate(decoded) if text == "<start_of_turn>"]
        # The last start marker in the prompt is the assistant generation boundary.
        start = starts[-1] if starts else max(0, limit - 4)
        left = max(0, start - 2)
        row = []
        for pos, keep in enumerate(seq_mask):
            text = decoded[pos] if pos < limit else ""
            row.append(bool(keep and left <= pos < limit and text in boundary_tokens))
        rows.append(row)
    return torch.tensor(rows, device=input_ids.device, dtype=torch.bool)


@torch.no_grad()
def collect_feature_stats(
    donor,
    recipient,
    tokenizer,
    saes,
    *,
    layers: tuple[int, ...],
    examples_per_split: int,
    batch_size: int,
    max_length: int,
    prompt_start: int,
    device: str,
    output_mode: str,
    feature_token_filter: str = "all",
):
    rows = make_prompt_rows(tokenizer, examples_per_split, max_length, prompt_start)
    donor_cache, donor_handles = make_mlp_cache_hooks(donor, layers, output_mode)
    recipient_cache, recipient_handles = make_mlp_cache_hooks(recipient, layers, output_mode)
    stats = {
        layer: {
            "harm_delta_abs": torch.zeros(saes[layer].W_enc.shape[1], dtype=torch.float32),
            "benign_delta_abs": torch.zeros(saes[layer].W_enc.shape[1], dtype=torch.float32),
            "harm_donor": torch.zeros(saes[layer].W_enc.shape[1], dtype=torch.float32),
            "active": torch.zeros(saes[layer].W_enc.shape[1], dtype=torch.float32),
            "harm_tokens": 0,
            "benign_tokens": 0,
        }
        for layer in layers
    }
    try:
        for start in range(0, len(rows), batch_size):
            chunk = rows[start : start + batch_size]
            batch = collate_prompt_rows(tokenizer, chunk)
            batch = {key: value.to(device) for key, value in batch.items()}
            for layer in layers:
                donor_cache[layer].clear()
                recipient_cache[layer].clear()
            _ = donor(**batch, use_cache=False)
            _ = recipient(**batch, use_cache=False)
            token_mask = token_filter_mask(tokenizer, batch["input_ids"], batch["attention_mask"], feature_token_filter)
            for split in ("harmful", "benign"):
                idx = torch.tensor([row["split"] == split for row in chunk], device=device)
                if not bool(idx.any().item()):
                    continue
                mask = token_mask[idx].bool()
                for layer in layers:
                    donor_out = flatten_valid(donor_cache[layer][0][idx], mask)
                    recipient_out = flatten_valid(recipient_cache[layer][0][idx], mask)
                    donor_f = saes[layer].encode(donor_out).float().cpu()
                    recipient_f = saes[layer].encode(recipient_out).float().cpu()
                    delta_abs = (donor_f - recipient_f).abs().sum(dim=0)
                    row = stats[layer]
                    row["active"] += ((donor_f > 0) | (recipient_f > 0)).float().sum(dim=0)
                    if split == "harmful":
                        row["harm_delta_abs"] += delta_abs
                        row["harm_donor"] += donor_f.sum(dim=0)
                        row["harm_tokens"] += int(donor_f.shape[0])
                    else:
                        row["benign_delta_abs"] += delta_abs
                        row["benign_tokens"] += int(donor_f.shape[0])
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)
    return stats


def parse_variant(raw: str) -> dict[str, object]:
    if raw in {"full_decode", "recipient_recon", "delta_add_all"}:
        return {"label": raw, "mode": raw, "selector": "all", "k": "all"}
    range_match = re.fullmatch(r"(delta_add|mix_decode)_delta_abs_rank(\d+)_(\d+)", raw)
    if range_match:
        rank_start = int(range_match.group(2))
        rank_end = int(range_match.group(3))
        if rank_start < 1 or rank_end < rank_start:
            raise ValueError(f"invalid 1-indexed rank range in variant: {raw}")
        return {
            "label": raw,
            "mode": range_match.group(1),
            "selector": "delta_abs_rank_range",
            "k": f"{rank_start}-{rank_end}",
            "rank_start": rank_start,
            "rank_end": rank_end,
        }
    plus_match = re.fullmatch(r"(delta_add|mix_decode)_delta_abs_k(\d+)_plus_l(\d+)_rank(\d+)_(\d+)", raw)
    if plus_match:
        prefix_k = int(plus_match.group(2))
        plus_layer = int(plus_match.group(3))
        rank_start = int(plus_match.group(4))
        rank_end = int(plus_match.group(5))
        if prefix_k < 1 or rank_start < 1 or rank_end < rank_start:
            raise ValueError(f"invalid prefix/rank range in variant: {raw}")
        return {
            "label": raw,
            "mode": plus_match.group(1),
            "selector": "delta_abs_prefix_plus_layer_range",
            "k": f"{prefix_k}+L{plus_layer}:{rank_start}-{rank_end}",
            "prefix_k": prefix_k,
            "plus_layer": plus_layer,
            "rank_start": rank_start,
            "rank_end": rank_end,
        }
    plus_feature_match = re.fullmatch(r"(delta_add|mix_decode)_delta_abs_k(\d+)_plus_l(\d+)_f(\d+)", raw)
    if plus_feature_match:
        prefix_k = int(plus_feature_match.group(2))
        plus_layer = int(plus_feature_match.group(3))
        feature_id = int(plus_feature_match.group(4))
        if prefix_k < 1 or feature_id < 0:
            raise ValueError(f"invalid prefix/feature ID in variant: {raw}")
        return {
            "label": raw,
            "mode": plus_feature_match.group(1),
            "selector": "delta_abs_prefix_plus_layer_feature",
            "k": f"{prefix_k}+L{plus_layer}:f{feature_id}",
            "prefix_k": prefix_k,
            "plus_layer": plus_layer,
            "feature_id": feature_id,
        }
    plus_feature_range_match = re.fullmatch(
        r"(delta_add|mix_decode)_delta_abs_k(\d+)_plus_l(\d+)_f(\d+)_plus_l(\d+)_rank(\d+)_(\d+)",
        raw,
    )
    if plus_feature_range_match:
        prefix_k = int(plus_feature_range_match.group(2))
        feature_layer = int(plus_feature_range_match.group(3))
        feature_id = int(plus_feature_range_match.group(4))
        range_layer = int(plus_feature_range_match.group(5))
        rank_start = int(plus_feature_range_match.group(6))
        rank_end = int(plus_feature_range_match.group(7))
        if prefix_k < 1 or feature_id < 0 or rank_start < 1 or rank_end < rank_start:
            raise ValueError(f"invalid prefix/feature/rank range in variant: {raw}")
        return {
            "label": raw,
            "mode": plus_feature_range_match.group(1),
            "selector": "delta_abs_prefix_plus_feature_plus_layer_range",
            "k": f"{prefix_k}+L{feature_layer}:f{feature_id}+L{range_layer}:{rank_start}-{rank_end}",
            "prefix_k": prefix_k,
            "feature_layer": feature_layer,
            "feature_id": feature_id,
            "range_layer": range_layer,
            "rank_start": rank_start,
            "rank_end": rank_end,
        }
    minus_feature_range_match = re.fullmatch(
        r"(delta_add|mix_decode)_delta_abs_k(\d+)_plus_l(\d+)_f(\d+)_minus_l(\d+)_rank(\d+)_(\d+)",
        raw,
    )
    if minus_feature_range_match:
        prefix_k = int(minus_feature_range_match.group(2))
        feature_layer = int(minus_feature_range_match.group(3))
        feature_id = int(minus_feature_range_match.group(4))
        range_layer = int(minus_feature_range_match.group(5))
        rank_start = int(minus_feature_range_match.group(6))
        rank_end = int(minus_feature_range_match.group(7))
        if prefix_k < 1 or feature_id < 0 or rank_start < 1 or rank_end < rank_start or rank_end > prefix_k:
            raise ValueError(f"invalid prefix/feature/rank range in variant: {raw}")
        return {
            "label": raw,
            "mode": minus_feature_range_match.group(1),
            "selector": "delta_abs_prefix_plus_feature_minus_layer_range",
            "k": f"{prefix_k}+L{feature_layer}:f{feature_id}-L{range_layer}:{rank_start}-{rank_end}",
            "prefix_k": prefix_k,
            "feature_layer": feature_layer,
            "feature_id": feature_id,
            "range_layer": range_layer,
            "rank_start": rank_start,
            "rank_end": rank_end,
        }
    minus_match = re.fullmatch(r"(delta_add|mix_decode)_delta_abs_k(\d+)_minus_l(\d+)_rank(\d+)_(\d+)", raw)
    if minus_match:
        prefix_k = int(minus_match.group(2))
        minus_layer = int(minus_match.group(3))
        rank_start = int(minus_match.group(4))
        rank_end = int(minus_match.group(5))
        if prefix_k < 1 or rank_start < 1 or rank_end < rank_start or rank_end > prefix_k:
            raise ValueError(f"invalid prefix/rank range in variant: {raw}")
        return {
            "label": raw,
            "mode": minus_match.group(1),
            "selector": "delta_abs_prefix_minus_layer_range",
            "k": f"{prefix_k}-L{minus_layer}:{rank_start}-{rank_end}",
            "prefix_k": prefix_k,
            "minus_layer": minus_layer,
            "rank_start": rank_start,
            "rank_end": rank_end,
        }
    minus_feature_match = re.fullmatch(r"(delta_add|mix_decode)_delta_abs_k(\d+)_minus_l(\d+)_f(\d+)", raw)
    if minus_feature_match:
        prefix_k = int(minus_feature_match.group(2))
        minus_layer = int(minus_feature_match.group(3))
        feature_id = int(minus_feature_match.group(4))
        if prefix_k < 1 or feature_id < 0:
            raise ValueError(f"invalid prefix/feature ID in variant: {raw}")
        return {
            "label": raw,
            "mode": minus_feature_match.group(1),
            "selector": "delta_abs_prefix_minus_layer_feature",
            "k": f"{prefix_k}-L{minus_layer}:f{feature_id}",
            "prefix_k": prefix_k,
            "minus_layer": minus_layer,
            "feature_id": feature_id,
        }
    match = re.fullmatch(r"(delta_add|mix_decode)_(delta_abs|delta_specific|donor_harm|random_active)_k(\d+)", raw)
    if not match:
        raise ValueError(f"unknown variant: {raw}")
    return {"label": raw, "mode": match.group(1), "selector": match.group(2), "k": int(match.group(3))}


def select_indices(stats, layers: tuple[int, ...], variants: list[dict[str, object]], seed: int, device: str):
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    selected: dict[str, dict[int, torch.Tensor | None]] = {}
    counts = []
    for variant in variants:
        label = str(variant["label"])
        selected[label] = {}
        selector = str(variant["selector"])
        k = variant["k"]
        for layer in layers:
            if selector == "all":
                selected[label][layer] = None
                counts.append({"variant": label, "layer": layer, "selector": selector, "k": "all", "selected_features": "all"})
                continue
            row = stats[layer]
            if selector == "delta_abs_rank_range":
                score = row["harm_delta_abs"]
                rank_start = int(variant["rank_start"])
                rank_end = min(int(variant["rank_end"]), int(score.numel()))
                if float(score.abs().sum().item()) == 0.0 or rank_start > rank_end:
                    idx = torch.empty(0, dtype=torch.long)
                else:
                    ranked = torch.topk(score, rank_end).indices
                    idx = ranked[rank_start - 1 : rank_end]
                selected[label][layer] = idx.to(device=device, dtype=torch.long)
                counts.append(
                    {
                        "variant": label,
                        "layer": layer,
                        "selector": selector,
                        "k": str(variant["k"]),
                        "selected_features": int(idx.numel()),
                        "score_sum": float(score[idx].sum().item()) if idx.numel() else 0.0,
                    }
                )
                continue
            if selector == "delta_abs_prefix_plus_layer_range":
                score = row["harm_delta_abs"]
                prefix_k = min(int(variant["prefix_k"]), int(score.numel()))
                rank_start = int(variant["rank_start"])
                rank_end = min(int(variant["rank_end"]), int(score.numel()))
                if float(score.abs().sum().item()) == 0.0:
                    idx = torch.empty(0, dtype=torch.long)
                else:
                    ranked = torch.topk(score, max(prefix_k, rank_end)).indices
                    parts = [ranked[:prefix_k]]
                    if layer == int(variant["plus_layer"]) and rank_start <= rank_end:
                        parts.append(ranked[rank_start - 1 : rank_end])
                    idx = torch.unique(torch.cat(parts), sorted=False)
                selected[label][layer] = idx.to(device=device, dtype=torch.long)
                counts.append(
                    {
                        "variant": label,
                        "layer": layer,
                        "selector": selector,
                        "k": str(variant["k"]),
                        "selected_features": int(idx.numel()),
                        "score_sum": float(score[idx].sum().item()) if idx.numel() else 0.0,
                    }
                )
                continue
            if selector == "delta_abs_prefix_plus_layer_feature":
                score = row["harm_delta_abs"]
                prefix_k = min(int(variant["prefix_k"]), int(score.numel()))
                feature_id = int(variant["feature_id"])
                if feature_id >= int(score.numel()):
                    raise ValueError(f"feature ID {feature_id} is outside layer {layer} feature dimension {score.numel()}")
                if float(score.abs().sum().item()) == 0.0:
                    idx = torch.empty(0, dtype=torch.long)
                else:
                    ranked = torch.topk(score, prefix_k).indices
                    parts = [ranked]
                    if layer == int(variant["plus_layer"]):
                        parts.append(torch.tensor([feature_id], dtype=torch.long))
                    idx = torch.unique(torch.cat(parts), sorted=False)
                selected[label][layer] = idx.to(device=device, dtype=torch.long)
                counts.append(
                    {
                        "variant": label,
                        "layer": layer,
                        "selector": selector,
                        "k": str(variant["k"]),
                        "selected_features": int(idx.numel()),
                        "score_sum": float(score[idx].sum().item()) if idx.numel() else 0.0,
                    }
                )
                continue
            if selector == "delta_abs_prefix_plus_feature_plus_layer_range":
                score = row["harm_delta_abs"]
                prefix_k = min(int(variant["prefix_k"]), int(score.numel()))
                feature_id = int(variant["feature_id"])
                rank_start = int(variant["rank_start"])
                rank_end = min(int(variant["rank_end"]), int(score.numel()))
                if feature_id >= int(score.numel()):
                    raise ValueError(f"feature ID {feature_id} is outside layer {layer} feature dimension {score.numel()}")
                if float(score.abs().sum().item()) == 0.0:
                    idx = torch.empty(0, dtype=torch.long)
                else:
                    ranked = torch.topk(score, max(prefix_k, rank_end)).indices
                    parts = [ranked[:prefix_k]]
                    if layer == int(variant["feature_layer"]):
                        parts.append(torch.tensor([feature_id], dtype=torch.long))
                    if layer == int(variant["range_layer"]) and rank_start <= rank_end:
                        parts.append(ranked[rank_start - 1 : rank_end])
                    idx = torch.unique(torch.cat(parts), sorted=False)
                selected[label][layer] = idx.to(device=device, dtype=torch.long)
                counts.append(
                    {
                        "variant": label,
                        "layer": layer,
                        "selector": selector,
                        "k": str(variant["k"]),
                        "selected_features": int(idx.numel()),
                        "score_sum": float(score[idx].sum().item()) if idx.numel() else 0.0,
                    }
                )
                continue
            if selector == "delta_abs_prefix_plus_feature_minus_layer_range":
                score = row["harm_delta_abs"]
                prefix_k = min(int(variant["prefix_k"]), int(score.numel()))
                feature_id = int(variant["feature_id"])
                rank_start = int(variant["rank_start"])
                rank_end = min(int(variant["rank_end"]), prefix_k)
                if feature_id >= int(score.numel()):
                    raise ValueError(f"feature ID {feature_id} is outside layer {layer} feature dimension {score.numel()}")
                if float(score.abs().sum().item()) == 0.0:
                    idx = torch.empty(0, dtype=torch.long)
                else:
                    ranked = torch.topk(score, prefix_k).indices
                    if layer == int(variant["range_layer"]) and rank_start <= rank_end:
                        parts = [ranked[: rank_start - 1], ranked[rank_end:]]
                    else:
                        parts = [ranked]
                    if layer == int(variant["feature_layer"]):
                        parts.append(torch.tensor([feature_id], dtype=torch.long))
                    idx = torch.unique(torch.cat(parts), sorted=False)
                selected[label][layer] = idx.to(device=device, dtype=torch.long)
                counts.append(
                    {
                        "variant": label,
                        "layer": layer,
                        "selector": selector,
                        "k": str(variant["k"]),
                        "selected_features": int(idx.numel()),
                        "score_sum": float(score[idx].sum().item()) if idx.numel() else 0.0,
                    }
                )
                continue
            if selector == "delta_abs_prefix_minus_layer_range":
                score = row["harm_delta_abs"]
                prefix_k = min(int(variant["prefix_k"]), int(score.numel()))
                rank_start = int(variant["rank_start"])
                rank_end = min(int(variant["rank_end"]), prefix_k)
                if float(score.abs().sum().item()) == 0.0:
                    idx = torch.empty(0, dtype=torch.long)
                else:
                    ranked = torch.topk(score, prefix_k).indices
                    if layer == int(variant["minus_layer"]) and rank_start <= rank_end:
                        idx = torch.cat([ranked[: rank_start - 1], ranked[rank_end:]])
                    else:
                        idx = ranked
                selected[label][layer] = idx.to(device=device, dtype=torch.long)
                counts.append(
                    {
                        "variant": label,
                        "layer": layer,
                        "selector": selector,
                        "k": str(variant["k"]),
                        "selected_features": int(idx.numel()),
                        "score_sum": float(score[idx].sum().item()) if idx.numel() else 0.0,
                    }
                )
                continue
            if selector == "delta_abs_prefix_minus_layer_feature":
                score = row["harm_delta_abs"]
                prefix_k = min(int(variant["prefix_k"]), int(score.numel()))
                feature_id = int(variant["feature_id"])
                if feature_id >= int(score.numel()):
                    raise ValueError(f"feature ID {feature_id} is outside layer {layer} feature dimension {score.numel()}")
                if float(score.abs().sum().item()) == 0.0:
                    idx = torch.empty(0, dtype=torch.long)
                else:
                    ranked = torch.topk(score, prefix_k).indices
                    if layer == int(variant["minus_layer"]):
                        idx = ranked[ranked != feature_id]
                    else:
                        idx = ranked
                selected[label][layer] = idx.to(device=device, dtype=torch.long)
                counts.append(
                    {
                        "variant": label,
                        "layer": layer,
                        "selector": selector,
                        "k": str(variant["k"]),
                        "selected_features": int(idx.numel()),
                        "score_sum": float(score[idx].sum().item()) if idx.numel() else 0.0,
                    }
                )
                continue
            kk = min(int(k), int(row["active"].numel()))
            if selector == "delta_abs":
                score = row["harm_delta_abs"]
                pool = torch.arange(score.numel())
            elif selector == "delta_specific":
                harm = row["harm_delta_abs"] / max(int(row["harm_tokens"]), 1)
                benign = row["benign_delta_abs"] / max(int(row["benign_tokens"]), 1)
                score = torch.clamp(harm - benign, min=0.0)
                pool = torch.arange(score.numel())
            elif selector == "donor_harm":
                score = row["harm_donor"] / max(int(row["harm_tokens"]), 1)
                pool = torch.arange(score.numel())
            elif selector == "random_active":
                pool = torch.nonzero(row["active"] > 0, as_tuple=False).flatten()
                if pool.numel() == 0:
                    pool = torch.arange(row["active"].numel())
                perm = torch.randperm(pool.numel(), generator=generator)
                idx = pool[perm[: min(kk, pool.numel())]]
                selected[label][layer] = idx.to(device=device, dtype=torch.long)
                counts.append(
                    {
                        "variant": label,
                        "layer": layer,
                        "selector": selector,
                        "k": kk,
                        "selected_features": int(idx.numel()),
                    }
                )
                continue
            else:
                raise ValueError(selector)
            if float(score.abs().sum().item()) == 0.0:
                idx = torch.empty(0, dtype=torch.long)
            else:
                idx = torch.topk(score, kk).indices
            selected[label][layer] = idx.to(device=device, dtype=torch.long)
            counts.append(
                {
                    "variant": label,
                    "layer": layer,
                    "selector": selector,
                    "k": kk,
                    "selected_features": int(idx.numel()),
                    "score_sum": float(score[idx].sum().item()) if idx.numel() else 0.0,
                }
            )
    return selected, counts


def apply_feature_patch(sae, recipient_out, donor_out, variant, selected_idx):
    donor_f = sae.encode(donor_out)
    if variant["mode"] == "full_decode":
        return sae.decode(donor_f)
    recipient_f = sae.encode(recipient_out)
    if variant["mode"] == "recipient_recon":
        return sae.decode(recipient_f)
    if variant["mode"] == "delta_add_all":
        delta = donor_f - recipient_f
        return recipient_out + delta @ sae.W_dec
    if selected_idx is None:
        selected_idx = torch.arange(donor_f.shape[-1], device=donor_f.device)
    if selected_idx.numel() == 0:
        return recipient_out
    donor_sel = donor_f.index_select(-1, selected_idx)
    recipient_sel = recipient_f.index_select(-1, selected_idx)
    if variant["mode"] == "delta_add":
        dec = (donor_sel - recipient_sel) @ sae.W_dec.index_select(0, selected_idx)
        return recipient_out + dec
    if variant["mode"] == "mix_decode":
        mixed = recipient_f.clone()
        mixed[..., selected_idx] = donor_sel
        return sae.decode(mixed)
    raise ValueError(str(variant["mode"]))


@torch.no_grad()
def feature_patch_generate(
    donor,
    recipient,
    tokenizer,
    saes,
    user: str,
    *,
    layers: tuple[int, ...],
    variant,
    selected,
    device: str,
    max_new_tokens: int,
    output_mode: str,
    patch_token_filter: str,
) -> str:
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
            current_patch_mask = patch_position_mask(
                tokenizer,
                input_ids,
                attention_mask,
                patch_token_filter,
                prompt_length=prompt_length,
            )
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
    return {
        "model": model_name,
        "n": len(rows),
        "harmful_ok_rate": sum(bool(row["ok"]) for row in harmful) / max(len(harmful), 1),
        "harmful_clean_generation_rate": sum(bool(row["clean_generation"]) for row in harmful) / max(len(harmful), 1),
        "harmful_attempted_refusal_rate": sum(bool(row.get("harmful_attempted_refusal")) for row in harmful) / max(len(harmful), 1),
        "harmful_bad_attempt_rate": sum(bool(row.get("harmful_bad_refusal_attempt")) for row in harmful) / max(len(harmful), 1),
        "harmful_unsafe_continuation_rate": sum(bool(row.get("harmful_unsafe_continuation")) for row in harmful) / max(len(harmful), 1),
        "benign_ok_rate": sum(bool(row["ok"]) for row in benign) / max(len(benign), 1),
        "benign_clean_generation_rate": sum(bool(row["clean_generation"]) for row in benign) / max(len(benign), 1),
        "benign_over_refusal_rate": sum(bool(row.get("benign_over_refusal")) for row in benign) / max(len(benign), 1),
    }


@torch.no_grad()
def run_generation(
    donor,
    recipient,
    tokenizer,
    saes,
    selected,
    variants,
    *,
    layers,
    examples_per_split,
    prompt_start,
    max_new_tokens,
    device,
    output_mode,
    skip_baselines,
    patch_token_filter="all",
):
    prompts = [("harmful", x) for x in prompt_slice(HARMFUL_PROMPTS, prompt_start, examples_per_split)] + [
        ("benign", x) for x in prompt_slice(BENIGN_PROMPTS, prompt_start, examples_per_split)
    ]
    by_model: dict[str, list[dict[str, object]]] = {}
    if not skip_baselines:
        for model_name, model in (("base", donor), ("abliterated", recipient)):
            rows = []
            for split, user in prompts:
                text = generate(model, tokenizer, str(user), device=device, max_new_tokens=max_new_tokens)
                record = {"model": model_name, "split": split, "prompt": str(user), "text": text}
                record.update(score_record(split, str(user), text))
                rows.append(record)
            by_model[model_name] = rows
    for variant in variants:
        label = str(variant["label"])
        model_name = f"feature_subset_{label}"
        rows = []
        print(f"[eval] {model_name}", flush=True)
        for split, user in prompts:
            text = feature_patch_generate(
                donor,
                recipient,
                tokenizer,
                saes,
                str(user),
                layers=layers,
                variant=variant,
                selected=selected,
                device=device,
                max_new_tokens=max_new_tokens,
                output_mode=output_mode,
                patch_token_filter=patch_token_filter,
            )
            record = {
                "model": model_name,
                "split": split,
                "prompt": str(user),
                "text": text,
                "patch_token_filter": patch_token_filter,
            }
            record.update(score_record(split, str(user), text))
            rows.append(record)
        by_model[model_name] = rows
    metrics = []
    for name, rows in by_model.items():
        summary = summarize_generation(name, rows)
        summary["patch_token_filter"] = patch_token_filter
        metrics.append(summary)
    return metrics, [row for rows in by_model.values() for row in rows]


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


def write_summary(
    path: Path,
    metrics: list[dict[str, object]],
    feature_counts,
    layers,
    *,
    basis_start: int,
    basis_examples_per_split: int,
    eval_start: int,
    examples_per_split: int,
    feature_token_filter: str,
    patch_token_filter: str,
) -> None:
    lines = [
        "# Gemma-2-2B GemmaScope MLP SAE Feature-Subset Patch",
        "",
        f"Layers: `{','.join(str(x) for x in layers)}`.",
        f"Feature-selection prompts: `{basis_start}:{basis_start + basis_examples_per_split}` per split.",
        f"Feature-selection token filter: `{feature_token_filter}`.",
        f"Patch token filter: `{patch_token_filter}`.",
        f"Evaluation prompts: `{eval_start}:{eval_start + examples_per_split}` per split.",
        "",
        "## Generation",
        "",
        "| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in metrics:
        lines.append(
            f"| `{row['model']}` | {row.get('harmful_ok_rate', 0.0):.3f} | "
            f"{row.get('harmful_attempted_refusal_rate', 0.0):.3f} | "
            f"{row.get('harmful_bad_attempt_rate', 0.0):.3f} | "
            f"{row.get('harmful_unsafe_continuation_rate', 0.0):.3f} | "
            f"{row.get('benign_ok_rate', 0.0):.3f} | {row.get('benign_over_refusal_rate', 0.0):.3f} |"
        )
    lines.extend(
        [
            "",
            "## Feature Counts",
            "",
            "| variant | per-layer selected feature range | total selected across layers |",
            "|---|---:|---:|",
        ]
    )
    by_variant: dict[str, list[int]] = {}
    for row in feature_counts:
        if row["selected_features"] == "all":
            continue
        by_variant.setdefault(str(row["variant"]), []).append(int(row["selected_features"]))
    for variant, vals in sorted(by_variant.items()):
        lines.append(f"| `{variant}` | {min(vals)}-{max(vals)} | {sum(vals)} |")
    lines.extend(
        [
            "",
            "## Decision Rule",
            "",
            "- A feature subset is interesting only if it restores harmful refusal, preserves benign helpfulness, and beats matched random active-feature controls.",
            "- Passing full decoded reconstruction is not enough for feature-level interpretability; this run tests whether selected SAE coordinates carry the repair.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--sae-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--basis-start", type=int, default=0)
    ap.add_argument("--basis-examples-per-split", type=int, default=12)
    ap.add_argument("--eval-start", type=int, default=0)
    ap.add_argument("--examples-per-split", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--variants", default=",".join(DEFAULT_VARIANTS))
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--feature-token-filter", choices=("all", "contentish"), default="all")
    ap.add_argument("--patch-token-filter", choices=PATCH_TOKEN_FILTERS, default="all")
    ap.add_argument("--random-seed", type=int, default=0)
    ap.add_argument("--skip-baselines", action="store_true")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_ints(args.layers)
    variants = [parse_variant(item.strip()) for item in args.variants.split(",") if item.strip()]
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
    selected, feature_counts = select_indices(stats, layers, variants, args.random_seed, args.device)

    print("[generation] running feature-subset patches", flush=True)
    metrics, records = run_generation(
        donor,
        recipient,
        tokenizer,
        saes,
        selected,
        variants,
        layers=layers,
        examples_per_split=args.examples_per_split,
        prompt_start=args.eval_start,
        max_new_tokens=args.max_new_tokens,
        device=args.device,
        output_mode=args.output_mode,
        skip_baselines=args.skip_baselines,
        patch_token_filter=args.patch_token_filter,
    )

    metrics_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_subset_metrics.csv"
    records_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_subset_records.jsonl"
    feature_counts_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_feature_subset_counts.csv"
    summary_path = args.result_dir / "GEMMA2_2B_GEMMASCOPE_MLP_SAE_FEATURE_SUBSET_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"
    write_csv(metrics_path, metrics)
    write_jsonl(records_path, records)
    write_csv(feature_counts_path, feature_counts)
    write_summary(
        summary_path,
        metrics,
        feature_counts,
        layers,
        basis_start=args.basis_start,
        basis_examples_per_split=args.basis_examples_per_split,
        eval_start=args.eval_start,
        examples_per_split=args.examples_per_split,
        feature_token_filter=args.feature_token_filter,
        patch_token_filter=args.patch_token_filter,
    )
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "layers": list(layers),
                "output_mode": args.output_mode,
                "feature_token_filter": args.feature_token_filter,
                "patch_token_filter": args.patch_token_filter,
                "basis_start": args.basis_start,
                "basis_examples_per_split": args.basis_examples_per_split,
                "eval_start": args.eval_start,
                "examples_per_split": args.examples_per_split,
                "variants": [str(v["label"]) for v in variants],
                "random_seed": args.random_seed,
                "outputs": {
                    "metrics": str(metrics_path),
                    "records": str(records_path),
                    "feature_counts": str(feature_counts_path),
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
