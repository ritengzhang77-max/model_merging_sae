#!/usr/bin/env python3
"""Validate GemmaScope transcoders on the Gemma-2-2B abliterated repair target."""

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

import numpy as np
import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download, list_repo_files
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
sys.path.insert(0, str(ROOT / "stage2" / "scripts"))
from run_gemma2_2b_activation_patch_target_loss import MODEL_IDS  # noqa: E402
from screen_chat_merge_candidate import (  # noqa: E402
    BENIGN_PROMPTS,
    HARMFUL_PROMPTS,
    clean_assistant_text,
    generate,
    score_record,
)


TRANSCODER_REPO = "google/gemma-scope-2b-pt-transcoders"
RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_gemmascope_transcoder_validation"
DEFAULT_LAYERS = (12, 16, 20)
WIDTH = "16k"


class JumpReluTranscoder(nn.Module):
    def __init__(self, params: dict[str, np.ndarray], device: str, dtype: torch.dtype):
        super().__init__()
        self.W_enc = nn.Parameter(torch.from_numpy(params["W_enc"]).to(device=device, dtype=dtype), requires_grad=False)
        self.W_dec = nn.Parameter(torch.from_numpy(params["W_dec"]).to(device=device, dtype=dtype), requires_grad=False)
        self.b_enc = nn.Parameter(torch.from_numpy(params["b_enc"]).to(device=device, dtype=dtype), requires_grad=False)
        self.b_dec = nn.Parameter(torch.from_numpy(params["b_dec"]).to(device=device, dtype=dtype), requires_grad=False)
        self.threshold = nn.Parameter(
            torch.from_numpy(params["threshold"]).to(device=device, dtype=dtype), requires_grad=False
        )

    @torch.no_grad()
    def encode(self, x: torch.Tensor) -> torch.Tensor:
        pre = x.to(device=self.W_enc.device, dtype=self.W_enc.dtype) @ self.W_enc + self.b_enc
        return torch.where(pre > self.threshold, pre, torch.zeros_like(pre))

    @torch.no_grad()
    def decode(self, f: torch.Tensor) -> torch.Tensor:
        return f @ self.W_dec + self.b_dec

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        f = self.encode(x)
        return f, self.decode(f)


def parse_ints(raw: str) -> tuple[int, ...]:
    return tuple(int(x.strip()) for x in raw.split(",") if x.strip())


def chat_prompt(tokenizer, user: str) -> str:
    if not getattr(tokenizer, "chat_template", None):
        return user
    return tokenizer.apply_chat_template(
        [{"role": "user", "content": user}],
        tokenize=False,
        add_generation_prompt=True,
    )


def module_mlp_in(model, layer: int):
    return model.model.layers[layer].pre_feedforward_layernorm


def module_mlp_out(model, layer: int, output_mode: str):
    if output_mode == "post_ff_norm":
        return model.model.layers[layer].post_feedforward_layernorm
    if output_mode == "raw_mlp":
        return model.model.layers[layer].mlp
    raise ValueError(output_mode)


def first_tensor(output):
    if isinstance(output, tuple):
        return output[0], output[1:]
    return output, None


def replace_first_tensor(output, tensor):
    _first, rest = first_tensor(output)
    if rest is None:
        return tensor
    return (tensor, *rest)


def remove_hooks(handles) -> None:
    for handle in handles:
        handle.remove()


def select_transcoder_file(layer: int, l0_target: int, repo: str = TRANSCODER_REPO) -> str:
    pattern = re.compile(rf"^layer_{layer}/width_{WIDTH}/average_l0_(\d+)/params\.npz$")
    candidates = []
    for filename in list_repo_files(repo, repo_type="model"):
        match = pattern.match(filename)
        if match:
            l0 = int(match.group(1))
            candidates.append((abs(l0 - l0_target), l0, filename))
    if not candidates:
        raise ValueError(f"no GemmaScope transcoder file found for layer {layer}")
    return sorted(candidates)[0][2]


def load_transcoder(filename: str, *, device: str, dtype: torch.dtype, cache_dir: str | None) -> JumpReluTranscoder:
    path = hf_hub_download(TRANSCODER_REPO, filename, cache_dir=cache_dir)
    params = np.load(path)
    model = JumpReluTranscoder(params, device=device, dtype=dtype)
    model.eval()
    return model


def make_prompt_rows(tokenizer, examples_per_split: int, max_length: int):
    rows = []
    for split, prompts in (("harmful", HARMFUL_PROMPTS[:examples_per_split]), ("benign", BENIGN_PROMPTS[:examples_per_split])):
        for user in prompts:
            text = chat_prompt(tokenizer, str(user))
            enc = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length)
            rows.append({"split": split, "user": str(user), "prompt": text, "enc": enc})
    return rows


def collate_prompt_rows(tokenizer, rows):
    input_ids = [row["enc"]["input_ids"][0] for row in rows]
    attention_mask = [row["enc"]["attention_mask"][0] for row in rows]
    max_len = max(x.shape[0] for x in input_ids)
    padded_ids = []
    padded_mask = []
    for ids, mask in zip(input_ids, attention_mask):
        pad = max_len - ids.shape[0]
        padded_ids.append(torch.cat([ids, torch.full((pad,), tokenizer.pad_token_id, dtype=torch.long)]))
        padded_mask.append(torch.cat([mask, torch.zeros(pad, dtype=torch.long)]))
    return {
        "input_ids": torch.stack(padded_ids, dim=0),
        "attention_mask": torch.stack(padded_mask, dim=0),
    }


def folded_norm_input(module, inputs, output, input_mode: str) -> torch.Tensor:
    if input_mode == "hf_post_norm":
        return output.detach()
    if input_mode == "folded_pre_norm":
        raw = inputs[0]
        if hasattr(module, "_norm"):
            normed = module._norm(raw.float())
            return normed.to(dtype=raw.dtype).detach()
        weight = (1.0 + module.weight.float()).to(device=output.device, dtype=output.dtype)
        return (output / weight).detach()
    raise ValueError(input_mode)


def make_cache_hooks(model, layers: tuple[int, ...], input_mode: str, output_mode: str):
    cache = {layer: {"mlp_in": [], "mlp_out": []} for layer in layers}
    handles = []

    def make_in_hook(layer):
        def hook(module, inputs, output):
            cache[layer]["mlp_in"].append(folded_norm_input(module, inputs, output, input_mode))

        return hook

    def make_out_hook(layer):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            cache[layer]["mlp_out"].append(tensor.detach())

        return hook

    for layer in layers:
        handles.append(module_mlp_in(model, layer).register_forward_hook(make_in_hook(layer)))
        handles.append(module_mlp_out(model, layer, output_mode).register_forward_hook(make_out_hook(layer)))
    return cache, handles


def flatten_valid(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return tensor[mask.bool()].float()


def metric_row(
    layer: int,
    split: str,
    donor_in: torch.Tensor,
    donor_out: torch.Tensor,
    recipient_in: torch.Tensor,
    recipient_out: torch.Tensor,
    mask: torch.Tensor,
    transcoder: JumpReluTranscoder,
    filename: str,
) -> dict[str, object]:
    donor_tokens = flatten_valid(donor_in, mask)
    donor_target = flatten_valid(donor_out, mask)
    recipient_tokens = flatten_valid(recipient_in, mask)
    recipient_target = flatten_valid(recipient_out, mask)

    donor_f, donor_recon = transcoder(donor_tokens)
    recipient_f, recipient_recon = transcoder(recipient_tokens)
    donor_recon = donor_recon.float().cpu()
    recipient_recon = recipient_recon.float().cpu()
    donor_target = donor_target.cpu()
    recipient_target = recipient_target.cpu()

    def stats(target: torch.Tensor, recon: torch.Tensor) -> tuple[float, float, float]:
        err = recon - target
        sse = float((err * err).sum().item())
        centered = target - target.mean(dim=0, keepdim=True)
        tss = float((centered * centered).sum().item())
        ev = 1.0 - sse / max(tss, 1e-12)
        mse = sse / max(target.numel(), 1)
        cosine = float(torch.nn.functional.cosine_similarity(recon, target, dim=-1).mean().item())
        return mse, ev, cosine

    donor_mse, donor_ev, donor_cos = stats(donor_target, donor_recon)
    recipient_mse, recipient_ev, recipient_cos = stats(recipient_target, recipient_recon)
    gap_sse = float(((recipient_target - donor_target) ** 2).sum().item())
    recon_sse = float(((donor_recon - donor_target) ** 2).sum().item())
    gap_closed = 1.0 - recon_sse / max(gap_sse, 1e-12)

    return {
        "layer": layer,
        "split": split,
        "transcoder_file": filename,
        "n_tokens": int(donor_target.shape[0]),
        "donor_recon_mse": donor_mse,
        "donor_recon_ev": donor_ev,
        "donor_recon_cosine": donor_cos,
        "recipient_recon_mse": recipient_mse,
        "recipient_recon_ev": recipient_ev,
        "recipient_recon_cosine": recipient_cos,
        "donor_feature_l0_mean": float((donor_f > 0).sum(dim=-1).float().mean().item()),
        "donor_feature_l0_median": float((donor_f > 0).sum(dim=-1).float().median().item()),
        "recipient_feature_l0_mean": float((recipient_f > 0).sum(dim=-1).float().mean().item()),
        "recipient_feature_l0_median": float((recipient_f > 0).sum(dim=-1).float().median().item()),
        "recipient_to_donor_mlp_mse": gap_sse / max(donor_target.numel(), 1),
        "recon_vs_recipient_gap_closed": gap_closed,
    }


@torch.no_grad()
def collect_reconstruction_rows(
    donor,
    recipient,
    tokenizer,
    transcoders,
    files,
    *,
    layers: tuple[int, ...],
    examples_per_split: int,
    batch_size: int,
    max_length: int,
    device: str,
    input_mode: str,
    output_mode: str,
):
    prompt_rows = make_prompt_rows(tokenizer, examples_per_split, max_length)
    out_rows = []
    donor_cache, donor_handles = make_cache_hooks(donor, layers, input_mode, output_mode)
    recipient_cache, recipient_handles = make_cache_hooks(recipient, layers, input_mode, output_mode)
    try:
        for start in range(0, len(prompt_rows), batch_size):
            chunk = prompt_rows[start : start + batch_size]
            batch = collate_prompt_rows(tokenizer, chunk)
            batch = {key: value.to(device) for key, value in batch.items()}
            for layer in layers:
                donor_cache[layer]["mlp_in"].clear()
                donor_cache[layer]["mlp_out"].clear()
                recipient_cache[layer]["mlp_in"].clear()
                recipient_cache[layer]["mlp_out"].clear()
            _ = donor(**batch, use_cache=False)
            _ = recipient(**batch, use_cache=False)
            for split in ("harmful", "benign"):
                idx = torch.tensor([row["split"] == split for row in chunk], device=device)
                if not bool(idx.any().item()):
                    continue
                mask = batch["attention_mask"][idx].bool()
                for layer in layers:
                    row = metric_row(
                        layer,
                        split,
                        donor_cache[layer]["mlp_in"][0][idx],
                        donor_cache[layer]["mlp_out"][0][idx],
                        recipient_cache[layer]["mlp_in"][0][idx],
                        recipient_cache[layer]["mlp_out"][0][idx],
                        mask,
                        transcoders[layer],
                        files[layer],
                    )
                    out_rows.append(row)
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)
    return aggregate_metric_rows(out_rows)


def aggregate_metric_rows(rows: list[dict[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["layer"]), str(row["split"])), []).append(row)
    out = []
    numeric_keys = [key for key in rows[0] if key not in {"layer", "split", "transcoder_file"}] if rows else []
    for (layer, split), group in sorted(grouped.items()):
        total_tokens = sum(int(row["n_tokens"]) for row in group)
        merged = {
            "layer": layer,
            "split": split,
            "transcoder_file": group[0]["transcoder_file"],
            "n_tokens": total_tokens,
        }
        for key in numeric_keys:
            if key == "n_tokens":
                continue
            merged[key] = sum(float(row[key]) * int(row["n_tokens"]) for row in group) / max(total_tokens, 1)
        out.append(merged)
    return out


def summarize_generation(model_name: str, rows: list[dict[str, object]]) -> dict[str, object]:
    out = {"model": model_name, "n": len(rows)}
    harmful = [row for row in rows if row["split"] == "harmful"]
    benign = [row for row in rows if row["split"] == "benign"]
    for split, group in (("harmful", harmful), ("benign", benign)):
        out[f"{split}_ok_rate"] = sum(bool(row["ok"]) for row in group) / max(len(group), 1)
        out[f"{split}_clean_generation_rate"] = sum(bool(row["clean_generation"]) for row in group) / max(len(group), 1)
    out["harmful_attempted_refusal_rate"] = sum(bool(row.get("harmful_attempted_refusal")) for row in harmful) / max(
        len(harmful), 1
    )
    out["harmful_bad_attempt_rate"] = sum(bool(row.get("harmful_bad_refusal_attempt")) for row in harmful) / max(
        len(harmful), 1
    )
    out["harmful_unsafe_continuation_rate"] = sum(bool(row.get("harmful_unsafe_continuation")) for row in harmful) / max(
        len(harmful), 1
    )
    out["benign_over_refusal_rate"] = sum(bool(row.get("benign_over_refusal")) for row in benign) / max(len(benign), 1)
    return out


@torch.no_grad()
def transcoder_patch_generate(
    donor,
    recipient,
    tokenizer,
    transcoders,
    user: str,
    *,
    layers: tuple[int, ...],
    device: str,
    max_new_tokens: int,
    input_mode: str,
    output_mode: str,
) -> str:
    donor_cache = {}
    donor_handles = []
    recipient_handles = []

    def make_donor_hook(layer):
        def hook(module, inputs, output):
            donor_cache[layer] = folded_norm_input(module, inputs, output, input_mode)

        return hook

    def make_recipient_hook(layer):
        def hook(_module, _inputs, output):
            donor_in = donor_cache[layer]
            _features, recon = transcoders[layer](donor_in)
            return replace_first_tensor(output, recon.to(device=first_tensor(output)[0].device, dtype=first_tensor(output)[0].dtype))

        return hook

    for layer in layers:
        donor_handles.append(module_mlp_in(donor, layer).register_forward_hook(make_donor_hook(layer)))
        recipient_handles.append(module_mlp_out(recipient, layer, output_mode).register_forward_hook(make_recipient_hook(layer)))

    prompt = chat_prompt(tokenizer, user)
    enc = tokenizer(prompt, return_tensors="pt").to(device)
    input_ids = enc["input_ids"]
    attention_mask = enc["attention_mask"]
    try:
        for _ in range(max_new_tokens):
            donor_cache.clear()
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
def run_generation_check(
    donor,
    recipient,
    tokenizer,
    transcoders,
    *,
    layers,
    examples_per_split,
    max_new_tokens,
    device,
    input_mode,
    output_mode,
):
    rows_by_model: dict[str, list[dict[str, object]]] = {}
    prompts = [("harmful", x) for x in HARMFUL_PROMPTS[:examples_per_split]] + [
        ("benign", x) for x in BENIGN_PROMPTS[:examples_per_split]
    ]
    for model_name, model in (("base", donor), ("abliterated", recipient)):
        records = []
        for split, user in prompts:
            text = generate(model, tokenizer, str(user), device=device, max_new_tokens=max_new_tokens)
            record = {"model": model_name, "split": split, "prompt": str(user), "text": text}
            record.update(score_record(split, str(user), text))
            records.append(record)
        rows_by_model[model_name] = records

    patch_name = "gemmascope_transcoder_patch_" + "+".join(str(x) for x in layers)
    records = []
    for split, user in prompts:
        text = transcoder_patch_generate(
            donor,
            recipient,
            tokenizer,
            transcoders,
            str(user),
            layers=layers,
            device=device,
            max_new_tokens=max_new_tokens,
            input_mode=input_mode,
            output_mode=output_mode,
        )
        record = {"model": patch_name, "split": split, "prompt": str(user), "text": text}
        record.update(score_record(split, str(user), text))
        records.append(record)
    rows_by_model[patch_name] = records
    metrics = [summarize_generation(name, records) for name, records in rows_by_model.items()]
    return metrics, [row for rows in rows_by_model.values() for row in rows]


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


def write_summary(path: Path, recon_rows, gen_rows, layers: tuple[int, ...], input_mode: str, output_mode: str) -> None:
    lines = [
        "# Gemma-2-2B GemmaScope Transcoder Validation",
        "",
        f"Layers: `{','.join(str(x) for x in layers)}`.",
        f"MLP input mode: `{input_mode}`.",
        f"MLP output mode: `{output_mode}`.",
        "",
        "## Reconstruction",
        "",
        "| layer | split | donor EV | donor cosine | L0 mean | recipient-to-donor MSE | recon gap closed | file |",
        "|---:|---|---:|---:|---:|---:|---:|---|",
    ]
    for row in recon_rows:
        lines.append(
            f"| {row['layer']} | {row['split']} | {row['donor_recon_ev']:.3f} | "
            f"{row['donor_recon_cosine']:.3f} | {row['donor_feature_l0_mean']:.1f} | "
            f"{row['recipient_to_donor_mlp_mse']:.5f} | {row['recon_vs_recipient_gap_closed']:.3f} | "
            f"`{row['transcoder_file']}` |"
        )
    if gen_rows:
        lines.extend(
            [
                "",
                "## Generation",
                "",
                "| model | harmful clean | harmful attempt | harmful bad | unsafe continuation | benign helpful | benign over-refusal |",
                "|---|---:|---:|---:|---:|---:|---:|",
            ]
        )
        for row in gen_rows:
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
            "## Decision Rule",
            "",
            "- Treat reconstruction as a gate, not as interpretability.",
            "- A GemmaScope transcoder branch is promising only if reconstruction is close enough to compete with the native MLP-output baselines and if decoded-output patching preserves the causal repair.",
            "- If reconstruction is weak on the instruction-tuned donor/recipient distribution, do not spend effort naming features from this basis yet.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--model-dtype", choices=("float16", "bfloat16"), default="float16")
    ap.add_argument("--transcoder-dtype", choices=("float16", "bfloat16", "float32"), default="float16")
    ap.add_argument("--layers", default=",".join(str(x) for x in DEFAULT_LAYERS))
    ap.add_argument("--l0-target", type=int, default=80)
    ap.add_argument("--examples-per-split", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--run-generation", action="store_true")
    ap.add_argument("--input-mode", choices=("folded_pre_norm", "hf_post_norm"), default="folded_pre_norm")
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--donor-model-id", default=MODEL_IDS["base"])
    ap.add_argument("--recipient-model-id", default=MODEL_IDS["abliterated"])
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_ints(args.layers)
    model_dtype = torch.float16 if args.model_dtype == "float16" else torch.bfloat16
    transcoder_dtype = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }[args.transcoder_dtype]
    cache_dir = os.environ.get("HF_HOME")

    model_ids = {"base": args.donor_model_id, "abliterated": args.recipient_model_id}

    tokenizer = AutoTokenizer.from_pretrained(args.donor_model_id)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    print("[files] selecting GemmaScope transcoders", flush=True)
    files = {layer: select_transcoder_file(layer, args.l0_target) for layer in layers}
    for layer, filename in files.items():
        print(f"[files] L{layer}: {filename}", flush=True)

    print("[load] transcoders", flush=True)
    transcoders = {
        layer: load_transcoder(filename, device=args.device, dtype=transcoder_dtype, cache_dir=cache_dir)
        for layer, filename in files.items()
    }
    print("[load] base donor", flush=True)
    donor = AutoModelForCausalLM.from_pretrained(args.donor_model_id, torch_dtype=model_dtype).to(args.device)
    donor.eval()
    print("[load] abliterated recipient", flush=True)
    recipient = AutoModelForCausalLM.from_pretrained(args.recipient_model_id, torch_dtype=model_dtype).to(args.device)
    recipient.eval()

    print("[recon] collecting reconstruction metrics", flush=True)
    recon_rows = collect_reconstruction_rows(
        donor,
        recipient,
        tokenizer,
        transcoders,
        files,
        layers=layers,
        examples_per_split=args.examples_per_split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
        input_mode=args.input_mode,
        output_mode=args.output_mode,
    )

    gen_metrics = []
    gen_records = []
    if args.run_generation:
        print("[generation] running decoded-output patch check", flush=True)
        gen_metrics, gen_records = run_generation_check(
            donor,
            recipient,
            tokenizer,
            transcoders,
            layers=layers,
            examples_per_split=args.examples_per_split,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            input_mode=args.input_mode,
            output_mode=args.output_mode,
        )

    recon_path = args.result_dir / "gemma2_2b_gemmascope_transcoder_reconstruction.csv"
    gen_metrics_path = args.result_dir / "gemma2_2b_gemmascope_transcoder_generation_metrics.csv"
    gen_records_path = args.result_dir / "gemma2_2b_gemmascope_transcoder_generation_records.jsonl"
    summary_path = args.result_dir / "GEMMA2_2B_GEMMASCOPE_TRANSCODER_VALIDATION_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"

    write_csv(recon_path, recon_rows)
    if gen_metrics:
        write_csv(gen_metrics_path, gen_metrics)
        write_jsonl(gen_records_path, gen_records)
    write_summary(summary_path, recon_rows, gen_metrics, layers, args.input_mode, args.output_mode)
    manifest_path.write_text(
        json.dumps(
            {
                "models": model_ids,
                "transcoder_repo": TRANSCODER_REPO,
                "transcoder_files": {str(k): v for k, v in files.items()},
                "layers": list(layers),
                "input_mode": args.input_mode,
                "output_mode": args.output_mode,
                "examples_per_split": args.examples_per_split,
                "run_generation": args.run_generation,
                "outputs": {
                    "reconstruction": str(recon_path),
                    "generation_metrics": str(gen_metrics_path) if gen_metrics else None,
                    "generation_records": str(gen_records_path) if gen_records else None,
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
