#!/usr/bin/env python3
"""Validate GemmaScope MLP-output SAEs on the Gemma-2-2B abliterated repair target."""

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


SAE_REPO = "google/gemma-scope-2b-pt-mlp"
RESULT_DIR = ROOT / "stage3" / "results" / "gemma2_2b_gemmascope_mlp_sae_validation"
DEFAULT_LAYERS = (12, 16, 20)
WIDTH = "16k"


class JumpReluSae(nn.Module):
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
    def decode(self, features: torch.Tensor) -> torch.Tensor:
        return features @ self.W_dec + self.b_dec

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encode(x)
        return features, self.decode(features)


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


def module_mlp(model, layer: int, output_mode: str):
    if output_mode == "raw_mlp":
        return model.model.layers[layer].mlp
    if output_mode == "post_ff_norm":
        return model.model.layers[layer].post_feedforward_layernorm
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


def select_sae_file(layer: int, l0_target: int, repo: str = SAE_REPO) -> str:
    pattern = re.compile(rf"^layer_{layer}/width_{WIDTH}/average_l0_(\d+)/params\.npz$")
    candidates = []
    for filename in list_repo_files(repo, repo_type="model"):
        match = pattern.match(filename)
        if match:
            l0 = int(match.group(1))
            candidates.append((abs(l0 - l0_target), l0, filename))
    if not candidates:
        raise ValueError(f"no GemmaScope MLP SAE file found for layer {layer}")
    return sorted(candidates)[0][2]


def load_sae(filename: str, *, device: str, dtype: torch.dtype, cache_dir: str | None) -> JumpReluSae:
    path = hf_hub_download(SAE_REPO, filename, cache_dir=cache_dir)
    params = np.load(path)
    sae = JumpReluSae(params, device=device, dtype=dtype)
    sae.eval()
    return sae


def make_prompt_rows(tokenizer, examples_per_split: int, max_length: int):
    rows = []
    for split, prompts in (("harmful", HARMFUL_PROMPTS[:examples_per_split]), ("benign", BENIGN_PROMPTS[:examples_per_split])):
        for user in prompts:
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
    return tensor[mask.bool()].float()


def metric_row(layer, split, donor_out, recipient_out, mask, sae, filename):
    donor_target = flatten_valid(donor_out, mask).cpu()
    recipient_target = flatten_valid(recipient_out, mask).cpu()
    features, recon = sae(donor_target.to(sae.W_enc.device))
    recon = recon.float().cpu()

    err = recon - donor_target
    sse = float((err * err).sum().item())
    centered = donor_target - donor_target.mean(dim=0, keepdim=True)
    tss = float((centered * centered).sum().item())
    gap_sse = float(((recipient_target - donor_target) ** 2).sum().item())
    return {
        "layer": layer,
        "split": split,
        "sae_file": filename,
        "n_tokens": int(donor_target.shape[0]),
        "donor_recon_mse": sse / max(donor_target.numel(), 1),
        "donor_recon_ev": 1.0 - sse / max(tss, 1e-12),
        "donor_recon_cosine": float(torch.nn.functional.cosine_similarity(recon, donor_target, dim=-1).mean().item()),
        "donor_feature_l0_mean": float((features > 0).sum(dim=-1).float().mean().item()),
        "donor_feature_l0_median": float((features > 0).sum(dim=-1).float().median().item()),
        "recipient_to_donor_mlp_mse": gap_sse / max(donor_target.numel(), 1),
        "recon_vs_recipient_gap_closed": 1.0 - sse / max(gap_sse, 1e-12),
    }


def aggregate_metric_rows(rows):
    grouped: dict[tuple[int, str], list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault((int(row["layer"]), str(row["split"])), []).append(row)
    out = []
    numeric = [key for key in rows[0] if key not in {"layer", "split", "sae_file"}] if rows else []
    for (layer, split), group in sorted(grouped.items()):
        tokens = sum(int(row["n_tokens"]) for row in group)
        merged = {"layer": layer, "split": split, "sae_file": group[0]["sae_file"], "n_tokens": tokens}
        for key in numeric:
            if key == "n_tokens":
                continue
            merged[key] = sum(float(row[key]) * int(row["n_tokens"]) for row in group) / max(tokens, 1)
        out.append(merged)
    return out


@torch.no_grad()
def collect_reconstruction_rows(
    donor,
    recipient,
    tokenizer,
    saes,
    files,
    *,
    layers,
    examples_per_split,
    batch_size,
    max_length,
    device,
    output_mode,
):
    prompt_rows = make_prompt_rows(tokenizer, examples_per_split, max_length)
    out_rows = []
    donor_cache, donor_handles = make_mlp_cache_hooks(donor, layers, output_mode)
    recipient_cache, recipient_handles = make_mlp_cache_hooks(recipient, layers, output_mode)
    try:
        for start in range(0, len(prompt_rows), batch_size):
            chunk = prompt_rows[start : start + batch_size]
            batch = collate_prompt_rows(tokenizer, chunk)
            batch = {key: value.to(device) for key, value in batch.items()}
            for layer in layers:
                donor_cache[layer].clear()
                recipient_cache[layer].clear()
            _ = donor(**batch, use_cache=False)
            _ = recipient(**batch, use_cache=False)
            for split in ("harmful", "benign"):
                idx = torch.tensor([row["split"] == split for row in chunk], device=device)
                if not bool(idx.any().item()):
                    continue
                mask = batch["attention_mask"][idx].bool()
                for layer in layers:
                    out_rows.append(
                        metric_row(
                            layer,
                            split,
                            donor_cache[layer][0][idx],
                            recipient_cache[layer][0][idx],
                            mask,
                            saes[layer],
                            files[layer],
                        )
                    )
    finally:
        remove_hooks(donor_handles)
        remove_hooks(recipient_handles)
    return aggregate_metric_rows(out_rows)


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
def sae_patch_generate(donor, recipient, tokenizer, saes, user: str, *, layers, device, max_new_tokens, output_mode):
    donor_cache = {}
    donor_handles = []
    recipient_handles = []

    def make_donor_hook(layer):
        def hook(_module, _inputs, output):
            tensor, _rest = first_tensor(output)
            donor_cache[layer] = tensor.detach()

        return hook

    def make_recipient_hook(layer):
        def hook(_module, _inputs, output):
            features, recon = saes[layer](donor_cache[layer])
            tensor, _rest = first_tensor(output)
            return replace_first_tensor(output, recon.to(device=tensor.device, dtype=tensor.dtype))

        return hook

    for layer in layers:
        donor_handles.append(module_mlp(donor, layer, output_mode).register_forward_hook(make_donor_hook(layer)))
        recipient_handles.append(module_mlp(recipient, layer, output_mode).register_forward_hook(make_recipient_hook(layer)))

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
def run_generation_check(donor, recipient, tokenizer, saes, *, layers, examples_per_split, max_new_tokens, device, output_mode):
    prompts = [("harmful", x) for x in HARMFUL_PROMPTS[:examples_per_split]] + [
        ("benign", x) for x in BENIGN_PROMPTS[:examples_per_split]
    ]
    rows_by_model: dict[str, list[dict[str, object]]] = {}
    for model_name, model in (("base", donor), ("abliterated", recipient)):
        records = []
        for split, user in prompts:
            text = generate(model, tokenizer, str(user), device=device, max_new_tokens=max_new_tokens)
            record = {"model": model_name, "split": split, "prompt": str(user), "text": text}
            record.update(score_record(split, str(user), text))
            records.append(record)
        rows_by_model[model_name] = records

    patch_name = "gemmascope_mlp_sae_patch_" + "+".join(str(x) for x in layers)
    records = []
    for split, user in prompts:
        text = sae_patch_generate(
            donor,
            recipient,
            tokenizer,
            saes,
            str(user),
            layers=layers,
            device=device,
            max_new_tokens=max_new_tokens,
            output_mode=output_mode,
        )
        record = {"model": patch_name, "split": split, "prompt": str(user), "text": text}
        record.update(score_record(split, str(user), text))
        records.append(record)
    rows_by_model[patch_name] = records
    return [summarize_generation(name, rows) for name, rows in rows_by_model.items()], [
        row for rows in rows_by_model.values() for row in rows
    ]


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


def write_summary(path: Path, recon_rows, gen_rows, layers, output_mode: str):
    lines = [
        "# Gemma-2-2B GemmaScope MLP SAE Validation",
        "",
        f"Layers: `{','.join(str(x) for x in layers)}`.",
        f"Output mode: `{output_mode}`.",
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
            f"`{row['sae_file']}` |"
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
            "- This basis is aligned to the successful post-feedforward MLP-update patch.",
            "- It must preserve the decoded MLP output well enough to compete with full `12-20:mlp`, PCA, and top-coordinate baselines.",
            "- If MLP SAE reconstruction cannot support decoded-output patching, feature naming is premature.",
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
    ap.add_argument("--examples-per-split", type=int, default=4)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--max-length", type=int, default=256)
    ap.add_argument("--max-new-tokens", type=int, default=64)
    ap.add_argument("--run-generation", action="store_true")
    ap.add_argument("--output-mode", choices=("post_ff_norm", "raw_mlp"), default="post_ff_norm")
    ap.add_argument("--result-dir", type=Path, default=RESULT_DIR)
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    args.result_dir.mkdir(parents=True, exist_ok=True)
    layers = parse_ints(args.layers)
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

    print("[recon] collecting reconstruction metrics", flush=True)
    recon_rows = collect_reconstruction_rows(
        donor,
        recipient,
        tokenizer,
        saes,
        files,
        layers=layers,
        examples_per_split=args.examples_per_split,
        batch_size=args.batch_size,
        max_length=args.max_length,
        device=args.device,
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
            saes,
            layers=layers,
            examples_per_split=args.examples_per_split,
            max_new_tokens=args.max_new_tokens,
            device=args.device,
            output_mode=args.output_mode,
        )

    recon_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_reconstruction.csv"
    gen_metrics_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_generation_metrics.csv"
    gen_records_path = args.result_dir / "gemma2_2b_gemmascope_mlp_sae_generation_records.jsonl"
    summary_path = args.result_dir / "GEMMA2_2B_GEMMASCOPE_MLP_SAE_VALIDATION_SUMMARY.md"
    manifest_path = args.result_dir / "manifest.json"

    write_csv(recon_path, recon_rows)
    if gen_metrics:
        write_csv(gen_metrics_path, gen_metrics)
        write_jsonl(gen_records_path, gen_records)
    write_summary(summary_path, recon_rows, gen_metrics, layers, args.output_mode)
    manifest_path.write_text(
        json.dumps(
            {
                "models": MODEL_IDS,
                "sae_repo": SAE_REPO,
                "sae_files": {str(k): v for k, v in files.items()},
                "layers": list(layers),
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
