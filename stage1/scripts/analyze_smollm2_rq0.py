#!/usr/bin/env python3
"""First-pass RQ0 analysis for SmolLM2 synthetic merges.

This uses teacher-forced loss and hidden-state similarity as fast transformer
proxies before doing expensive generation-based or full cross-layer patching.
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

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModelForCausalLM, AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
import run_smollm2_synthetic_experts as smol  # noqa: E402


RESULT_DIR = ROOT / "stage1" / "results"
CACHE_DIR = ROOT / "stage1" / "cache"
ARTIFACT_DIR = ROOT / "stage0" / "artifacts" / "smollm2_synthetic_experts"
TASKS = ("arith", "polite", "refusal")
EXPERT_TO_TASK = {"arith": "arith", "polite": "polite", "refusal": "refusal"}
PATCH_LAYERS = (0, 5, 10, 15, 20, 25, 29)
MERGE_SPECS = {
    "merge_all_linear": ("arith", "polite", "refusal"),
    "merge_arith_polite": ("arith", "polite"),
    "merge_arith_refusal": ("arith", "refusal"),
    "merge_polite_refusal": ("polite", "refusal"),
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)


def load_expert_state(task: str) -> dict[str, torch.Tensor]:
    return torch.load(
        ARTIFACT_DIR / f"expert_{task}.pt",
        map_location="cpu",
        weights_only=False,
    )["state"]


def merge_states(base, experts: list[dict[str, torch.Tensor]], *, label: str):
    out = {}
    n = len(experts)
    for i, (k, base_v) in enumerate(base.items(), start=1):
        if i == 1 or i % 80 == 0:
            print(f"  [{label}] tensor {i}/{len(base)}", flush=True)
        if torch.is_floating_point(base_v):
            merged = base_v.float().clone()
            base_f = base_v.float()
            for e in experts:
                merged.add_(e[k].float() - base_f, alpha=1.0 / n)
            out[k] = merged.half()
        else:
            out[k] = base_v
    return out


def layer_keys(state: dict[str, torch.Tensor], layer: int) -> list[str]:
    prefix = f"model.layers.{layer}."
    return [k for k in state if k.startswith(prefix)]


def patch_layer(target, donor, layer: int):
    out = {k: v.detach().cpu().clone() for k, v in target.items()}
    for k in layer_keys(out, layer):
        out[k] = donor[k].detach().cpu().clone()
    return out


def flatten_delta(base, state, keys=None) -> torch.Tensor:
    if keys is None:
        keys = [k for k, v in base.items() if torch.is_floating_point(v)]
    parts = []
    for k in keys:
        if torch.is_floating_point(base[k]):
            parts.append((state[k].float() - base[k].float()).reshape(-1))
    if not parts:
        return torch.empty(0)
    return torch.cat(parts)


def sampled_delta(base, state, keys, *, max_values: int, seed: int) -> torch.Tensor:
    """Deterministic sampled delta vector for fast geometry screening."""
    parts = []
    per_key = max(1, max_values // max(len(keys), 1))
    for key_i, k in enumerate(keys):
        if not torch.is_floating_point(base[k]):
            continue
        delta = (state[k].float() - base[k].float()).reshape(-1)
        if delta.numel() > per_key:
            gen = torch.Generator()
            gen.manual_seed(seed + key_i * 9973)
            idx = torch.randperm(delta.numel(), generator=gen)[:per_key]
            delta = delta[idx]
        parts.append(delta)
    if not parts:
        return torch.empty(0)
    return torch.cat(parts)


def cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = float(a.norm().item() * b.norm().item())
    if denom == 0.0:
        return float("nan")
    return float(torch.dot(a, b).item() / denom)


def sign_agreement(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0:
        return float("nan")
    active = (a != 0) | (b != 0)
    if int(active.sum().item()) == 0:
        return float("nan")
    return float((torch.sign(a[active]) == torch.sign(b[active])).float().mean().item())


def load_delta_cache(path: Path | None):
    if path is None or not path.exists():
        return None
    print(f"[cache] loading delta vectors from {path}", flush=True)
    return torch.load(path, map_location="cpu", weights_only=False)


def load_state_cache(cache_dir: Path | None) -> dict[str, dict[str, torch.Tensor]]:
    if cache_dir is None:
        return {}
    manifest_path = cache_dir / "manifest.json"
    if not manifest_path.exists():
        print(f"[state-cache] no manifest at {manifest_path}; rebuilding missing merge states", flush=True)
        return {}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    states = {}
    for name, rel_path in manifest.get("merge_state_files", {}).items():
        path = cache_dir / rel_path
        if not path.exists():
            print(f"[state-cache] missing {path}; rebuilding {name}", flush=True)
            continue
        print(f"[state-cache] loading {name} from {path}", flush=True)
        states[name] = torch.load(path, map_location="cpu", weights_only=False)
    return states


def get_delta_vec(
    cache,
    model_name: str,
    scope: str,
    *,
    base_state,
    state,
    keys,
    max_values: int,
    seed: int,
) -> torch.Tensor:
    if cache is not None:
        try:
            return cache["vectors"][model_name][scope].float()
        except KeyError:
            print(f"[cache] miss model={model_name} scope={scope}; sampling on the fly", flush=True)
    return sampled_delta(base_state, state, keys, max_values=max_values, seed=seed)


def make_task_loader(tokenizer, task: str, *, n: int, seed: int, max_length: int, batch_size: int):
    examples = smol.make_examples(task, n, seed + 50_000 + smol.TASK_SEED_OFFSET[task])
    ds = smol.ChatSFTDataset(tokenizer, examples, max_length)
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=lambda b: smol.collate(b, tokenizer.pad_token_id),
    )


@torch.no_grad()
def task_loss_and_reps(model, loader, *, device: str, rep_layers: tuple[int, ...]):
    total_loss = 0.0
    total_tokens = 0
    reps = {layer: [] for layer in rep_layers}
    reps["logits"] = []
    for batch in loader:
        batch = {k: v.to(device) for k, v in batch.items()}
        out = model(
            input_ids=batch["input_ids"],
            attention_mask=batch["attention_mask"],
            labels=batch["labels"],
            output_hidden_states=True,
            use_cache=False,
        )
        # Recompute token-count weighted loss because HF returns mean over labels.
        n_tok = int((batch["labels"] != -100).sum().item())
        total_loss += float(out.loss.item()) * n_tok
        total_tokens += n_tok

        last_pos = batch["attention_mask"].sum(dim=1) - 1
        for layer in rep_layers:
            h = out.hidden_states[layer + 1]  # hidden_states[0] is embeddings
            reps[layer].append(h[torch.arange(h.shape[0], device=device), last_pos].detach().cpu().float())
        logits = out.logits[torch.arange(out.logits.shape[0], device=device), last_pos]
        reps["logits"].append(logits.detach().cpu().float())
    mean_loss = total_loss / max(total_tokens, 1)
    reps = {k: torch.cat(v, dim=0) for k, v in reps.items()}
    return mean_loss, reps


def row_cosine_mean(a: torch.Tensor, b: torch.Tensor) -> float:
    n = min(len(a), len(b))
    a = F.normalize(a[:n].float(), dim=1)
    b = F.normalize(b[:n].float(), dim=1)
    return float((a * b).sum(1).mean().item())


def load_model_with_state(model, state, device: str):
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def correlation_rows(df: pd.DataFrame) -> list[dict[str, object]]:
    targets = ["target_loss", "loss_improvement_vs_base", "loss_gap_closed_vs_base"]
    skip = {
        "target_model",
        "expert",
        "task",
        "target_loss",
        "expert_loss",
        "base_loss",
        "loss_improvement_vs_base",
        "loss_gap_closed_vs_base",
    }
    rows = []
    for pred in [c for c in df.columns if c not in skip]:
        if not pd.api.types.is_numeric_dtype(df[pred]):
            continue
        x = df[pred].replace([np.inf, -np.inf], np.nan)
        for target in targets:
            tmp = pd.concat([x, df[target]], axis=1).dropna()
            if len(tmp) < 4:
                continue
            rows.append(
                {
                    "predictor": pred,
                    "target": target,
                    "n": len(tmp),
                    "pearson": tmp[pred].corr(tmp[target], method="pearson"),
                    "spearman": tmp[pred].corr(tmp[target], method="spearman"),
                }
            )
    return sorted(rows, key=lambda r: (r["target"], -abs(r["spearman"])))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--eval-examples", type=int, default=24)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--max-length", type=int, default=160)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--rep-layers", default="0,5,10,15,20,25,29")
    ap.add_argument("--max-delta-values", type=int, default=120_000)
    ap.add_argument(
        "--delta-cache",
        type=Path,
        default=CACHE_DIR / "smollm2_delta_cache.pt",
        help="Optional sampled-delta cache from build_smollm2_delta_cache.py.",
    )
    ap.add_argument(
        "--no-delta-cache",
        action="store_true",
        help="Ignore any cached sampled deltas and recompute on the fly.",
    )
    ap.add_argument(
        "--state-cache-dir",
        type=Path,
        default=CACHE_DIR / "smollm2_state_cache",
        help="Optional full merged-state cache from build_smollm2_state_cache.py.",
    )
    ap.add_argument(
        "--no-state-cache",
        action="store_true",
        help="Ignore cached merged states and rebuild them in memory.",
    )
    ap.add_argument(
        "--skip-patch",
        action="store_true",
        help="Skip selected layer patching for faster predictor-only runs.",
    )
    ap.add_argument(
        "--result-dir",
        type=Path,
        default=RESULT_DIR,
        help="Directory for RQ0 CSV/JSON outputs.",
    )
    args = ap.parse_args()

    args.result_dir.mkdir(parents=True, exist_ok=True)
    rep_layers = tuple(int(x) for x in args.rep_layers.split(",") if x)
    delta_cache = None if args.no_delta_cache else load_delta_cache(args.delta_cache)

    print("[load] tokenizer/base/experts", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(smol.TOKENIZER_ID)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    base_model_cpu = AutoModelForCausalLM.from_pretrained(smol.BASE_ID, torch_dtype=torch.float16, device_map="cpu")
    base_state = {k: v.detach().cpu().half() for k, v in base_model_cpu.state_dict().items()}
    del base_model_cpu
    experts = {task: load_expert_state(task) for task in TASKS}

    states = {"base": base_state}
    states.update({f"expert_{k}": v for k, v in experts.items()})
    cached_merge_states = {} if args.no_state_cache else load_state_cache(args.state_cache_dir)
    print("[merge] loading/building target states", flush=True)
    for merge_name, expert_names in MERGE_SPECS.items():
        if merge_name in cached_merge_states:
            states[merge_name] = cached_merge_states[merge_name]
        else:
            states[merge_name] = merge_states(
                base_state,
                [experts[name] for name in expert_names],
                label=merge_name,
            )

    loaders = {
        task: make_task_loader(
            tokenizer,
            task,
            n=args.eval_examples,
            seed=args.seed,
            max_length=args.max_length,
            batch_size=args.batch_size,
        )
        for task in TASKS
    }

    print("[eval] losses and reps", flush=True)
    runtime = AutoModelForCausalLM.from_pretrained(smol.BASE_ID, torch_dtype=torch.float16).to(args.device)
    loss_cache: dict[tuple[str, str], float] = {}
    rep_cache: dict[tuple[str, str], dict[int | str, torch.Tensor]] = {}
    for name, state in states.items():
        print(f"  [eval] {name}", flush=True)
        load_model_with_state(runtime, state, args.device)
        for task, loader in loaders.items():
            loss, reps = task_loss_and_reps(runtime, loader, device=args.device, rep_layers=rep_layers)
            loss_cache[(name, task)] = loss
            rep_cache[(name, task)] = reps

    target_names = ["base", "merge_all_linear", "merge_arith_polite", "merge_arith_refusal", "merge_polite_refusal"]
    predictor_rows = []
    print("[predictors]", flush=True)
    all_float_keys = [k for k, v in base_state.items() if torch.is_floating_point(v)]
    for target in target_names:
        print(f"  [predictors] target={target}", flush=True)
        target_delta_all = get_delta_vec(
            delta_cache,
            target,
            "all",
            base_state=base_state,
            state=states[target],
            keys=all_float_keys,
            max_values=args.max_delta_values,
            seed=args.seed,
        )
        for expert in TASKS:
            task = EXPERT_TO_TASK[expert]
            expert_name = f"expert_{expert}"
            print(f"    [expert] {expert}", flush=True)
            expert_delta_all = get_delta_vec(
                delta_cache,
                expert_name,
                "all",
                base_state=base_state,
                state=experts[expert],
                keys=all_float_keys,
                max_values=args.max_delta_values,
                seed=args.seed,
            )
            base_loss = loss_cache[("base", task)]
            expert_loss = loss_cache[(expert_name, task)]
            target_loss = loss_cache[(target, task)]
            row: dict[str, object] = {
                "target_model": target,
                "expert": expert,
                "task": task,
                "target_loss": target_loss,
                "expert_loss": expert_loss,
                "base_loss": base_loss,
                "loss_improvement_vs_base": base_loss - target_loss,
                "loss_gap_closed_vs_base": (
                    (base_loss - target_loss) / (base_loss - expert_loss)
                    if base_loss != expert_loss
                    else np.nan
                ),
                "weight_cos_all": cosine(expert_delta_all, target_delta_all),
                "sign_agree_all": sign_agreement(expert_delta_all, target_delta_all),
            }
            for layer in rep_layers:
                keys = layer_keys(base_state, layer)
                scope = f"l{layer}"
                e_delta = get_delta_vec(
                    delta_cache,
                    expert_name,
                    scope,
                    base_state=base_state,
                    state=experts[expert],
                    keys=keys,
                    max_values=args.max_delta_values,
                    seed=args.seed + layer,
                )
                t_delta = get_delta_vec(
                    delta_cache,
                    target,
                    scope,
                    base_state=base_state,
                    state=states[target],
                    keys=keys,
                    max_values=args.max_delta_values,
                    seed=args.seed + layer,
                )
                row[f"weight_cos_l{layer}"] = cosine(e_delta, t_delta)
                row[f"sign_agree_l{layer}"] = sign_agreement(e_delta, t_delta)
                row[f"act_rowcos_l{layer}"] = row_cosine_mean(
                    rep_cache[(expert_name, task)][layer],
                    rep_cache[(target, task)][layer],
                )
            row["act_rowcos_logits"] = row_cosine_mean(
                rep_cache[(expert_name, task)]["logits"],
                rep_cache[(target, task)]["logits"],
            )
            predictor_rows.append(row)

    patch_rows = []
    if args.skip_patch:
        print("[patch] skipped", flush=True)
    else:
        print("[patch] selected layer patches into merge_all_linear", flush=True)
        target_name = "merge_all_linear"
        for expert in TASKS:
            expert_name = f"expert_{expert}"
            task = EXPERT_TO_TASK[expert]
            target_loss = loss_cache[(target_name, task)]
            expert_loss = loss_cache[(expert_name, task)]
            for layer in PATCH_LAYERS:
                print(f"  [patch] donor={expert} layer={layer}", flush=True)
                patched = patch_layer(states[target_name], experts[expert], layer)
                load_model_with_state(runtime, patched, args.device)
                patched_loss, _ = task_loss_and_reps(runtime, loaders[task], device=args.device, rep_layers=())
                patch_rows.append(
                    {
                        "target_model": target_name,
                        "donor_expert": expert,
                        "task": task,
                        "layer": layer,
                        "target_loss": target_loss,
                        "expert_loss": expert_loss,
                        "patched_loss": patched_loss,
                        "loss_delta_vs_target": target_loss - patched_loss,
                        "loss_gap_closed_vs_expert": (
                            (target_loss - patched_loss) / (target_loss - expert_loss)
                            if target_loss != expert_loss
                            else np.nan
                        ),
                    }
                )

    del runtime
    torch.cuda.empty_cache()

    predictor_path = args.result_dir / "smollm2_rq0_predictors.csv"
    patch_path = args.result_dir / "smollm2_rq0_patch_effects.csv"
    corr_path = args.result_dir / "smollm2_rq0_correlations.csv"
    write_csv(predictor_path, predictor_rows)
    write_csv(patch_path, patch_rows)
    corr = correlation_rows(pd.DataFrame(predictor_rows))
    write_csv(corr_path, corr)
    (args.result_dir / "smollm2_rq0_summary.json").write_text(
        json.dumps(
            {
                "predictor_rows": len(predictor_rows),
                "patch_rows": len(patch_rows),
                "outputs": {
                    "predictors": str(predictor_path),
                    "patch_effects": str(patch_path),
                    "correlations": str(corr_path),
                },
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"[save] {predictor_path}")
    print(f"[save] {patch_path}")
    print(f"[save] {corr_path}")
    print("\nTop correlations with loss_gap_closed_vs_base:")
    for row in [r for r in corr if r["target"] == "loss_gap_closed_vs_base"][:10]:
        print(f"  {row['predictor']:<24} spearman={row['spearman']:+.3f} pearson={row['pearson']:+.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
