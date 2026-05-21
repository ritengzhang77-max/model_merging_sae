#!/usr/bin/env python3
"""RQ0 analysis on Stage 0 MNIST merge artifacts.

This script compares several candidate explanatory bases:

- weight/task-vector geometry;
- sign agreement;
- activation similarity;
- module-wise causal patching.

It produces wide predictor rows for each (target model, expert capability) and
detailed patching rows for each donor/target/module/eval-domain combination.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "stage0" / "scripts"))
import run_mnist_domain_merge as mm  # noqa: E402


ARTIFACT_PATH = ROOT / "stage0" / "artifacts" / "mnist_domain_merge" / "checkpoints.pt"
STAGE0_METRICS = ROOT / "stage0" / "results" / "mnist_domain_merge_metrics.csv"
RESULT_DIR = ROOT / "stage1" / "results"

MODULES = {
    "conv1": ("conv1.weight", "conv1.bias"),
    "conv2": ("conv2.weight", "conv2.bias"),
    "fc1": ("fc1.weight", "fc1.bias"),
    "fc2": ("fc2.weight", "fc2.bias"),
}
MODULES_WITH_ALL = {"all": tuple(), **MODULES}
EXPERT_TO_DOMAIN = {
    "clean": "clean",
    "rotate25": "rotate25",
    "noise": "noise",
    "invert": "invert",
    "labelperm": "labelperm_task",
}


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    keys = sorted({k for r in rows for k in r})
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def flatten_delta(
    base: dict[str, torch.Tensor],
    state: dict[str, torch.Tensor],
    keys: Iterable[str] | None = None,
) -> torch.Tensor:
    if keys is None:
        keys = [k for k, v in base.items() if torch.is_floating_point(v)]
    parts = []
    for k in keys:
        if torch.is_floating_point(base[k]):
            parts.append((state[k].float() - base[k].float()).reshape(-1))
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
    agree = torch.sign(a[active]) == torch.sign(b[active])
    return float(agree.float().mean().item())


def norm_ratio(a: torch.Tensor, b: torch.Tensor) -> float:
    # How much of expert-delta norm appears in target-delta.
    denom = float(a.norm().item())
    if denom == 0.0:
        return float("nan")
    return float(b.norm().item() / denom)


def patch_state(
    target: dict[str, torch.Tensor],
    donor: dict[str, torch.Tensor],
    module: str,
) -> dict[str, torch.Tensor]:
    out = {k: v.detach().cpu().clone() for k, v in target.items()}
    if module == "all":
        for k in out:
            out[k] = donor[k].detach().cpu().clone()
        return out
    for k in MODULES[module]:
        out[k] = donor[k].detach().cpu().clone()
    return out


@torch.no_grad()
def evaluate_state(
    state: dict[str, torch.Tensor],
    loaders: dict[str, torch.utils.data.DataLoader],
    device: str,
) -> dict[str, float]:
    model = mm.clone_model_from_state(state, device)
    out = {}
    for domain, loader in loaders.items():
        out[domain] = mm.evaluate(model, loader, device=device)
    del model
    return out


@torch.no_grad()
def extract_reps(
    state: dict[str, torch.Tensor],
    loader: torch.utils.data.DataLoader,
    *,
    device: str,
    max_examples: int,
) -> dict[str, torch.Tensor]:
    model = mm.clone_model_from_state(state, device)
    model.eval()
    reps = {"conv1": [], "conv2": [], "fc1": [], "logits": []}
    seen = 0
    for x, _ in loader:
        x = x.to(device)
        x = F.relu(model.conv1(x))
        x = F.max_pool2d(x, 2)
        reps["conv1"].append(x.flatten(1).detach().cpu())
        x = F.relu(model.conv2(x))
        x = F.max_pool2d(x, 2)
        reps["conv2"].append(x.flatten(1).detach().cpu())
        x = x.flatten(1)
        x = F.relu(model.fc1(x))
        reps["fc1"].append(x.detach().cpu())
        x = model.fc2(x)
        reps["logits"].append(x.detach().cpu())
        seen += len(reps["logits"][-1])
        if seen >= max_examples:
            break
    del model
    return {k: torch.cat(v, dim=0)[:max_examples].float() for k, v in reps.items()}


def reduce_features(x: torch.Tensor, max_features: int) -> torch.Tensor:
    if x.shape[1] <= max_features:
        return x
    idx = torch.linspace(0, x.shape[1] - 1, steps=max_features).long()
    return x[:, idx]


def linear_cka(x: torch.Tensor, y: torch.Tensor, *, max_features: int) -> float:
    n = min(len(x), len(y))
    x = x[:n].float()
    y = y[:n].float()
    x = reduce_features(x, max_features)
    y = reduce_features(y, max_features)
    x = x - x.mean(0, keepdim=True)
    y = y - y.mean(0, keepdim=True)
    k = x @ x.T
    l = y @ y.T
    denom = torch.sqrt((k * k).sum() * (l * l).sum())
    if float(denom.item()) == 0.0:
        return float("nan")
    return float(((k * l).sum() / denom).item())


def row_cosine_mean(x: torch.Tensor, y: torch.Tensor, *, max_features: int) -> float:
    n = min(len(x), len(y))
    x = x[:n].float()
    y = y[:n].float()
    x = reduce_features(x, max_features)
    y = reduce_features(y, max_features)
    x = F.normalize(x, dim=1)
    y = F.normalize(y, dim=1)
    return float((x * y).sum(1).mean().item())


def make_loaders(config: dict, device: str, *, num_workers: int):
    cfg = mm.TrainConfig(
        batch_size=int(config.get("batch_size", 256)),
        base_epochs=int(config.get("base_epochs", 2)),
        expert_epochs=int(config.get("expert_epochs", 2)),
        base_lr=float(config.get("base_lr", 1e-3)),
        expert_lr=float(config.get("expert_lr", 5e-4)),
        train_size=int(config.get("train_size", 12000)),
        expert_size=int(config.get("expert_size", 8000)),
        test_size=int(config.get("test_size", 3000)),
        seed=int(config.get("seed", 42)),
        device=device,
    )
    # Rebuild only eval loaders here with configurable workers. The Stage 0
    # helper uses worker processes, which is good for training but expensive for
    # hundreds of tiny patch-eval passes.
    test_raw = mm.datasets.MNIST(mm.DATA_ROOT, train=False, download=True)
    test_subset = mm.make_subset(test_raw, cfg.test_size, cfg.seed + 29)

    def eval_loader(domain: str):
        if domain == "labelperm_task":
            ds = mm.DomainMNIST(
                test_subset,
                domain="clean",
                label_perm=True,
                seed=cfg.seed + 1000,
            )
        else:
            ds = mm.DomainMNIST(test_subset, domain=domain, seed=cfg.seed + 1000)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=num_workers)

    return {d: eval_loader(d) for d in mm.ALL_EVALS}


def predictor_correlations(df: pd.DataFrame) -> list[dict[str, object]]:
    targets = ["target_acc", "target_gain_vs_base", "retention_frac_expert"]
    skip = {
        "target_model",
        "expert",
        "eval_domain",
        "target_acc",
        "expert_acc",
        "base_acc",
        "target_gain_vs_base",
        "retention_frac_expert",
    }
    rows = []
    for pred in [c for c in df.columns if c not in skip]:
        if not pd.api.types.is_numeric_dtype(df[pred]):
            continue
        valid_pred = df[pred].replace([np.inf, -np.inf], np.nan)
        if valid_pred.notna().sum() < 4:
            continue
        for target in targets:
            tmp = pd.concat([valid_pred, df[target]], axis=1).dropna()
            if len(tmp) < 4:
                continue
            pearson = tmp[pred].corr(tmp[target], method="pearson")
            spearman = tmp[pred].corr(tmp[target], method="spearman")
            rows.append(
                {
                    "predictor": pred,
                    "target": target,
                    "n": len(tmp),
                    "pearson": pearson,
                    "spearman": spearman,
                    "abs_pearson": abs(pearson) if not math.isnan(pearson) else np.nan,
                    "abs_spearman": abs(spearman) if not math.isnan(spearman) else np.nan,
                }
            )
    return sorted(rows, key=lambda r: (-(r["abs_spearman"] or 0), r["predictor"], r["target"]))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--artifact", type=Path, default=ARTIFACT_PATH)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-activation-examples", type=int, default=192)
    ap.add_argument("--max-activation-features", type=int, default=2048)
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--patch-eval-mode",
        choices=["expert_domain", "all_domains"],
        default="expert_domain",
        help="expert_domain is the fast RQ0 default; all_domains creates the full cross-domain patch matrix.",
    )
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    payload = torch.load(args.artifact, map_location="cpu", weights_only=False)
    base = payload["base"]
    experts = payload["experts"]
    merges = payload["merges"]
    config = payload.get("config", {})
    loaders = make_loaders(config, args.device, num_workers=args.num_workers)

    metric_df = pd.read_csv(STAGE0_METRICS)
    metric_lookup = metric_df.set_index("model").to_dict(orient="index")

    states = {"base": base}
    states.update({f"expert_{k}": v for k, v in experts.items()})
    states.update(merges)

    target_names = sorted(["base"] + list(merges.keys()))
    expert_names = sorted(experts.keys())

    # Cache activations only for models and domains needed by predictor rows.
    print("[activation] caching expert/target reps", flush=True)
    rep_cache: dict[tuple[str, str], dict[str, torch.Tensor]] = {}
    for model_name in [f"expert_{e}" for e in expert_names] + target_names:
        state = states[model_name]
        for expert in expert_names:
            domain = EXPERT_TO_DOMAIN[expert]
            key = (model_name, domain)
            if key not in rep_cache:
                rep_cache[key] = extract_reps(
                    state,
                    loaders[domain],
                    device=args.device,
                    max_examples=args.max_activation_examples,
                )

    print("[patch] evaluating module patches", flush=True)
    eval_cache: dict[str, dict[str, float]] = {}
    for name, state in states.items():
        if name in target_names or name.startswith("expert_"):
            eval_cache[name] = {
                k.replace("acc_", ""): v
                for k, v in metric_lookup.get(name, {}).items()
                if k.startswith("acc_")
            }
            missing = set(mm.ALL_EVALS) - set(eval_cache[name])
            if missing:
                eval_cache[name].update(evaluate_state(state, loaders, args.device))

    patch_rows: list[dict[str, object]] = []
    patch_gain_lookup: dict[tuple[str, str, str], float] = {}
    patch_idx = 0
    patch_total = len(target_names) * len(expert_names) * len(MODULES_WITH_ALL)
    for target_name in target_names:
        target_state = states[target_name]
        target_accs = eval_cache[target_name]
        for expert in expert_names:
            donor_name = f"expert_{expert}"
            donor_state = experts[expert]
            donor_accs = eval_cache[donor_name]
            for module in MODULES_WITH_ALL:
                patch_idx += 1
                if patch_idx == 1 or patch_idx % 50 == 0 or patch_idx == patch_total:
                    print(f"  [patch] {patch_idx}/{patch_total}", flush=True)
                patched = patch_state(target_state, donor_state, module)
                expert_domain = EXPERT_TO_DOMAIN[expert]
                eval_domains = (
                    mm.ALL_EVALS
                    if args.patch_eval_mode == "all_domains"
                    else (expert_domain,)
                )
                patched_accs = {}
                patched_model = mm.clone_model_from_state(patched, args.device)
                for domain in eval_domains:
                    patched_accs[domain] = mm.evaluate(patched_model, loaders[domain], device=args.device)
                del patched_model
                patch_gain_lookup[(target_name, expert, module)] = patched_accs[expert_domain] - target_accs[expert_domain]
                for domain in eval_domains:
                    patch_rows.append(
                        {
                            "target_model": target_name,
                            "donor_expert": expert,
                            "module": module,
                            "eval_domain": domain,
                            "target_acc": target_accs[domain],
                            "donor_acc": donor_accs[domain],
                            "patched_acc": patched_accs[domain],
                            "patch_gain_vs_target": patched_accs[domain] - target_accs[domain],
                            "patch_gap_closed_vs_donor": (
                                (patched_accs[domain] - target_accs[domain])
                                / (donor_accs[domain] - target_accs[domain])
                                if donor_accs[domain] != target_accs[domain]
                                else np.nan
                            ),
                        }
                    )

    print("[predictors] computing wide rows", flush=True)
    predictor_rows: list[dict[str, object]] = []
    pred_idx = 0
    pred_total = len(target_names) * len(expert_names)
    for target_name in target_names:
        target_state = states[target_name]
        target_accs = eval_cache[target_name]
        target_delta_all = flatten_delta(base, target_state)
        for expert in expert_names:
            pred_idx += 1
            if pred_idx == 1 or pred_idx % 10 == 0 or pred_idx == pred_total:
                print(f"  [predictors] {pred_idx}/{pred_total}", flush=True)
            expert_state = experts[expert]
            expert_model_name = f"expert_{expert}"
            expert_domain = EXPERT_TO_DOMAIN[expert]
            expert_acc = eval_cache[expert_model_name][expert_domain]
            base_acc = eval_cache["base"][expert_domain]
            target_acc = target_accs[expert_domain]
            expert_delta_all = flatten_delta(base, expert_state)

            row: dict[str, object] = {
                "target_model": target_name,
                "expert": expert,
                "eval_domain": expert_domain,
                "target_acc": target_acc,
                "expert_acc": expert_acc,
                "base_acc": base_acc,
                "target_gain_vs_base": target_acc - base_acc,
                "retention_frac_expert": target_acc / expert_acc if expert_acc != 0 else np.nan,
                "weight_cos_all": cosine(expert_delta_all, target_delta_all),
                "weight_norm_ratio_all": norm_ratio(expert_delta_all, target_delta_all),
                "sign_agree_all": sign_agreement(expert_delta_all, target_delta_all),
            }

            for module, keys in MODULES.items():
                e_delta = flatten_delta(base, expert_state, keys)
                t_delta = flatten_delta(base, target_state, keys)
                row[f"weight_cos_{module}"] = cosine(e_delta, t_delta)
                row[f"weight_norm_ratio_{module}"] = norm_ratio(e_delta, t_delta)
                row[f"sign_agree_{module}"] = sign_agreement(e_delta, t_delta)
                row[f"patch_gain_{module}"] = patch_gain_lookup[(target_name, expert, module)]
            row["patch_gain_all"] = patch_gain_lookup[(target_name, expert, "all")]

            expert_reps = rep_cache[(expert_model_name, expert_domain)]
            target_reps = rep_cache[(target_name, expert_domain)]
            for layer in ["conv1", "conv2", "fc1", "logits"]:
                row[f"act_cka_{layer}"] = linear_cka(
                    expert_reps[layer],
                    target_reps[layer],
                    max_features=args.max_activation_features,
                )
                row[f"act_rowcos_{layer}"] = row_cosine_mean(
                    expert_reps[layer],
                    target_reps[layer],
                    max_features=args.max_activation_features,
                )

            predictor_rows.append(row)

    predictor_path = RESULT_DIR / "mnist_rq0_predictors.csv"
    patch_path = RESULT_DIR / "mnist_rq0_patch_effects.csv"
    corr_path = RESULT_DIR / "mnist_rq0_correlations.csv"
    write_csv(predictor_path, predictor_rows)
    write_csv(patch_path, patch_rows)

    pred_df = pd.DataFrame(predictor_rows)
    corr_rows = predictor_correlations(pred_df)
    write_csv(corr_path, corr_rows)

    summary = {
        "predictor_rows": len(predictor_rows),
        "patch_rows": len(patch_rows),
        "top_spearman_target_acc": [
            r for r in corr_rows if r["target"] == "target_acc"
        ][:12],
        "outputs": {
            "predictors": str(predictor_path),
            "patch_effects": str(patch_path),
            "correlations": str(corr_path),
        },
    }
    (RESULT_DIR / "mnist_rq0_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"[save] {predictor_path}")
    print(f"[save] {patch_path}")
    print(f"[save] {corr_path}")
    print("\nTop correlations with target_acc:")
    for r in [x for x in corr_rows if x["target"] == "target_acc"][:10]:
        print(
            f"  {r['predictor']:<28} spearman={r['spearman']:+.3f} "
            f"pearson={r['pearson']:+.3f} n={r['n']}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
