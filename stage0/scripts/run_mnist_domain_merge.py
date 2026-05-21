#!/usr/bin/env python3
"""Stage 0 MNIST domain-merge reproduction.

This is intentionally small and controlled. It creates one base model and a
handful of domain experts, then evaluates simple merge recipes. The goal is not
MNIST performance; it is a clean local merge dataset for later mechanistic
analysis.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, Subset
from torchvision import datasets
from torchvision.transforms import functional as TF


ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path("/data/gavin/model_merging/data")
ARTIFACT_DIR = ROOT / "stage0" / "artifacts" / "mnist_domain_merge"
RESULT_DIR = ROOT / "stage0" / "results"

TARGET_DOMAINS = ("clean", "rotate25", "noise", "invert")
ALL_EVALS = TARGET_DOMAINS + ("labelperm_task",)
EXPERTS = ("clean", "rotate25", "noise", "invert", "labelperm")


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class DomainMNIST(Dataset):
    def __init__(
        self,
        base: Dataset,
        *,
        domain: str,
        label_perm: bool = False,
        noise_std: float = 0.45,
        seed: int = 0,
    ) -> None:
        self.base = base
        self.domain = domain
        self.label_perm = label_perm
        self.noise_std = noise_std
        self.seed = seed

    def __len__(self) -> int:
        return len(self.base)

    def __getitem__(self, idx: int):
        img, y = self.base[idx]
        x = TF.to_tensor(img)

        if self.domain == "clean":
            pass
        elif self.domain == "rotate25":
            x = TF.rotate(x, angle=25.0)
        elif self.domain == "noise":
            gen = torch.Generator()
            gen.manual_seed(self.seed + idx)
            x = (x + torch.randn(x.shape, generator=gen) * self.noise_std).clamp(0.0, 1.0)
        elif self.domain == "invert":
            x = 1.0 - x
        else:
            raise ValueError(f"unknown domain {self.domain!r}")

        x = (x - 0.1307) / 0.3081
        if self.label_perm:
            y = (int(y) + 1) % 10
        return x, int(y)


class SmallCNN(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


@dataclass
class TrainConfig:
    batch_size: int
    base_epochs: int
    expert_epochs: int
    base_lr: float
    expert_lr: float
    train_size: int
    expert_size: int
    test_size: int
    seed: int
    device: str


def make_subset(ds: Dataset, n: int, seed: int) -> Subset:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(ds))[:n]
    return Subset(ds, idx.tolist())


def make_loaders(cfg: TrainConfig):
    DATA_ROOT.mkdir(parents=True, exist_ok=True)
    train_raw = datasets.MNIST(DATA_ROOT, train=True, download=True)
    test_raw = datasets.MNIST(DATA_ROOT, train=False, download=True)

    train_subset = make_subset(train_raw, cfg.train_size, cfg.seed)
    expert_subset = make_subset(train_raw, cfg.expert_size, cfg.seed + 17)
    test_subset = make_subset(test_raw, cfg.test_size, cfg.seed + 29)

    def train_loader(domain: str, *, label_perm: bool = False) -> DataLoader:
        ds = DomainMNIST(
            train_subset if not label_perm else expert_subset,
            domain=domain,
            label_perm=label_perm,
            seed=cfg.seed,
        )
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    def expert_loader(domain: str, *, label_perm: bool = False) -> DataLoader:
        ds = DomainMNIST(expert_subset, domain=domain, label_perm=label_perm, seed=cfg.seed)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=2)

    def eval_loader(domain: str) -> DataLoader:
        if domain == "labelperm_task":
            ds = DomainMNIST(test_subset, domain="clean", label_perm=True, seed=cfg.seed + 1000)
        else:
            ds = DomainMNIST(test_subset, domain=domain, seed=cfg.seed + 1000)
        return DataLoader(ds, batch_size=cfg.batch_size, shuffle=False, num_workers=2)

    return train_loader, expert_loader, eval_loader


def train(model: nn.Module, loader: DataLoader, *, epochs: int, lr: float, device: str) -> list[float]:
    model.to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    losses: list[float] = []
    for _ in range(epochs):
        model.train()
        total = 0.0
        count = 0
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            opt.zero_grad(set_to_none=True)
            loss = F.cross_entropy(model(x), y)
            loss.backward()
            opt.step()
            total += float(loss.item()) * len(y)
            count += len(y)
        losses.append(total / max(count, 1))
    return losses


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, *, device: str) -> float:
    model.to(device)
    model.eval()
    correct = 0
    total = 0
    for x, y in loader:
        x = x.to(device)
        y = y.to(device)
        pred = model(x).argmax(dim=-1)
        correct += int((pred == y).sum().item())
        total += len(y)
    return correct / max(total, 1)


def clone_model_from_state(state: dict[str, torch.Tensor], device: str) -> SmallCNN:
    m = SmallCNN().to(device)
    m.load_state_dict(state)
    return m


def float_items(state: dict[str, torch.Tensor]) -> Iterable[str]:
    for k, v in state.items():
        if torch.is_floating_point(v):
            yield k


def linear_merge(
    base: dict[str, torch.Tensor],
    experts: list[dict[str, torch.Tensor]],
    *,
    scale: float = 1.0,
) -> dict[str, torch.Tensor]:
    out = {k: v.detach().cpu().clone() for k, v in base.items()}
    for k in float_items(base):
        delta = torch.stack([e[k].detach().cpu() - base[k].detach().cpu() for e in experts]).mean(0)
        out[k] = base[k].detach().cpu() + scale * delta
    return out


def soup_merge(experts: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    out = {k: v.detach().cpu().clone() for k, v in experts[0].items()}
    for k in float_items(experts[0]):
        out[k] = torch.stack([e[k].detach().cpu() for e in experts]).mean(0)
    return out


def ties_merge(
    base: dict[str, torch.Tensor],
    experts: list[dict[str, torch.Tensor]],
    *,
    keep_frac: float = 0.2,
) -> dict[str, torch.Tensor]:
    out = {k: v.detach().cpu().clone() for k, v in base.items()}
    for k in float_items(base):
        deltas = torch.stack([e[k].detach().cpu() - base[k].detach().cpu() for e in experts])
        flat_abs = deltas.abs().flatten(1)
        kth = torch.quantile(flat_abs, q=max(0.0, 1.0 - keep_frac), dim=1)
        mask = deltas.abs() >= kth.view(-1, *([1] * (deltas.ndim - 1)))
        trimmed = deltas * mask
        sign_vote = trimmed.sign().sum(0)
        elected = torch.sign(sign_vote)
        aligned = trimmed * (trimmed.sign() == elected).to(trimmed.dtype)
        denom = (aligned != 0).sum(0).clamp_min(1)
        merged_delta = aligned.sum(0) / denom
        out[k] = base[k].detach().cpu() + merged_delta
    return out


def dare_merge(
    base: dict[str, torch.Tensor],
    experts: list[dict[str, torch.Tensor]],
    *,
    drop_prob: float = 0.8,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    keep_prob = 1.0 - drop_prob
    out = {k: v.detach().cpu().clone() for k, v in base.items()}
    for k in float_items(base):
        merged = []
        for e in experts:
            delta = e[k].detach().cpu() - base[k].detach().cpu()
            mask = (torch.rand(delta.shape, generator=gen) < keep_prob).to(delta.dtype)
            merged.append(delta * mask / keep_prob)
        out[k] = base[k].detach().cpu() + torch.stack(merged).mean(0)
    return out


def della_like_merge(
    base: dict[str, torch.Tensor],
    experts: list[dict[str, torch.Tensor]],
    *,
    min_keep: float = 0.05,
    max_keep: float = 0.35,
    seed: int = 0,
) -> dict[str, torch.Tensor]:
    gen = torch.Generator()
    gen.manual_seed(seed)
    out = {k: v.detach().cpu().clone() for k, v in base.items()}
    for k in float_items(base):
        merged = []
        for e in experts:
            delta = e[k].detach().cpu() - base[k].detach().cpu()
            mag = delta.abs()
            if mag.numel() == 0 or float(mag.max()) == 0.0:
                merged.append(delta)
                continue
            rank_proxy = mag / (mag.max() + 1e-12)
            keep_prob = min_keep + (max_keep - min_keep) * rank_proxy
            mask = (torch.rand(delta.shape, generator=gen) < keep_prob).to(delta.dtype)
            merged.append(delta * mask / keep_prob.clamp_min(1e-6))
        out[k] = base[k].detach().cpu() + torch.stack(merged).mean(0)
    return out


def conflict_stats(
    base: dict[str, torch.Tensor],
    experts: list[dict[str, torch.Tensor]],
    expert_names: list[str],
    group: str,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for k in float_items(base):
        deltas = torch.stack([e[k].detach().cpu() - base[k].detach().cpu() for e in experts])
        signs = deltas.sign()
        nonzero = signs != 0
        any_pos = (signs > 0).any(0)
        any_neg = (signs < 0).any(0)
        conflict = any_pos & any_neg
        active = nonzero.any(0)
        rows.append(
            {
                "group": group,
                "tensor": k,
                "experts": "+".join(expert_names),
                "numel": int(deltas[0].numel()),
                "active_frac": float(active.float().mean().item()),
                "sign_conflict_frac_all": float(conflict.float().mean().item()),
                "sign_conflict_frac_active": float((conflict & active).sum().item() / max(active.sum().item(), 1)),
                "mean_abs_delta": float(deltas.abs().mean().item()),
                "max_abs_delta": float(deltas.abs().max().item()),
            }
        )
    return rows


def evaluate_all(
    states: dict[str, dict[str, torch.Tensor]],
    eval_loaders: dict[str, DataLoader],
    *,
    device: str,
) -> list[dict[str, object]]:
    rows = []
    for model_name, state in states.items():
        model = clone_model_from_state(state, device)
        accs = {}
        for domain, loader in eval_loaders.items():
            accs[domain] = evaluate(model, loader, device=device)
        target_accs = [accs[k] for k in TARGET_DOMAINS]
        rows.append(
            {
                "model": model_name,
                **{f"acc_{k}": v for k, v in accs.items()},
                "acc_mean": float(np.mean(target_accs)),
                "acc_worst": float(np.min(target_accs)),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--base-epochs", type=int, default=2)
    ap.add_argument("--expert-epochs", type=int, default=2)
    ap.add_argument("--base-lr", type=float, default=1e-3)
    ap.add_argument("--expert-lr", type=float, default=5e-4)
    ap.add_argument("--train-size", type=int, default=12000)
    ap.add_argument("--expert-size", type=int, default=8000)
    ap.add_argument("--test-size", type=int, default=3000)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--no-save-artifacts", action="store_true")
    args = ap.parse_args()

    set_seed(args.seed)
    cfg = TrainConfig(
        batch_size=args.batch_size,
        base_epochs=args.base_epochs,
        expert_epochs=args.expert_epochs,
        base_lr=args.base_lr,
        expert_lr=args.expert_lr,
        train_size=args.train_size,
        expert_size=args.expert_size,
        test_size=args.test_size,
        seed=args.seed,
        device=args.device,
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)

    train_loader_fn, expert_loader_fn, eval_loader_fn = make_loaders(cfg)
    eval_loaders = {d: eval_loader_fn(d) for d in ALL_EVALS}

    print(f"[stage0] device={cfg.device}")
    print("[base] train clean")
    base_model = SmallCNN()
    base_losses = train(
        base_model,
        train_loader_fn("clean"),
        epochs=cfg.base_epochs,
        lr=cfg.base_lr,
        device=cfg.device,
    )
    base_state = {k: v.detach().cpu().clone() for k, v in base_model.state_dict().items()}

    expert_states: dict[str, dict[str, torch.Tensor]] = {}
    expert_losses: dict[str, list[float]] = {}
    for name in EXPERTS:
        print(f"[expert] fine-tune {name}")
        model = clone_model_from_state(base_state, cfg.device)
        label_perm = name == "labelperm"
        domain = "clean" if label_perm else name
        losses = train(
            model,
            expert_loader_fn(domain, label_perm=label_perm),
            epochs=cfg.expert_epochs,
            lr=cfg.expert_lr,
            device=cfg.device,
        )
        expert_states[name] = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        expert_losses[name] = losses

    good_names = ["clean", "rotate25", "noise", "invert"]
    good_experts = [expert_states[n] for n in good_names]
    conflict_names = good_names + ["labelperm"]
    conflict_experts = [expert_states[n] for n in conflict_names]

    states: dict[str, dict[str, torch.Tensor]] = {"base": base_state}
    states.update({f"expert_{k}": v for k, v in expert_states.items()})

    merge_groups = {
        "good": (good_names, good_experts),
        "with_labelperm": (conflict_names, conflict_experts),
        "clean_vs_labelperm": (["clean", "labelperm"], [expert_states["clean"], expert_states["labelperm"]]),
    }

    conflict_rows: list[dict[str, object]] = []
    for group, (names, experts) in merge_groups.items():
        states[f"{group}_linear_taskavg"] = linear_merge(base_state, experts)
        states[f"{group}_soup"] = soup_merge(experts)
        states[f"{group}_ties_keep20"] = ties_merge(base_state, experts, keep_frac=0.2)
        states[f"{group}_dare_drop80"] = dare_merge(base_state, experts, drop_prob=0.8, seed=args.seed)
        states[f"{group}_della_like"] = della_like_merge(base_state, experts, seed=args.seed)
        conflict_rows.extend(conflict_stats(base_state, experts, names, group))

    print("[eval] evaluating models")
    metric_rows = evaluate_all(states, eval_loaders, device=cfg.device)
    metric_rows = sorted(metric_rows, key=lambda r: (-float(r["acc_mean"]), r["model"]))

    metric_path = RESULT_DIR / "mnist_domain_merge_metrics.csv"
    conflict_path = RESULT_DIR / "mnist_domain_merge_conflicts.csv"
    summary_path = RESULT_DIR / "mnist_domain_merge_summary.json"
    write_csv(metric_path, metric_rows)
    write_csv(conflict_path, conflict_rows)

    summary = {
        "config": vars(args),
        "base_losses": base_losses,
        "expert_losses": expert_losses,
        "best_by_mean": metric_rows[0],
        "best_by_worst": max(metric_rows, key=lambda r: float(r["acc_worst"])),
        "metrics_csv": str(metric_path),
        "conflicts_csv": str(conflict_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    if not args.no_save_artifacts:
        torch.save(
            {
                "base": base_state,
                "experts": expert_states,
                "merges": {k: v for k, v in states.items() if k.startswith("good_") or k.startswith("with_")},
                "config": vars(args),
            },
            ARTIFACT_DIR / "checkpoints.pt",
        )

    print(f"[save] {metric_path}")
    print(f"[save] {conflict_path}")
    print(f"[save] {summary_path}")
    print("\nTop models by mean accuracy:")
    for r in metric_rows[:10]:
        print(
            f"  {r['model']:<28} mean={r['acc_mean']:.3f} worst={r['acc_worst']:.3f} "
            + " ".join(f"{d}={r[f'acc_{d}']:.3f}" for d in TARGET_DOMAINS)
            + f" labelperm_task={r['acc_labelperm_task']:.3f}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
