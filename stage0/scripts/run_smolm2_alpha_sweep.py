#!/usr/bin/env python3
"""SmolLM2 base->instruct task-vector alpha sweep.

This is a qualitative transformer sanity check. It interpolates/extrapolates
between a base checkpoint and its instruct fine-tune:

    theta(alpha) = theta_base + alpha * (theta_instruct - theta_base)

and saves deterministic generations for a small prompt suite.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


ROOT = Path(__file__).resolve().parents[2]
RESULT_DIR = ROOT / "stage0" / "results"
BASE_ID = "HuggingFaceTB/SmolLM2-135M"
INSTRUCT_ID = "HuggingFaceTB/SmolLM2-135M-Instruct"


PROMPTS = [
    {
        "id": "math_simple",
        "mode": "chat",
        "text": "What is 2 + 3? Answer with just the number.",
    },
    {
        "id": "instruction_style",
        "mode": "chat",
        "text": "Rewrite this sentence politely: give me the file now.",
    },
    {
        "id": "short_reasoning",
        "mode": "chat",
        "text": "If Alice has 4 apples and gives Bob 1, how many apples does Alice have?",
    },
    {
        "id": "raw_completion",
        "mode": "raw",
        "text": "Question: What is the capital of France? Answer:",
    },
]


def render_prompt(tokenizer, item: dict[str, str]) -> str:
    if item["mode"] == "raw":
        return item["text"]
    messages = [{"role": "user", "content": item["text"]}]
    try:
        return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception:
        return f"User: {item['text']}\nAssistant:"


def merged_state(base_state, instruct_state, alpha: float):
    out = {}
    for k, base_v in base_state.items():
        inst_v = instruct_state[k]
        if torch.is_floating_point(base_v):
            out[k] = base_v + alpha * (inst_v - base_v)
        else:
            out[k] = base_v
    return out


@torch.no_grad()
def generate(model, tokenizer, prompt: str, max_new_tokens: int) -> str:
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    out = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0], skip_special_tokens=False)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--alphas", default="0,0.25,0.5,0.75,1.0,1.25")
    ap.add_argument("--device", default="cuda:0" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--max-new-tokens", type=int, default=48)
    args = ap.parse_args()

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = RESULT_DIR / "smollm2_alpha_sweep_generations.jsonl"

    print(f"[load tokenizer] {INSTRUCT_ID}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(INSTRUCT_ID)

    print(f"[load base cpu] {BASE_ID}", flush=True)
    base_model = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float32, device_map="cpu")
    base_state = {k: v.detach().cpu() for k, v in base_model.state_dict().items()}
    del base_model

    print(f"[load instruct cpu] {INSTRUCT_ID}", flush=True)
    instruct_model = AutoModelForCausalLM.from_pretrained(INSTRUCT_ID, torch_dtype=torch.float32, device_map="cpu")
    instruct_state = {k: v.detach().cpu() for k, v in instruct_model.state_dict().items()}
    del instruct_model

    print("[init runtime model]", flush=True)
    model = AutoModelForCausalLM.from_pretrained(BASE_ID, torch_dtype=torch.float16, device_map=args.device)

    alphas = [float(x) for x in args.alphas.split(",")]
    with out_path.open("w", encoding="utf-8") as f:
        for alpha in alphas:
            print(f"[alpha] {alpha}", flush=True)
            state = merged_state(base_state, instruct_state, alpha)
            state = {k: v.to(dtype=torch.float16) if torch.is_floating_point(v) else v for k, v in state.items()}
            model.load_state_dict(state, strict=True)
            model.eval()
            for item in PROMPTS:
                prompt = render_prompt(tokenizer, item)
                text = generate(model, tokenizer, prompt, args.max_new_tokens)
                rec = {
                    "base_model": BASE_ID,
                    "instruct_model": INSTRUCT_ID,
                    "alpha": alpha,
                    "prompt_id": item["id"],
                    "mode": item["mode"],
                    "prompt": item["text"],
                    "rendered_prompt": prompt,
                    "generation": text,
                }
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    print(f"[save] {out_path}")
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())

