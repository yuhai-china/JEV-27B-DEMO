"""Tiny client for autotrust/JEV-27B served by vLLM — one engine, two systems.

System 1 · decide()  typed decisions (noul true/false · choice 2-16 options · score 0-5) from the `jev-decision` LoRA:
                     one prefill step, calibrated probabilities, no text generation.
System 2 · chat()    the unmodified Qwen3.8-27B in the same engine (optionally with step-by-step thinking).

Start the server first (see common/serve_jev27b.sh). Set JEV_URL if it is not on localhost:8000.
"""
from __future__ import annotations

import concurrent.futures as cf
import json
import math
import os

import requests
from huggingface_hub import hf_hub_download

URL = os.environ.get("JEV_URL", "http://localhost:8000")
REPO = "autotrust/JEV-27B"
_local = os.environ.get("JEV_BUNDLE")  # optional local bundle directory instead of the Hub
_get = (lambda f: os.path.join(_local, f)) if _local else (lambda f: hf_hub_download(REPO, f))
DH = json.load(open(_get("adapter_vllm/decision_head.json")))  # head bias + verbalizer token ids
T = json.load(open(_get("calibration.json")))["per_kind"]       # per-kind temperatures
LETTERS = "ABCDEFGHIJKLMNOP"
SCORE = [str(i) for i in range(6)]


def as_text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


def decide(kind: str, state, question: str, options: list[str] | None = None) -> dict[str, float]:
    """Return {option: probability}. kind = 'noul' | 'choice' | 'score'."""
    options = {"noul": ["false", "true"], "score": SCORE}.get(kind, options)
    lines = options if kind != "choice" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
    prompt = f"[kind] {kind}\n[state] {as_text(state)}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
    s = DH["slots"]["ranges"][kind][0]
    ids = DH["verbalizer_ids"][s: s + len(options)]
    r = requests.post(f"{URL}/v1/completions", json={
        "model": "jev-decision", "prompt": prompt, "max_tokens": 1, "temperature": 1.0,
        "logprobs": len(options), "allowed_token_ids": ids,
        "add_special_tokens": False, "return_tokens_as_token_ids": True}, timeout=120)
    r.raise_for_status()
    lp = {int(k.split(":")[1]): v for k, v in r.json()["choices"][0]["logprobs"]["top_logprobs"][0].items()}
    z = [(lp.get(t, -1e9) + DH["bias"][s + i]) / T[kind] for i, t in enumerate(ids)]
    e = [math.exp(x - max(z)) for x in z]
    return {o: x / sum(e) for o, x in zip(options, e)}


def decide_many(reqs: list[tuple], workers: int = 64) -> list[dict[str, float]]:
    """reqs = [(kind, state, question[, options]), ...] — sent concurrently; vLLM batches them on the GPU."""
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda a: decide(*a), reqs))


def expected_score(p: dict[str, float]) -> float:
    return sum(int(k) * v for k, v in p.items())


def p_true(p: dict[str, float]) -> float:
    return p["true"]


def chat(prompt: str, thinking: bool = False, max_tokens: int = 1024) -> str:
    """System 2. With thinking=True the reply is '<reasoning></think><answer>' (use split_thinking)."""
    r = requests.post(f"{URL}/v1/chat/completions", json={
        "model": REPO, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
        "temperature": 0.6 if thinking else 0.0, "chat_template_kwargs": {"enable_thinking": thinking}}, timeout=900)
    r.raise_for_status()
    return r.json()["choices"][0]["message"]["content"]


def split_thinking(text: str) -> tuple[str, str]:
    """(reasoning, final answer) from a System 2 reply."""
    if "</think>" in text:
        a, b = text.rsplit("</think>", 1)
        return a.replace("<think>", "").strip(), b.strip()
    return "", text.strip()
