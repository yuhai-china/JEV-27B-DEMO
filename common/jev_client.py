"""Tiny client for autotrust/JEV-27B — one engine, two systems.

System 1 · decide()  typed decisions (noul true/false · choice 2-16 options · score 0-5): one forward pass,
                     calibrated probabilities, no text generation.
System 2 · chat()    the unmodified Qwen3.8-27B in the same engine (optionally with step-by-step thinking).

Two backends, chosen by environment variables:
  * your own vLLM server (default)   JEV_URL=http://localhost:8000          (see common/serve_jev27b.sh)
  * a hosted JEV API                 JEV_URL=https://<host>  JEV_API_KEY=<key>
    System 1 then goes through POST /v1/decide and every request carries "Authorization: Bearer <key>".
    Set JEV_BACKEND=vllm or JEV_BACKEND=decide to override the automatic choice.
"""
from __future__ import annotations

import concurrent.futures as cf
import functools
import json
import math
import os

import requests

URL = os.environ.get("JEV_URL", "http://localhost:8000").rstrip("/").removesuffix("/v1")
KEY = os.environ.get("JEV_API_KEY")
HEADERS = {"Authorization": f"Bearer {KEY}"} if KEY else {}
BACKEND = os.environ.get("JEV_BACKEND") or ("decide" if KEY else "vllm")
REPO = "autotrust/JEV-27B"
LETTERS = "ABCDEFGHIJKLMNOP"
SCORE = [str(i) for i in range(6)]
_http = requests.Session()
_http.headers.update(HEADERS)


def as_text(x) -> str:
    return x if isinstance(x, str) else json.dumps(x, ensure_ascii=False)


@functools.lru_cache(maxsize=1)
def _bundle():
    """Decision-head bias, verbalizer token ids and per-kind temperatures (only needed for the raw vLLM backend)."""
    local = os.environ.get("JEV_BUNDLE")  # optional local bundle directory instead of the Hub
    if local:
        get = lambda f: os.path.join(local, f)
    else:
        from huggingface_hub import hf_hub_download
        get = lambda f: hf_hub_download(REPO, f)
    return json.load(open(get("adapter_vllm/decision_head.json"))), json.load(open(get("calibration.json")))["per_kind"]


def _decide_hosted(kind, state, question, options):
    body = {"kind": kind, "state": as_text(state), "question": question}
    if kind == "choice":
        body["options"] = options
    r = _http.post(f"{URL}/v1/decide", json=body, timeout=120)
    r.raise_for_status()
    d = r.json()
    return dict(zip(d["options"], d["probabilities"]))


# Read the full distribution over the option tokens: override the model's generation_config defaults
# (Qwen ships top_k=20, top_p=0.95), which would otherwise truncate the processed logprobs and zero out tail options.
_FULL = {"max_tokens": 1, "temperature": 1.0, "top_p": 1.0, "top_k": 0, "min_p": 0.0, "repetition_penalty": 1.0,
         "add_special_tokens": False, "return_tokens_as_token_ids": True}


def _decide_vllm(kind, state, question, options):
    dh, temps = _bundle()
    lines = options if kind != "choice" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
    prompt = f"[kind] {kind}\n[state] {as_text(state)}\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"
    s = dh["slots"]["ranges"][kind][0]
    ids = dh["verbalizer_ids"][s: s + len(options)]
    r = _http.post(f"{URL}/v1/completions", json={
        "model": "jev-decision", "prompt": prompt, "logprobs": len(options), "allowed_token_ids": ids, **_FULL}, timeout=120)
    r.raise_for_status()
    lp = {int(k.split(":")[1]): v for k, v in r.json()["choices"][0]["logprobs"]["top_logprobs"][0].items()}
    z = [(lp.get(t, -1e9) + dh["bias"][s + i]) / temps[kind] for i, t in enumerate(ids)]
    e = [math.exp(x - max(z)) for x in z]
    return {o: x / sum(e) for o, x in zip(options, e)}


def decide(kind: str, state, question: str, options: list[str] | None = None) -> dict[str, float]:
    """Return {option: probability}. kind = 'noul' | 'choice' | 'score'."""
    options = {"noul": ["false", "true"], "score": SCORE}.get(kind, options)
    return (_decide_hosted if BACKEND == "decide" else _decide_vllm)(kind, state, question, options)


# Raw template: renders the decision prompt verbatim, with each image as a vision placeholder at its position.
_RAW_MM_TEMPLATE = ("{%- for m in messages -%}{%- if m['content'] is string -%}{{ m['content'] }}{%- else -%}{%- for c in m['content'] -%}"
                    "{%- if c['type'] == 'text' -%}{{ c['text'] }}{%- else -%}<|vision_start|><|image_pad|><|vision_end|>{%- endif -%}"
                    "{%- endfor -%}{%- endif -%}{%- endfor -%}")


def _image_part(img) -> dict:
    """img: http(s) URL, data URL, or a local file path."""
    if isinstance(img, str) and img.startswith(("http://", "https://", "data:")):
        url = img
    else:
        import base64, mimetypes
        mime = mimetypes.guess_type(str(img))[0] or "image/jpeg"
        url = f"data:{mime};base64," + base64.b64encode(open(img, "rb").read()).decode()
    return {"type": "image_url", "image_url": {"url": url}}


def decide_mm(kind: str, state_parts: list, question: str, options: list[str] | None = None) -> dict[str, float]:
    """System 1 with images (needs the multimodal server, see common/serve_jev27b_mm.sh).
    state_parts: list of str and {"image": url_or_path} items, rendered in order inside [state]."""
    options = {"noul": ["false", "true"], "score": SCORE}.get(kind, options)
    if BACKEND == "decide":  # self-hosted serve_decide.py accepts images in state (the hosted API does not)
        parts = [{"image": _image_part(p["image"])["image_url"]["url"]} if isinstance(p, dict) and "image" in p else as_text(p)
                 for p in state_parts]
        body = {"kind": kind, "state": parts, "question": question, **({"options": options} if kind == "choice" else {})}
        r = _http.post(f"{URL}/v1/decide", json=body, timeout=300)
        r.raise_for_status()
        return dict(zip(r.json()["options"], r.json()["probabilities"]))
    dh, temps = _bundle()
    lines = options if kind != "choice" else [f"{LETTERS[i]}) {o}" for i, o in enumerate(options)]
    content = [{"type": "text", "text": f"[kind] {kind}\n[state] "}]
    for p in state_parts:
        content.append(_image_part(p["image"]) if isinstance(p, dict) and "image" in p else {"type": "text", "text": as_text(p)})
    content.append({"type": "text", "text": f"\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"})
    s = dh["slots"]["ranges"][kind][0]
    ids = dh["verbalizer_ids"][s: s + len(options)]
    r = _http.post(f"{URL}/v1/chat/completions", json={
        "model": "jev-decision", "messages": [{"role": "user", "content": content}], "chat_template": _RAW_MM_TEMPLATE,
        "add_generation_prompt": False, "logprobs": True, "top_logprobs": len(options), "allowed_token_ids": ids, **_FULL}, timeout=300)
    r.raise_for_status()
    top = r.json()["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    lp = {int(t["token"].split(":")[1]): t["logprob"] for t in top}
    z = [(lp.get(t, -1e9) + dh["bias"][s + i]) / temps[kind] for i, t in enumerate(ids)]
    e = [math.exp(x - max(z)) for x in z]
    return {o: x / sum(e) for o, x in zip(options, e)}


def decide_many(reqs: list[tuple], workers: int = 32) -> list[dict[str, float]]:
    """reqs = [(kind, state, question[, options]), ...] — sent concurrently; the server batches them on the GPU."""
    with cf.ThreadPoolExecutor(workers) as ex:
        return list(ex.map(lambda a: decide(*a), reqs))


def expected_score(p: dict[str, float]) -> float:
    return sum(int(k) * v for k, v in p.items())


def p_true(p: dict[str, float]) -> float:
    return p["true"]


def chat_raw(prompt: str, thinking: bool = False, max_tokens: int = 1024, **extra) -> dict:
    """System 2, full OpenAI-style response."""
    body = {"model": REPO, "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens,
            "temperature": 0.6 if thinking else 0.0, "chat_template_kwargs": {"enable_thinking": thinking}, **extra}
    r = _http.post(f"{URL}/v1/chat/completions", json=body, timeout=1800)
    r.raise_for_status()
    return r.json()


def reply_parts(resp: dict) -> tuple[str, str]:
    """(reasoning, final answer) from a chat response. Servers with a reasoning parser put the reasoning in
    message.reasoning / reasoning_content; servers without one return '<reasoning></think><answer>' in content."""
    m = resp["choices"][0]["message"]
    content = m.get("content") or ""
    reasoning = m.get("reasoning") or m.get("reasoning_content") or ""
    if reasoning:
        return reasoning.strip(), content.strip()
    return split_thinking(content)


def chat(prompt: str, thinking: bool = False, max_tokens: int = 1024) -> str:
    """System 2. Returns '<reasoning></think><answer>' when thinking (use split_thinking), else the answer."""
    reasoning, answer = reply_parts(chat_raw(prompt, thinking, max_tokens))
    return f"{reasoning}</think>{answer}" if reasoning else answer


def split_thinking(text: str) -> tuple[str, str]:
    """(reasoning, final answer) from a System 2 reply."""
    if "</think>" in text:
        a, b = text.rsplit("</think>", 1)
        return a.replace("<think>", "").strip(), b.strip()
    return "", text.strip()
