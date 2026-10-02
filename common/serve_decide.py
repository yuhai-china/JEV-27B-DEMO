#!/usr/bin/env python3
"""vLLM OpenAI-compatible server with a built-in System 1 endpoint (POST /v1/decide) and adaptive thinking.

One engine serves both systems, for JEV-27B / JEV-27B-VL (Qwen3.8) and GEV-26B-A4B (Gemma-4):
  System 2  the unmodified base model, through the usual OpenAI endpoints (served model name)
  System 1  the LoRA module "jev-decision" (backbone LoRA + decision head as an lm_head LoRA), through /v1/decide

POST /v1/decide
  kind       "noul" (yes/no), "score" (0-5) or "choice" (2-256 options)
  state      string, JSON object, or a list mixing text and images: ["text", {"image": "https://... | data:..."}]
  question   string
  options    list of strings (choice only)
  strategy   choices with more than 16 options: "single" (one pass, labels A-P then Q-Z, AA, ...), "tournament"
             (groups of <=16 + a final of 16), "permute" (single pass over 4 option orders, averaged); default per model
  thinking   "off", "auto" (think only when the leading option is below `threshold`), "on" (always think);
             default per model (GEV: "auto"; JEV-27B: "off")
  threshold  System 1 confidence below which "auto" switches thinking on (default per model)
  reasoning controls, as for the base model's own chat API:
    chat_template_kwargs  passed to the base model's chat template (e.g. Qwen3.8: {"reasoning_effort": "low"})
    reasoning_effort      shorthand for chat_template_kwargs.reasoning_effort, if the base model has it
                          (Qwen3.8 / JEV-27B: xhigh (default), medium, low; Gemma-4 / GEV: not available)
    think_budget          maximum thinking tokens; default: no limit beyond the context window
  return_reasoning    include System 2's reasoning text
  debug      include the System 1 and System 2 distributions

Adaptive thinking: System 1 gives p1 in one pass. If thinking is switched on, the base model reasons in its thinking
mode over the same state, question and options; when the thinking channel closes, the answer-letter distribution p2 is
read in one step, and the result is p = (1 - w) * p1 + w * p2 (default w = 0.5: the fast decision and the reasoning get
equal weight, which keeps the calibration of System 1; p2 alone is close to one-hot and over-confident).

Response (same shape as the hosted JEV API, plus "thinking" when requested):
  {"kind", "effective_kind", "options", "probabilities", "choice_index", "choice", "adaptation", "protocol", "model",
   "usage", "elapsed_seconds", "num_model_requests", "thinking": {...}}

Run it like `python -m vllm.entrypoints.openai.api_server` (same flags), with --lora-modules jev-decision=<adapter dir>
and --trust-request-chat-template. decision_head.json is read from the adapter directory, calibration.json from its
parent (override with JEV_DECIDE_CALIBRATION).
"""
import asyncio
import json
import math
import os
import string
import time
import zlib
from typing import Literal

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import vllm.entrypoints.launchers.api_server.entry as entry
from vllm.entrypoints.openai.chat_completion.protocol import ChatCompletionRequest
from vllm.entrypoints.openai.completion.protocol import CompletionRequest

LORA = os.environ.get("JEV_DECIDE_LORA", "jev-decision")
MAX_OPTIONS = 256
CHUNK = 128  # vLLM caps logprob_token_ids at 128 per request
PROTOCOL = "jev27-bare-v1"
PLACEHOLDER = "XQXCONTENTXQX"

# Per-model settings. threshold / mix: adaptive-thinking defaults. GEV: fitted on 1,754 questions outside the Decision
# Index (CommonsenseQA, OpenBookQA, AQuA-RAT, MedMCQA, LogiQA, StrategyQA). JEV-27B: same defaults, not yet validated.
PROFILES = {
    "qwen": {"prefix": "", "image": "<|vision_start|><|image_pad|><|vision_end|>", "end_think": "</think>",
             "after_think": "\n\n", "strategy": "single", "threshold": 0.8, "mix": 0.5, "thinking": "off"},
    "gemma": {"prefix": "<bos>", "image": "<|image|>", "end_think": "<channel|>", "after_think": "",
              "strategy": "tournament", "threshold": 0.8, "mix": 0.5, "thinking": "auto"},
}
# Read the full distribution: override generation_config defaults (top_k/top_p) that would truncate processed logprobs.
READ = dict(max_tokens=1, temperature=1.0, top_p=1.0, top_k=0, min_p=0.0, repetition_penalty=1.0,
            add_special_tokens=False, return_tokens_as_token_ids=True)
S = {}


class DecideRequest(BaseModel):
    kind: Literal["noul", "score", "choice"]
    state: str | dict | list = ""
    question: str
    options: list[str] | None = None
    strategy: Literal["auto", "single", "tournament", "permute"] = "auto"
    thinking: Literal["default", "off", "auto", "on"] = "default"
    threshold: float | None = None
    think_budget: int | None = None
    reasoning_effort: str | None = None
    chat_template_kwargs: dict | None = None
    return_reasoning: bool = False
    debug: bool = False
    system2_only: bool = False   # skip System 1: think and return System 2's answer distribution (for callers that already have p1)


# ------------------------------------------------------------------ setup
def setup(args):
    from transformers import AutoConfig, AutoTokenizer

    path = next((m.path for m in (args.lora_modules or []) if m.name == LORA), None)
    if path is None:
        raise SystemExit(f"serve_decide: start vLLM with --lora-modules {LORA}=<adapter dir>")
    head = json.load(open(os.path.join(path, "decision_head.json")))
    calib = os.environ.get("JEV_DECIDE_CALIBRATION") or os.path.join(os.path.dirname(os.path.abspath(path)), "calibration.json")
    temps = json.load(open(calib))["per_kind"] if os.path.exists(calib) else {"noul": 1.0, "score": 1.0, "choice": 1.0}
    temps.setdefault("score", 1.0)
    src = args.tokenizer or args.model
    tok = AutoTokenizer.from_pretrained(src, trust_remote_code=args.trust_remote_code)
    mtype = AutoConfig.from_pretrained(args.model, trust_remote_code=args.trust_remote_code).model_type
    prof = dict(PROFILES["gemma" if "gemma" in mtype else "qwen"])
    for k in ("strategy", "threshold", "mix", "thinking"):
        if os.environ.get(f"JEV_DECIDE_{k.upper()}"):
            v = os.environ[f"JEV_DECIDE_{k.upper()}"]
            prof[k] = v if k in ("strategy", "thinking") else float(v)

    def single_token_labels(context):
        out = []
        for lab in list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]:
            t = tok.encode(lab, add_special_tokens=False)
            if len(t) == 1 and t[0] in tok.encode(context.format(lab), add_special_tokens=False):
                out.append((lab, t[0]))
            if len(out) == MAX_OPTIONS:
                break
        return out

    dec = single_token_labels("x\n{}) y")                      # System 1 option-line labels
    lo, hi = head["slots"]["ranges"]["choice"]
    assert [t for _, t in dec[: hi - lo]] == head["verbalizer_ids"][lo:hi], "first labels must be the trained A-P head"
    base = tok.encode("Answer: (", add_special_tokens=False)   # System 2 answer labels
    ans = []
    for lab in list(string.ascii_uppercase) + [a + b for a in string.ascii_uppercase for b in string.ascii_uppercase]:
        ids = tok.encode(f"Answer: ({lab})", add_special_tokens=False)
        if ids[: len(base)] == base and len(ids) == len(base) + 2:
            ans.append((lab, ids[len(base)]))
        if len(ans) == MAX_OPTIONS:
            break
    chat = tok.apply_chat_template([{"role": "user", "content": PLACEHOLDER}], tokenize=False, add_generation_prompt=True,
                                   enable_thinking=True)
    pre, post = chat.split(PLACEHOLDER)
    raw = ("{%- for m in messages -%}{%- if m['content'] is string -%}{{ m['content'] }}{%- else -%}"
           "{%- for c in m['content'] -%}{%- if c['type'] == 'text' -%}{{ c['text'] }}{%- else -%}" + prof["image"] +
           "{%- endif -%}{%- endfor -%}{%- endif -%}{%- endfor -%}")
    names = args.served_model_name
    S.update(tok=tok, has_effort="reasoning_effort" in (tok.chat_template or ""), think_cache={},
             max_logprobs=int(getattr(args, "max_logprobs", 20) or 20),
             head=head, temps=temps, labels=[l for l, _ in dec], label_ids=[t for _, t in dec], ans_labels=ans, prof=prof,
             think_pre=pre, think_post=post, raw=raw, end_think_id=tok.convert_tokens_to_ids(prof["end_think"]),
             model=(names[0] if isinstance(names, list) else names) or args.model, model_type=mtype)


# ------------------------------------------------------------------ helpers
def _parts(state):
    if isinstance(state, str):
        return [{"type": "text", "text": state}], False
    if isinstance(state, dict):
        return [{"type": "text", "text": json.dumps(state, ensure_ascii=False)}], False
    out, img = [], False
    for p in state:
        if isinstance(p, str):
            out.append({"type": "text", "text": p})
        elif isinstance(p, dict) and "image" in p:
            out.append({"type": "image_url", "image_url": {"url": p["image"]}}); img = True
        elif isinstance(p, dict) and p.get("type") == "image_url":
            out.append(p); img = True
        elif isinstance(p, dict) and p.get("type") == "text":
            out.append(p)
        else:
            out.append({"type": "text", "text": json.dumps(p, ensure_ascii=False)})
    return out, img


def _softmax(z):
    m = max(z); e = [math.exp(x - m) for x in z]; s = sum(e)
    return [x / s for x in e]


def _err(msg, code=400):
    return JSONResponse({"error": {"message": msg, "type": "BadRequestError", "code": code}}, status_code=code)


class Upstream(Exception):
    def __init__(self, resp):
        self.resp = resp


def _check(out):
    if hasattr(out, "error"):
        raise Upstream(out)
    return out


class Ctx:
    def __init__(self, raw):
        self.raw, self.requests, self.prompt_tokens, self.completion_tokens = raw, 0, 0, 0

    def count(self, out):
        self.requests += 1
        if getattr(out, "usage", None):
            self.prompt_tokens += out.usage.prompt_tokens or 0
            self.completion_tokens += out.usage.completion_tokens or 0


def _readout_allowed(ids):
    """Read-out path: restrict the next token to `ids` and return all of them as top logprobs when the server allows that
    many (--max-logprobs); otherwise ask for `ids` explicitly with logprob_token_ids (chunks of 128). The first path also
    works with speculative decoding; logprob_token_ids does not in this vLLM build. Both give the same distribution."""
    return ids if len(ids) <= S["max_logprobs"] else None


async def _logprobs(ctx, content, has_img, ids, model, allowed=None):
    """One-token read-out with a fallback to the other read-out path if the first one fails."""
    try:
        return await _logprobs_once(ctx, content, has_img, ids, model, allowed)
    except Upstream:
        raise
    except Exception:
        other = None if allowed is not None else (ids if len(ids) <= S["max_logprobs"] else None)
        if other is None and allowed is None:
            raise
        return await _logprobs_once(ctx, content, has_img, ids, model, other)


async def _logprobs_once(ctx, content, has_img, ids, model, allowed=None):
    """One-token read-out: {token_id: logprob} for `ids` after the content (text or chat parts)."""
    lp = {}
    step = CHUNK if allowed is None else len(ids)
    for i in range(0, len(ids), step):
        chunk = ids[i: i + step]
        kw = dict(logprob_token_ids=chunk) if allowed is None else dict(allowed_token_ids=allowed)
        if has_img:
            r = ChatCompletionRequest(model=model, messages=[{"role": "user", "content": content}], chat_template=S["raw"],
                                      add_generation_prompt=False, logprobs=True, top_logprobs=len(chunk) if allowed else 1, **kw, **READ)
            out = _check(await ctx.raw.app.state.openai_serving_chat.create_chat_completion(r, ctx.raw)); ctx.count(out)
            lp.update({int(t.token.split(":")[1]): t.logprob for t in out.choices[0].logprobs.content[0].top_logprobs})
        else:
            text = "".join(c["text"] for c in content)
            r = CompletionRequest(model=model, prompt=text, logprobs=len(chunk) if allowed else 1, **kw, **READ)
            out = _check(await ctx.raw.app.state.openai_serving_completion.create_completion(r, ctx.raw)); ctx.count(out)
            lp.update({int(k.split(":")[1]): v for k, v in out.choices[0].logprobs.top_logprobs[0].items()})
        if allowed is not None:
            break
    return lp


# ------------------------------------------------------------------ System 1
async def s1_pass(ctx, kind, parts, has_img, question, opts):
    """One System 1 pass over `opts` (<=256). Returns probabilities aligned with opts."""
    head, temps, prof = S["head"], S["temps"], S["prof"]
    lo, hi = head["slots"]["ranges"][kind]
    if kind == "choice":
        n = len(opts); ids = S["label_ids"][:n]
        bias = [head["bias"][lo + i] if lo + i < hi else 0.0 for i in range(n)]
        lines = [f"{S['labels'][i]}) {o}" for i, o in enumerate(opts)]
    else:
        ids = head["verbalizer_ids"][lo:hi]; bias = head["bias"][lo:hi]; lines = opts
    content = ([{"type": "text", "text": f"{prof['prefix']}[kind] {kind}\n[state] "}] + parts +
               [{"type": "text", "text": f"\n[question] {question}\n[options]\n" + "\n".join(lines) + "\n[decision]:"}])
    lp = await _logprobs(ctx, content, has_img, ids, LORA, allowed=_readout_allowed(ids))
    return _softmax([(max(lp.get(t, -1e9), -1e9) + b) / temps[kind] for t, b in zip(ids, bias)])


def _groups(n, k=16):
    g = math.ceil(n / k); base, extra = divmod(n, g); out, i = [], 0
    for j in range(g):
        size = base + (1 if j < extra else 0); out.append(list(range(i, i + size))); i += size
    return out


async def s1_dist(ctx, kind, parts, has_img, question, opts, strategy):
    if kind != "choice" or len(opts) <= 16 or strategy == "single":
        return await s1_pass(ctx, kind, parts, has_img, question, opts)
    n = len(opts)
    if strategy == "permute":
        import random
        rng = random.Random(zlib.crc32(question.encode()))
        orders = [list(range(n))] + [rng.sample(range(n), n) for _ in range(3)]
        res = await asyncio.gather(*(s1_pass(ctx, kind, parts, has_img, question, [opts[i] for i in o]) for o in orders))
        p = [0.0] * n
        for o, r in zip(orders, res):
            for i, v in zip(o, r):
                p[i] += v / len(orders)
        return p
    groups = _groups(n)  # tournament: groups of <=16 in the given order (in parallel), then a final of 16
    parts_g = await asyncio.gather(*(s1_pass(ctx, kind, parts, has_img, question, [opts[i] for i in g]) for g in groups))
    in_group = {o: p for g, ps in zip(groups, parts_g) for o, p in zip(g, ps)}
    chosen = [max(g, key=lambda o: (in_group[o], -o)) for g in groups]
    rest = sorted((o for g in groups for o in g if o not in set(chosen)), key=lambda o: (-in_group[o], o))
    fin = sorted(chosen + rest[: max(0, 16 - len(chosen))])
    final = dict(zip(fin, await s1_pass(ctx, kind, parts, has_img, question, [opts[i] for i in fin])))
    group_of = {o: gi for gi, g in enumerate(groups) for o in g}
    share = [0.0] * len(groups); cap = [0.0] * len(groups)
    for f in fin:
        share[group_of[f]] += final[f]; cap[group_of[f]] += in_group[f]
    among = sum(a * b for a, b in zip(share, cap))
    p = [final[o] * among if o in final else share[group_of[o]] * in_group[o] for o in range(n)]
    s = sum(p)
    return [x / s for x in p]


# ------------------------------------------------------------------ System 2 (thinking)
def _think_frame(kwargs):
    """Chat-template prefix / suffix around the user turn, thinking on, with the caller's template kwargs (cached)."""
    key = json.dumps(kwargs or {}, sort_keys=True)
    if key not in S["think_cache"]:
        kw = {k: v for k, v in (kwargs or {}).items() if k != "enable_thinking"}
        chat = S["tok"].apply_chat_template([{"role": "user", "content": PLACEHOLDER}], tokenize=False, add_generation_prompt=True,
                                            enable_thinking=True, **kw)
        S["think_cache"][key] = tuple(chat.split(PLACEHOLDER))
    return S["think_cache"][key]


async def s2_dist(ctx, kind, parts, has_img, question, opts, budget, want_text, tmpl_kwargs=None):
    """The base model thinks over the same input; the answer-letter distribution is read after the thinking channel."""
    shown = ["Yes (true)", "No (false)"] if kind == "noul" else opts
    labs = S["ans_labels"][: len(shown)]
    body = "\n".join(f"({l}) {o}" for (l, _), o in zip(labs, shown))
    tail = (f"\n\nQuestion: {question}\n\nOptions:\n{body}\n\n"
            "Think it through carefully, then give your final answer on the last line in the form: Answer: (X)")
    pre, post = _think_frame(tmpl_kwargs)
    user = [{"type": "text", "text": pre}] + parts + [{"type": "text", "text": tail + post}]
    seed = zlib.crc32((question + body).encode()) & 0x7FFFFFFF
    t0 = time.time()
    gen = ChatCompletionRequest(model=S["model"], messages=[{"role": "user", "content": user}], chat_template=S["raw"],
                                add_generation_prompt=False, add_special_tokens=False, max_tokens=budget, seed=seed,  # None: up to the context window
                                stop_token_ids=[S["end_think_id"]], skip_special_tokens=False)
    out = _check(await ctx.raw.app.state.openai_serving_chat.create_chat_completion(gen, ctx.raw)); ctx.count(out)
    thought = out.choices[0].message.content or ""
    finished = out.choices[0].finish_reason == "stop"
    ntok = out.usage.completion_tokens if out.usage else None
    read = user + [{"type": "text", "text": thought + S["prof"]["end_think"] + S["prof"]["after_think"] + "Answer: ("}]
    ids = [t for _, t in labs]
    lp = await _logprobs(ctx, read, True, ids, S["model"], allowed=_readout_allowed(ids))
    p = _softmax([max(lp.get(t, -1e9), -1e9) for t in ids])
    if kind == "noul":
        p = [p[1], p[0]]  # (A) yes / (B) no -> [P(false), P(true)]
    info = {"think_tokens": ntok, "think_seconds": round(time.time() - t0, 3), "finished_within_budget": finished}
    if want_text:
        info["reasoning"] = thought.replace("<|channel>thought\n", "").strip()
    return p, info


def _fold(p1, p2, w):
    return [(1 - w) * x + w * y for x, y in zip(p1, p2)]


# ------------------------------------------------------------------ routes
router = APIRouter()


@router.get("/v1/decide/info")
async def info():
    p = S["prof"]
    return {"protocol": PROTOCOL, "model": S["model"], "model_type": S["model_type"], "max_options": len(S["labels"]),
            "native_choice_options": len(S["head"]["slots"]["verbalizers"]) - 8, "temperatures": S["temps"],
            "defaults": {"strategy": p["strategy"], "thinking": p["thinking"], "threshold": p["threshold"], "mix": p["mix"],
                         "think_budget": None},
            "reasoning_controls": {"enable_thinking": True, "reasoning_effort": ["xhigh", "medium", "low"] if S["has_effort"] else None,
                                   "think_budget": "max thinking tokens (default: up to the context window)"}}


@router.post("/v1/decide")
async def decide(req: DecideRequest, raw: Request):
    t0 = time.time()
    if req.kind == "choice":
        opts = req.options or []
        if not 2 <= len(opts) <= len(S["labels"]):
            return _err(f"choice needs 2-{len(S['labels'])} options, got {len(opts)}")
    else:
        opts = ["false", "true"] if req.kind == "noul" else [str(i) for i in range(6)]
    prof = S["prof"]
    thinking = prof["thinking"] if req.thinking == "default" else req.thinking
    if req.kind == "score":
        if req.thinking in ("auto", "on"):
            return _err("thinking is supported for noul and choice")
        thinking = "off"
    tmpl = dict(req.chat_template_kwargs or {})
    if req.reasoning_effort is not None:
        if not S["has_effort"]:
            return _err("this base model has no reasoning_effort setting; control the reasoning length with think_budget")
        tmpl["reasoning_effort"] = req.reasoning_effort
    strategy = prof["strategy"] if req.strategy == "auto" else req.strategy
    tau = prof["threshold"] if req.threshold is None else req.threshold
    parts, has_img = _parts(req.state)
    ctx = Ctx(raw)
    try:
        if req.system2_only:
            if req.kind == "score":
                return _err("system2_only is supported for noul and choice")
            p2, think = await s2_dist(ctx, req.kind, parts, has_img, req.question, opts, req.think_budget, req.return_reasoning, tmpl)
            k = max(range(len(p2)), key=p2.__getitem__)
            return {"kind": req.kind, "options": opts, "probabilities": p2, "choice_index": k, "choice": opts[k], "system": 2,
                    "model": S["model"], "thinking": {"used": True, "budget": req.think_budget, **think},
                    "usage": {"prompt_tokens": ctx.prompt_tokens, "completion_tokens": ctx.completion_tokens},
                    "elapsed_seconds": time.time() - t0, "num_model_requests": ctx.requests}
        p1 = await s1_dist(ctx, req.kind, parts, has_img, req.question, opts, strategy)
        probs, think = p1, None
        if thinking == "on" or (thinking == "auto" and max(p1) < tau):
            p2, think = await s2_dist(ctx, req.kind, parts, has_img, req.question, opts, req.think_budget, req.return_reasoning, tmpl)
            probs = _fold(p1, p2, prof["mix"])
            think = {"used": True, **think}
            if req.debug:
                think.update(system1=p1, system2=p2)
        elif thinking != "off":
            think = {"used": False}
            if req.debug:
                think["system1"] = p1
    except Upstream as e:
        return JSONResponse(e.resp.model_dump(), status_code=e.resp.error.code)
    k = max(range(len(probs)), key=probs.__getitem__)
    native = req.kind != "choice" or len(opts) <= 16
    resp = {"kind": req.kind, "effective_kind": req.kind, "options": opts, "probabilities": probs, "choice_index": k,
            "choice": opts[k], "adaptation": "native" if native else f"{strategy}", "protocol": PROTOCOL, "model": S["model"],
            "usage": {"prompt_tokens": ctx.prompt_tokens, "completion_tokens": ctx.completion_tokens,
                      "total_tokens": ctx.prompt_tokens + ctx.completion_tokens},
            "elapsed_seconds": time.time() - t0, "num_model_requests": ctx.requests}
    if think is not None:
        resp["thinking"] = {"mode": thinking, "threshold": tau, "budget": req.think_budget, **({"chat_template_kwargs": tmpl} if tmpl else {}), **think}
    return resp


_build_app = entry.build_app


def build_app(args, *a, **kw):
    app = _build_app(args, *a, **kw)
    setup(args)
    app.include_router(router)
    return app


entry.build_app = build_app

if __name__ == "__main__":
    entry.main()
