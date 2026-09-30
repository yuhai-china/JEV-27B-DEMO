"""Multimodal RewardBench 2 (MMRB2, Meta, Dec 2025 / Jan 2026): JEV-27B-VL System 1 as a judge on all four tasks.

text-to-image, image editing, interleaved generation and multimodal reasoning; 1,000 expert-annotated pairs each.
Both presentation orders are asked and averaged. Needs the multimodal server (bash common/serve_jev27b_mm.sh).

    python mmrb2.py                 # all four tasks (~4 GB download, ~20 min on one GPU)
    ONLY=t2i python mmrb2.py        # one task
"""
import concurrent.futures as cf, glob, io, json, os, re, sys, time
import numpy as np, pandas as pd
from PIL import Image
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "common"))
import jev_client as j

D = os.path.join(os.environ.get("MMRB2_DIR", os.path.expanduser("~/.cache/jev-demo/mmrb2")), "img"); os.makedirs(D, exist_ok=True)
from huggingface_hub import snapshot_download
SNAP = snapshot_download("rl-research/multimodal-rewardbench-2", repo_type="dataset")
cut = lambda s, n=3500: s if len(s) <= n else s[:n] + " …[truncated]"


def save(img, key, size=768):
    p = f"{D}/{key}.jpg"
    if not os.path.exists(p):
        im = Image.open(io.BytesIO(img["bytes"])).convert("RGB"); im.thumbnail((size, size)); im.save(p, quality=90)
    return p


def t2i_parts(r, flip):
    a = save(r.response_a_images[0], f"{r.pair_id}_a"); b = save(r.response_b_images[0], f"{r.pair_id}_b")
    x, y = (b, a) if flip else (a, b)
    return ["Text-to-image prompt: " + cut(r.prompt_text, 1500) + "\n\n[Image A]\n", {"image": x}, "\n\n[Image B]\n", {"image": y}]


def reasoning_parts(r, flip, level=0):
    size = 768 if level < 2 else 448
    pim = [save(im, f"{r.pair_id}_p{k}_{size}", size) for k, im in enumerate(list(r.prompt_images)[:4])]
    segs = re.split(r"(<image_\d+>)", r.prompt_text)
    parts, used = ["Request:\n" if level >= 0 else "", 0][:1], 0
    for s in segs:
        m = re.fullmatch(r"<image_(\d+)>", s)
        if m:
            k = int(m.group(1))
            if k < len(pim):
                parts.append({"image": pim[k]}); used += 1
        elif s.strip():
            parts.append(cut(s, 2500))
    budget = 0 if level >= 1 else 8 - used
    def resp(tag, text, ims):
        out = [f"\n\n[{tag}]\n" + cut(text, 3500 if level == 0 else 2000)]
        for k, im in enumerate(list(ims)[: min(2, budget // 2)]):
            out += ["\n(image from this response)\n", {"image": save(im, f"{r.pair_id}_{tag[-1]}{k}_{hash(text) % 10**6}")}]
        return out
    A, B = (r.response_b_text, r.response_b_images), (r.response_a_text, r.response_a_images)
    if not flip:
        A, B = B, A
    return parts + resp("Response A", A[0], A[1] if A[1] is not None else []) + resp("Response B", B[0], B[1] if B[1] is not None else [])


TASKS_DEF = None
TASKS = {"t2i": (t2i_parts, "Which generated image follows the prompt better and has higher overall quality (faithful to every detail of the prompt, "
                              "no artifacts)?", ["Image A", "Image B"]),
         "edit": (None, "Which edited image follows the editing instruction better, changes only what was asked, and looks natural?",
                  ["Response A", "Response B"]),
         "interleaved": (None, "Which response fulfils the request better: correct, helpful, and with high-quality images that are consistent "
                               "with the text and with each other?", ["Response A", "Response B"]),
         "reasoning": (reasoning_parts, "Which response solves the task better: correct final answer and accurate, well-grounded reasoning "
                                        "about the images?", ["Response A", "Response B"])}
out = {}
ONLY = os.environ.get("ONLY")
for task, (fn, Q, OPTS) in TASKS.items():
    fn = fn or reasoning_parts
    if ONLY and task != ONLY:
        continue
    df = pd.concat([pd.read_parquet(f) for f in sorted(glob.glob(f"{SNAP}/{task}/*.parquet"))]).reset_index(drop=True)
    jobs = []
    for r in df.itertuples():
        jobs += [(r, False), (r, True)]
    n_fallback = [0]
    def run_one(job):
        r, flip = job
        for level in (0, 1, 2):
            try:
                parts = fn(r, flip, level) if task != "t2i" else fn(r, flip)
                return j.decide_mm("choice", parts, Q, OPTS)
            except Exception:
                n_fallback[0] += 1
                if task == "t2i":
                    raise
        return {OPTS[0]: 0.5, OPTS[1]: 0.5}
    t0 = time.time()
    with cf.ThreadPoolExecutor(16) as ex:
        res = list(ex.map(run_one, jobs))
    print(task, "fallbacks", n_fallback[0], flush=True)
    p_a = np.array([(res[2 * k][OPTS[0]] + res[2 * k + 1][OPTS[1]]) / 2 for k in range(len(df))])
    acc = float(((p_a > 0.5) == (df.chosen.str.lower() == "a")).mean())
    agree = float(np.mean([(res[2 * k][OPTS[0]] > 0.5) == (res[2 * k + 1][OPTS[1]] > 0.5) for k in range(len(df))]))
    out[task] = {"n": len(df), "accuracy": acc, "order_agreement": agree, "seconds": time.time() - t0}
    print(task, out[task], flush=True)
    pd.DataFrame({"pair_id": df.pair_id, "chosen": df.chosen, "p_a": p_a, "source": df.prompt_source}).to_json(os.path.join(HERE, f"mmrb2_{task}.json"), orient="records")
json.dump(out, open(os.path.join(HERE, f"mmrb2_summary_{ONLY or 'all'}.json"), "w"), indent=1)
