"""E2 kill test: does keeping visual sinks in FP16 matter end-to-end under 4-bit quantization?

Model: Qwen3-VL (Thinking), V* multiple-choice. Only the language model is fake-quantized;
weights stay BF16. Two settings, run separately:
  kv  : K and V of every position, per token and per KV head over head_dim
        (int4_tok = symmetric INT4, mxfp4 = 32-blocks, nvfp4 = 16-blocks)
  act : inputs of every LM linear layer (q/k/v/o/gate/up/down), per token
Protection (equal budget k = number of visual sinks of the image, positions kept in FP16):
  none, vsink (visual sinks, E0 x100 criterion at the peak layer), att (k most-attended
  non-sink image tokens at the E0 peak layer), tsink (the text sink only, budget 1),
  rand#r (k random non-sink image tokens, N_RAND draws).
Metric: one prefill per configuration; the prompt ends with '\\boxed{' so the next token is the
answer letter. We read the distribution over the option letters: accuracy, log-prob of the
correct letter, and KL(FP16 || quantized) over the options.

Usage: python e2_protect.py --model-dir M --model-tag T --vstar-root V --e0-run-dir RUN/e0 \
           --formats int4_tok,mxfp4 --settings kv,act --out-dir OUT [--limit N]
"""
import argparse
import json
import os
import re
import sys
import time

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AttentionInterface, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.integrations.sdpa_attention import sdpa_attention_forward

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_sink_stats import pick_pope, pick_vstar  # noqa: E402

from sinkquant import e2summary  # noqa: E402
from sinkquant import quant as Q  # noqa: E402
from sinkquant import sinkstats as S  # noqa: E402

N_RAND = 3
STATE = {"kv_fmt": None, "act_fmt": None, "protect": None, "kv_err": [], "act_err": []}


def fq_last_dim(x, fmt):
    xf = x.float()
    if fmt == "nvfp4":
        q = Q.nvfp4(xf, tensor_amax=xf.abs().amax())
    else:
        q = Q.FORMATS[fmt](xf)
    return q


def quantize_positions(x, fmt, protect, pos_dim):
    """Fake-quantize x along its last dim; positions in `protect` (along pos_dim) stay exact."""
    q = fq_last_dim(x, fmt)
    if protect is not None and len(protect):
        idx = [slice(None)] * x.dim()
        idx[pos_dim] = protect
        q[tuple(idx)] = x[tuple(idx)].float()
    return q.to(x.dtype)


def kvq_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    fmt = STATE["kv_fmt"]
    if fmt is not None:
        kq = quantize_positions(key, fmt, STATE["protect"], 2)            # [B, Hkv, L, D]
        vq = quantize_positions(value, fmt, STATE["protect"], 2)
        if len(STATE["kv_err"]) < 4:
            STATE["kv_err"].append(float(((kq.float() - key.float()).norm() / key.float().norm()).item()))
        key, value = kq, vq
    return sdpa_attention_forward(module, query, key, value, attention_mask,
                                  scaling=scaling, dropout=dropout, **kwargs)


AttentionInterface.register("kvq_sdpa", kvq_attention)


def act_hook(mod, inp):
    fmt = STATE["act_fmt"]
    if fmt is None:
        return None
    x = inp[0]                                                           # [B, L, d]
    xq = quantize_positions(x, fmt, STATE["protect"], 1)
    if len(STATE["act_err"]) < 4:
        STATE["act_err"].append(float(((xq.float() - x.float()).norm() / x.float().norm()).item()))
    return (xq,) + tuple(inp[1:])


def build_inputs(proc, cfg, img, question):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text += "\n</think>\n\n" if text.rstrip().endswith("<think>") else "<think>\n\n</think>\n\n"
    text += "\\boxed{"
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    ids = inputs["input_ids"][0].cpu().numpy()
    img_idx = np.nonzero(ids == cfg.image_token_id)[0]
    return inputs, ids, img_idx, text


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--dataset", choices=["vstar", "pope"], default="vstar")
    ap.add_argument("--vstar-root")
    ap.add_argument("--pope-root")
    ap.add_argument("--e0-run-dir", required=True)
    ap.add_argument("--formats", default="int4_tok,mxfp4")
    ap.add_argument("--settings", default="kv,act")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--skip-done", default="", help="comma-separated per_item.jsonl files whose items are skipped")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    assert torch.cuda.device_count() == 1
    formats = args.formats.split(",")
    settings = args.settings.split(",")

    e0 = json.load(open(os.path.join(args.e0_run_dir, "summary.json")))
    assert e0["model_tag"] == args.model_tag
    peak = e0["criteria"]["x100"]["peak_layer"]
    meta = {m["image_id"]: m for m in map(json.loads, open(os.path.join(args.e0_run_dir, "meta.jsonl")))}

    proc = AutoProcessor.from_pretrained(args.model_dir)
    tok = proc.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
    model.model.language_model.config._attn_implementation = "kvq_sdpa"
    cfg = model.config
    for layer in model.model.language_model.layers:
        for m in [layer.self_attn.q_proj, layer.self_attn.k_proj, layer.self_attn.v_proj, layer.self_attn.o_proj,
                  layer.mlp.gate_proj, layer.mlp.up_proj, layer.mlp.down_proj]:
            m.register_forward_pre_hook(act_hook)

    # answer tokens per option; an option's logit is the logsumexp over its variants
    variants = {L: [L] for L in "ABCD"}
    variants.update({"yes": ["yes", "Yes"], "no": ["no", "No"]})
    opt_ids = {}
    for o, vs in variants.items():
        ids_o = []
        for v in vs:
            t = tok.encode(v, add_special_tokens=False)
            assert len(t) == 1, (v, t)
            ids_o.append(t[0])
        opt_ids[o] = ids_o
    root = args.vstar_root if args.dataset == "vstar" else args.pope_root
    picker = pick_vstar if args.dataset == "vstar" else pick_pope
    items = [it for it in picker(root, 10_000) if it["image_id"] in meta]
    if args.limit:
        items = items[:args.limit]
    done = set()
    for f in filter(None, args.skip_done.split(",")):
        if os.path.exists(f):
            done |= {json.loads(l)["image_id"] for l in open(f)}
    items = [it for it in items[args.shard_index::args.num_shards] if it["image_id"] not in done]
    print(f"shard {args.shard_index}/{args.num_shards}: {len(items)} items to run, {len(done)} skipped as done", flush=True)
    configs = [("fp16", None, "none")]
    for st in settings:
        for fm in formats:
            for pr in ["none", "vsink", "att", "tsink"] + [f"rand#{r}" for r in range(N_RAND)]:
                configs.append((st, fm, pr))

    out_f = open(os.path.join(args.out_dir, "per_item.jsonl"), "w")
    t0 = time.time()
    for k_it, it in enumerate(items):
        img = Image.open(os.path.join(root, it["file"])).convert("RGB")
        inputs, ids, img_idx, text = build_inputs(proc, cfg, img, it["question"])
        options = ["yes", "no"] if args.dataset == "pope" else re.findall(r"^\(([A-D])\)", it["question"], flags=re.M)
        assert options and it["label"] in options, (options, it["label"])
        z = np.load(os.path.join(args.e0_run_dir, "per_image", f"{it['image_id']}.npz"))
        assert z["x_uniform"].shape[1] == len(img_idx), (z["x_uniform"].shape, len(img_idx))
        xu = z["x_uniform"][peak].astype(np.float64)
        sink_l = S.sink_mask(xu / len(xu), z["inside"])
        k = int(sink_l.sum())
        nonsink = np.nonzero(~sink_l)[0]
        att_l = nonsink[np.argsort(-xu[nonsink])[:k]]
        sets = {"none": np.array([], dtype=np.int64), "vsink": img_idx[sink_l], "att": img_idx[att_l],
                "tsink": np.array([int(meta[it["image_id"]]["t_star"])])}
        rng = np.random.default_rng([args.seed, int(it["image_id"].lstrip("qp"))])   # per-item, shard-independent
        for r in range(N_RAND):
            sets[f"rand#{r}"] = img_idx[rng.choice(nonsink, size=k, replace=False)] if k else np.array([], dtype=np.int64)
        rec = {"image_id": it["image_id"], "category": it["category"], "label": it["label"], "options": options,
               "k": k, "n_img_tokens": int(len(img_idx)), "res": {}}
        for st, fm, pr in configs:
            STATE["kv_fmt"] = fm if st == "kv" else None
            STATE["act_fmt"] = fm if st == "act" else None
            STATE["protect"] = torch.as_tensor(sets[pr], device="cuda", dtype=torch.long)
            logits = model(**inputs, use_cache=False, logits_to_keep=1).logits[0, -1].float()
            lo = torch.stack([torch.logsumexp(logits[opt_ids[o]], 0) for o in options])
            p = torch.softmax(lo, 0).cpu().numpy()
            rec["res"][f"{st}|{fm}|{pr}"] = {"p": p.round(6).tolist(),
                                              "pred": options[int(p.argmax())],
                                              "top_is_letter": bool(int(logits.argmax()) in [i for o in options for i in opt_ids[o]])}
        STATE.update(kv_fmt=None, act_fmt=None, protect=None)
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        if k_it % 10 == 0:
            fp = rec["res"]["fp16|None|none"]
            print(f"[{k_it}/{len(items)}] {time.time() - t0:.0f}s k={k} n_img={len(img_idx)} fp16 pred={fp['pred']} "
                  f"label={it['label']} top_is_letter={fp['top_is_letter']} kv_err={STATE['kv_err'][:2]} act_err={STATE['act_err'][:2]}",
                  flush=True)
    out_f.close()
    recs = [json.loads(l) for l in open(os.path.join(args.out_dir, "per_item.jsonl"))]
    if recs:
        summary = e2summary.summarize_records(recs, settings, formats, N_RAND)
        summary.update(model_tag=args.model_tag, transformers=transformers.__version__,
                       shard=[args.shard_index, args.num_shards],
                       kv_rel_err_sample=STATE["kv_err"], act_rel_err_sample=STATE["act_err"])
        json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
        print(json.dumps({k: v for k, v in summary.items() if k != "table"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
