"""E3 round 1: does per-token INT4 activation quantization make the model 'blind'?

Model: Qwen3-VL (Thinking). Only the inputs of LM linear layers (q/k/v/o/gate/up/down) are
fake-quantized; weights and KV stay BF16. One prefill per configuration; the prompt ends with
'\\boxed{' and we read the distribution over the answer options (A-D for V*, yes/no for POPE).

Configurations (image = the real image unless marked 'gray'):
  fp16                 no quantization
  int4_all             per-token INT4 on every row
  int4_visonly         INT4 on image-token rows only (text rows exact)
  int4_textonly        INT4 on text rows only (image rows exact)
  int4_had             INT4 after a randomized Hadamard rotation of each layer input (QuaRot-style,
                       x -> Q(x H) H^T; weights unrotated, so it is exact without quantization)
  int4_top{c}          INT4 with the c largest-|x| entries of each row kept exact and the row scale
                       set by the rest (c = 1, 4, 16)
  mxfp4_all            MXFP4 on every row (reference)
  gray|fp16            mid-gray image of the same size (same token layout), no quantization
  gray|int4_all        mid-gray image, INT4 on every row
Round 2 (--round 2): fp16; {mxfp4, nvfp4, had_int3 (randomized Hadamard + per-token INT3)} x
  {all, visonly, textonly}; int4_had; mxfp4_had (MXFP4 after a full randomized Hadamard rotation).
Diagnostics (fp16 pass, inputs of q_proj and down_proj, every layer): share of rows whose max-|x|
channel is one of the text sink's top-4 channels (image rows; text rows without the sink), median
max|x| / mean|x|, and the share of entries INT4 rounds to zero, for image and text rows.

Usage: python e3_blind.py --model-dir M --model-tag T --dataset pope --pope-root P \
           --e0-run-dir RUN/e0 --out-dir OUT [--limit N] [--shard-index i --num-shards n]
"""
import argparse
import functools
import json
import os
import re
import sys
import time

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_sink_stats import pick_pope, pick_vstar  # noqa: E402

from sinkquant import quant as Q  # noqa: E402

TOPC = [1, 4, 16]
STATE = {"mode": None, "protect": None, "diag": None, "sink_pos": None, "img_mask": None, "err": []}
HAD = {}


def had(d):
    if d not in HAD:
        HAD[d] = Q.random_hadamard(d, seed=0, device="cuda")
    return HAD[d]


def fq(x, mode):
    xf = x.float()
    if mode == "int4":
        return Q.int4_per_token(xf)
    if mode == "had":
        h = had(xf.shape[-1])
        return Q.int4_per_token(xf @ h) @ h.T
    if mode.startswith("top"):
        return Q.int4_keep_topc(xf, int(mode[3:]))
    if mode == "mxfp4":
        return Q.mxfp4(xf)
    if mode == "nvfp4":
        return Q.nvfp4(xf, tensor_amax=xf.abs().amax())
    if mode == "had3":
        h = had(xf.shape[-1])
        return Q.int_per_token(xf @ h, 3) @ h.T
    if mode == "mxhad":
        h = had(xf.shape[-1])
        return Q.mxfp4(xf @ h) @ h.T
    raise ValueError(mode)


def diag_stats(x):
    """x: [L, d] float. Returns 6 numbers (image rows / text rows)."""
    a = x.abs()
    img = STATE["img_mask"]
    sp = STATE["sink_pos"]
    top4 = a[sp].topk(4).indices
    hit = (a.argmax(-1)[:, None] == top4[None, :]).any(-1).float()
    txt = ~img
    txt[sp] = False
    ratio = a.amax(-1) / a.mean(-1).clamp(min=1e-12)
    zf = Q.int4_zero_frac(x)
    return [hit[img].mean().item(), hit[txt].mean().item(),
            ratio[img].median().item(), ratio[txt].median().item(),
            zf[img].mean().item(), zf[txt].mean().item()]


def act_hook(mod, inp, layer=None, name=None):
    x = inp[0]                                                           # [B, L, d]
    if STATE["diag"] is not None and name in ("q", "down"):
        STATE["diag"][name][layer] = diag_stats(x[0].float())
    mode = STATE["mode"]
    if mode is None:
        return None
    q = fq(x, mode)
    if STATE["protect"] is not None and len(STATE["protect"]):
        q[:, STATE["protect"]] = x[:, STATE["protect"]].float()
    if len(STATE["err"]) < 4:
        STATE["err"].append(float(((q - x.float()).norm() / x.float().norm()).item()))
    return (q.to(x.dtype),) + tuple(inp[1:])


def build_inputs(proc, cfg, img, question):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text += "\n</think>\n\n" if text.rstrip().endswith("<think>") else "<think>\n\n</think>\n\n"
    text += "\\boxed{"
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    ids = inputs["input_ids"][0].cpu().numpy()
    img_idx = np.nonzero(ids == cfg.image_token_id)[0]
    return inputs, ids, img_idx


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--dataset", choices=["vstar", "pope"], default="pope")
    ap.add_argument("--vstar-root")
    ap.add_argument("--pope-root")
    ap.add_argument("--e0-run-dir", required=True, help="E0 run of the same model/dataset (for the text-sink position)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--skip-done", default="", help="comma-separated per_item.jsonl files whose items are skipped")
    ap.add_argument("--round", type=int, choices=[1, 2], default=1, help="2 = MXFP4/NVFP4/rotated-INT3 modality split")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    assert torch.cuda.device_count() == 1

    e0 = json.load(open(os.path.join(args.e0_run_dir, "summary.json")))
    assert e0["model_tag"] == args.model_tag
    meta = {m["image_id"]: m for m in map(json.loads, open(os.path.join(args.e0_run_dir, "meta.jsonl")))}

    proc = AutoProcessor.from_pretrained(args.model_dir)
    tok = proc.tokenizer
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
    cfg = model.config
    layers = model.model.language_model.layers
    for li, layer in enumerate(layers):
        for name, m in [("q", layer.self_attn.q_proj), ("k", layer.self_attn.k_proj), ("v", layer.self_attn.v_proj),
                        ("o", layer.self_attn.o_proj), ("gate", layer.mlp.gate_proj), ("up", layer.mlp.up_proj),
                        ("down", layer.mlp.down_proj)]:
            m.register_forward_pre_hook(functools.partial(act_hook, layer=li, name=name))

    variants = {L: [L] for L in "ABCD"}
    variants.update({"yes": ["yes", "Yes"], "no": ["no", "No"]})
    opt_ids = {}
    for o, vs in variants.items():
        opt_ids[o] = []
        for v in vs:
            t = tok.encode(v, add_special_tokens=False)
            assert len(t) == 1, (v, t)
            opt_ids[o].append(t[0])
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
    print(f"shard {args.shard_index}/{args.num_shards}: {len(items)} items", flush=True)

    # (image, mode, protect)  protect: positions kept exact
    configs = [("real", None, "none"), ("real", "int4", "none"), ("real", "int4", "text"), ("real", "int4", "img"),
               ("real", "had", "none")] + [("real", f"top{c}", "none") for c in TOPC] + \
              [("real", "mxfp4", "none"), ("gray", None, "none"), ("gray", "int4", "none")]
    names = {("real", None, "none"): "fp16", ("real", "int4", "none"): "int4_all",
             ("real", "int4", "text"): "int4_visonly", ("real", "int4", "img"): "int4_textonly",
             ("real", "had", "none"): "int4_had", ("real", "mxfp4", "none"): "mxfp4_all",
             ("gray", None, "none"): "gray|fp16", ("gray", "int4", "none"): "gray|int4_all"}
    names.update({("real", f"top{c}", "none"): f"int4_top{c}" for c in TOPC})
    if args.round == 2:   # practical formats: where does the residual damage sit, and does rotation remove it?
        configs, names = [("real", None, "none")], {("real", None, "none"): "fp16"}
        for mode, tag in [("mxfp4", "mxfp4"), ("nvfp4", "nvfp4"), ("had3", "had_int3")]:
            for pr, sfx in [("none", "all"), ("text", "visonly"), ("img", "textonly")]:
                configs.append(("real", mode, pr))
                names[("real", mode, pr)] = f"{tag}_{sfx}"
        for mode, nm in [("had", "int4_had"), ("mxhad", "mxfp4_had")]:
            configs.append(("real", mode, "none"))
            names[("real", mode, "none")] = nm

    out_f = open(os.path.join(args.out_dir, "per_item.jsonl"), "w")
    t0 = time.time()
    for k_it, it in enumerate(items):
        img = Image.open(os.path.join(root, it["file"])).convert("RGB")
        gray = Image.new("RGB", img.size, (128, 128, 128))
        ins = {}
        for tag, im in [("real", img), ("gray", gray)]:
            ins[tag] = build_inputs(proc, cfg, im, it["question"])
        inputs, ids, img_idx = ins["real"]
        assert np.array_equal(ins["gray"][2], img_idx), "gray image changed the token layout"
        options = ["yes", "no"] if args.dataset == "pope" else re.findall(r"^\(([A-D])\)", it["question"], flags=re.M)
        assert options and it["label"] in options, (options, it["label"])
        sink_pos = int(meta[it["image_id"]]["t_star"])
        assert sink_pos not in set(img_idx.tolist())
        n = len(ids)
        img_mask = torch.zeros(n, dtype=torch.bool, device="cuda")
        img_mask[torch.as_tensor(img_idx, device="cuda")] = True
        sets = {"none": torch.zeros(0, dtype=torch.long, device="cuda"),
                "img": torch.nonzero(img_mask)[:, 0], "text": torch.nonzero(~img_mask)[:, 0]}
        rec = {"image_id": it["image_id"], "category": it.get("category"), "label": it["label"], "options": options,
               "n_tokens": n, "n_img_tokens": int(len(img_idx)), "sink_pos": sink_pos, "res": {}}
        for cfg_t in configs:
            tag, mode, pr = cfg_t
            STATE.update(mode=mode, protect=sets[pr])
            if cfg_t == ("real", None, "none"):
                L = len(layers)
                STATE.update(diag={"q": [None] * L, "down": [None] * L}, sink_pos=sink_pos, img_mask=img_mask)
            logits = model(**ins[tag][0], use_cache=False, logits_to_keep=1).logits[0, -1].float()
            if STATE["diag"] is not None:
                rec["diag"] = {k: np.round(np.array(v), 4).tolist() for k, v in STATE["diag"].items()}
                STATE["diag"] = None
            lo = torch.stack([torch.logsumexp(logits[opt_ids[o]], 0) for o in options])
            p = torch.softmax(lo, 0).cpu().numpy()
            lsm = torch.log_softmax(logits, 0)
            top = int(logits.argmax())
            rec["res"][names[cfg_t]] = {"p": p.round(6).tolist(), "pred": options[int(p.argmax())],
                                        "top_is_option": bool(top in [i for o in options for i in opt_ids[o]]),
                                        "opt_mass": round(float(torch.logsumexp(lo, 0).sub(torch.logsumexp(logits, 0)).exp()), 6),
                                        "top_tok": tok.decode([top]), "top_p": round(float(lsm[top].exp()), 4)}
        STATE.update(mode=None, protect=None)
        out_f.write(json.dumps(rec) + "\n")
        out_f.flush()
        if k_it % 10 == 0:
            r = rec["res"]
            print(f"[{k_it}/{len(items)}] {time.time() - t0:.0f}s n={n} n_img={len(img_idx)} label={it['label']} "
                  + " ".join(f"{c}={r[c]['pred']}" for c in list(names.values())[:6])
                  + f" err={STATE['err'][:2]}", flush=True)
    out_f.close()
    json.dump({"model_tag": args.model_tag, "dataset": args.dataset, "transformers": transformers.__version__,
               "n_items": len(items), "shard": [args.shard_index, args.num_shards], "configs": list(names.values()),
               "act_rel_err_sample": STATE["err"]}, open(os.path.join(args.out_dir, "run_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
