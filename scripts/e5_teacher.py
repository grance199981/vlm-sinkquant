"""E5 stage 1: teacher-forced per-position effect of KV-cache quantization on a BF16 thinking chain.

For every item of the stage-0 output, the prompt (same image and question) is rebuilt and the BF16
generated ids are appended; one forward per configuration gives, at every generated position, the
next-token distribution. Only K/V are fake-quantized (per token and per KV head over head_dim);
weights and activations stay BF16. Trajectory divergence is excluded by construction.

Configurations (name: format, which positions are quantized, image):
  bf16                reference                                  real image
  bf16_gray           -                                          mid-gray image (same token layout)
  kv{4,3,2}_all       symmetric per-token INT-b on every position
  kv4had_all          INT4 after a randomized Hadamard rotation of head_dim (rotate, quantize, rotate back)
  kv{4,3}_img         only image positions quantized
  kv{4,3}_gen         only generated positions quantized (prompt exact)
  kv{4,3}_all_gray    as kv{4,3}_all, gray image
Per generated position (arrays saved per item as npz):
  kl_<cfg>      KL(bf16 || cfg) of the next-token distribution
  agree_<cfg>   argmax(cfg) == argmax(bf16)
  nll_<cfg>     -log p_cfg(generated token)
  ent_bf16      entropy of the BF16 next-token distribution (for KL normalization)
  dimg_<p>      KL(real || gray) at the same precision p in {bf16, kv4_all, kv3_all} = image dependence
  attn_<p>      share of attention on image tokens, per layer [L, n] (mean over heads), p in {bf16, kv4_all, kv3_all}

Usage: python e5_teacher.py --model-dir M --model-tag T --dataset {vstar,mathvista} --data-root D --gen-files g1.jsonl[,g2] --out-dir OUT
           [--limit N] [--shard-index i --num-shards n]
"""
import argparse
import json
import os
import sys
import tarfile
import time

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AttentionInterface, AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.integrations.sdpa_attention import sdpa_attention_forward

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e5_generate import build_prompt, ids_hash, pick  # noqa: E402

from sinkquant import quant as Q  # noqa: E402

STATE = {"mode": None, "protect": None, "probe": None, "rows": None, "img_idx": None, "err": []}
HAD = {}
CHUNK = 256


def kv_fq(x, mode):
    xf = x.float()
    if mode.startswith("had"):
        d = xf.shape[-1]
        if d not in HAD:
            HAD[d] = Q.random_hadamard(d, seed=0, device=xf.device)
        h = HAD[d]
        return Q.int_per_token(xf @ h, int(mode[3:])) @ h.T
    return Q.int_per_token(xf, int(mode[3:]))


def quant_kv(x, mode, protect):
    q = kv_fq(x, mode)
    if protect is not None and len(protect):
        q[:, :, protect] = x[:, :, protect].float()
    return q.to(x.dtype)


def e5_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    mode = STATE["mode"]
    if mode is not None:
        kq, vq = quant_kv(key, mode, STATE["protect"]), quant_kv(value, mode, STATE["protect"])
        if len(STATE["err"]) < 4:
            STATE["err"].append(float(((kq.float() - key.float()).norm() / key.float().norm()).item()))
        key, value = kq, vq
    if STATE["probe"] is not None:
        rows, img = STATE["rows"], STATE["img_idx"]
        rep = query.shape[1] // key.shape[1]
        kf = key.float().repeat_interleave(rep, dim=1)[0]                # [H, L, D]
        sc = scaling if scaling is not None else query.shape[-1] ** -0.5
        out = []
        for r0 in range(0, len(rows), CHUNK):
            rr = rows[r0:r0 + CHUNK]
            s = torch.einsum("hqd,hkd->hqk", query[0, :, rr].float(), kf) * sc
            causal = torch.arange(kf.shape[1], device=s.device)[None, :] > rr[:, None]
            s.masked_fill_(causal[None], float("-inf"))
            out.append(torch.softmax(s, -1)[:, :, img].sum(-1).mean(0))
        STATE["probe"][module.layer_idx] = torch.cat(out).cpu().numpy()
    return sdpa_attention_forward(module, query, key, value, attention_mask, scaling=scaling, dropout=dropout, **kwargs)


AttentionInterface.register("e5_sdpa", e5_attention)

CONFIGS = [  # name, mode, protect-set, image, probe
    ("bf16", None, "none", "real", True), ("bf16_gray", None, "none", "gray", False),
    ("kv4_all", "int4", "none", "real", True), ("kv3_all", "int3", "none", "real", True),
    ("kv2_all", "int2", "none", "real", False), ("kv4had_all", "had4", "none", "real", False),
    ("kv4_img", "int4", "nonimg", "real", False), ("kv4_gen", "int4", "prompt", "real", False),
    ("kv3_img", "int3", "nonimg", "real", False), ("kv3_gen", "int3", "prompt", "real", False),
    ("kv4_all_gray", "int4", "none", "gray", False), ("kv3_all_gray", "int3", "none", "gray", False),
]
GRAY_PAIRS = {"bf16": "bf16_gray", "kv4_all": "kv4_all_gray", "kv3_all": "kv3_all_gray"}


def compare(lm_head, h_ref, h, gen):
    """Per position KL(ref || h), argmax agreement, nll of the generated token under h."""
    kl, ag, nll = [], [], []
    for r0 in range(0, h.shape[0], 512):
        a = torch.log_softmax(lm_head(h_ref[r0:r0 + 512]).float(), -1)
        b = torch.log_softmax(lm_head(h[r0:r0 + 512]).float(), -1)
        kl.append((a.exp() * (a - b)).sum(-1))
        ag.append(a.argmax(-1) == b.argmax(-1))
        nll.append(-b.gather(1, gen[r0:r0 + 512, None])[:, 0])
    return [torch.cat(x).cpu().numpy() for x in (kl, ag, nll)]


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--dataset", choices=["vstar", "mathvista"], default="vstar")
    ap.add_argument("--data-root", "--vstar-root", dest="data_root", required=True)
    ap.add_argument("--gen-files", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--shard-index", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--max-pixels", type=int, default=0,
                    help="cap on image pixels (image_processor size longest_edge); 0 = model default")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "per_item"), exist_ok=True)
    assert torch.cuda.device_count() == 1

    gens = {}
    for f in args.gen_files.split(","):
        for line in open(f):
            r = json.loads(line)
            assert r["model_tag"] == args.model_tag
            gens.setdefault(r["image_id"], r)
    proc = AutoProcessor.from_pretrained(args.model_dir)
    if args.max_pixels:
        proc.image_processor.size["longest_edge"] = args.max_pixels
        proc.image_processor.max_pixels = args.max_pixels
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
    model.model.language_model.config._attn_implementation = "e5_sdpa"
    n_layers = model.config.text_config.num_hidden_layers
    lm_head = model.lm_head

    items = [it for it in pick(args.dataset, args.data_root) if it["image_id"] in gens]
    if args.limit:
        items = items[:args.limit]
    items = items[args.shard_index::args.num_shards]
    print(f"{args.model_tag} shard {args.shard_index}/{args.num_shards}: {len(items)} items", flush=True)
    meta_f = open(os.path.join(args.out_dir, "items.jsonl"), "w")
    t0 = time.time()
    for k, it in enumerate(items):
        g = gens[it["image_id"]]
        img = Image.open(os.path.join(args.data_root, it["file"])).convert("RGB")
        gray = Image.new("RGB", img.size, (128, 128, 128))
        ins = {}
        for tag, im in [("real", img), ("gray", gray)]:
            x = build_prompt(proc, im, it["question"])
            p = x["input_ids"][0]
            if tag == "real":
                assert len(p) == g["prompt_len"] and ids_hash(p.cpu().numpy()) == g["prompt_sha1"], "prompt mismatch"
            gen_t = torch.tensor(g["gen_ids"], device="cuda", dtype=x["input_ids"].dtype)
            x["input_ids"] = torch.cat([x["input_ids"], gen_t[None]], 1)
            x["attention_mask"] = torch.ones_like(x["input_ids"])
            ins[tag] = x
        P, n = g["prompt_len"], g["n_gen"]
        if n < 2:
            continue
        ids = ins["real"]["input_ids"][0]
        img_idx = torch.nonzero(ids == model.config.image_token_id)[:, 0]
        assert torch.equal(img_idx, torch.nonzero(ins["gray"]["input_ids"][0] == model.config.image_token_id)[:, 0])
        L = len(ids)
        is_img = torch.zeros(L, dtype=torch.bool, device="cuda")
        is_img[img_idx] = True
        sets = {"none": torch.zeros(0, dtype=torch.long, device="cuda"),
                "nonimg": torch.nonzero(~is_img)[:, 0], "prompt": torch.arange(P, device="cuda")}
        rows = torch.arange(P - 1, P - 1 + n, device="cuda")              # positions predicting the generated tokens
        gen = torch.tensor(g["gen_ids"], device="cuda")
        H, arrs = {}, {}
        for name, mode, pr, imtag, probe in CONFIGS:
            STATE.update(mode=mode, protect=sets[pr], rows=rows, img_idx=img_idx,
                         probe=[None] * n_layers if probe else None)
            hs = model.model(**ins[imtag], use_cache=False).last_hidden_state[0, P - 1:P - 1 + n]
            if probe:
                arrs[f"attn_{name}"] = np.stack(STATE["probe"]).astype(np.float16)
            if name in GRAY_PAIRS or name.endswith("_gray"):
                H[name] = hs
            if name == "bf16" or name.endswith("_gray"):
                continue
            kl, ag, nll = compare(lm_head, H["bf16"], hs, gen)
            arrs[f"kl_{name}"], arrs[f"agree_{name}"], arrs[f"nll_{name}"] = kl.astype(np.float32), ag, nll.astype(np.float32)
        STATE.update(mode=None, protect=None, probe=None)
        arrs["nll_bf16"] = compare(lm_head, H["bf16"], H["bf16"], gen)[2].astype(np.float32)
        ent = []
        for r0 in range(0, n, 512):
            a = torch.log_softmax(lm_head(H["bf16"][r0:r0 + 512]).float(), -1)
            ent.append(-(a.exp() * a).sum(-1))
        arrs["ent_bf16"] = torch.cat(ent).cpu().numpy().astype(np.float32)
        greedy_match = float(np.mean([bool((lm_head(H["bf16"][r0:r0 + 512]).argmax(-1) == gen[r0:r0 + 512]).all().item())
                                      for r0 in range(0, n, 512)]))
        greedy_rate = float(torch.cat([(lm_head(H["bf16"][r0:r0 + 512]).argmax(-1) == gen[r0:r0 + 512])
                                       for r0 in range(0, n, 512)]).float().mean())
        top20_rate = float(torch.cat([(lm_head(H["bf16"][r0:r0 + 512]).topk(20, -1).indices == gen[r0:r0 + 512, None]).any(-1)
                                      for r0 in range(0, n, 512)]).float().mean())   # sampled chains: top_k = 20
        for p, pg in GRAY_PAIRS.items():
            arrs[f"dimg_{p}"] = compare(lm_head, H[p], H[pg], gen)[0].astype(np.float32)
        np.savez_compressed(os.path.join(args.out_dir, "per_item", f"{it['image_id']}.npz"), **arrs)
        q = lambda a: [float(a[:max(1, n // 4)].mean()), float(a[-max(1, n // 4):].mean())]
        meta_f.write(json.dumps({"image_id": it["image_id"], "prompt_len": P, "n_gen": n, "think_end": g["think_end"],
                                 "n_img_tokens": int(len(img_idx)), "correct": g["correct"],
                                 "greedy_match_rate": greedy_rate, "greedy_match_chunks": greedy_match,
                                 "top20_rate": top20_rate, "rep32_last1k": g.get("rep32_last1k"),
                                 "hit_max": g.get("hit_max"), "decoding": g.get("decoding", "greedy"),
                                 "kl_first_last_quarter": {c[0]: q(arrs[f"kl_{c[0]}"]) for c in CONFIGS
                                                           if f"kl_{c[0]}" in arrs},
                                 "dimg_first_last_quarter": {p: q(arrs[f"dimg_{p}"]) for p in GRAY_PAIRS}}) + "\n")
        meta_f.flush()
        print(f"[{k}/{len(items)}] {time.time() - t0:.0f}s P={P} n={n} greedy_match={greedy_rate:.4f} top20={top20_rate:.4f} kl4={arrs['kl_kv4_all'].mean():.4f} "
              f"kl3={arrs['kl_kv3_all'].mean():.4f} kl2={arrs['kl_kv2_all'].mean():.4f} "
              f"dimg_bf16={arrs['dimg_bf16'].mean():.4f} attn_img_bf16={arrs['attn_bf16'].mean():.3f} "
              f"kv_err={STATE['err'][:2]}", flush=True)
    meta_f.close()
    with tarfile.open(os.path.join(args.out_dir, "per_item.tar"), "w") as tf:
        tf.add(os.path.join(args.out_dir, "per_item"), arcname="per_item")
    json.dump({"model_tag": args.model_tag, "transformers": transformers.__version__, "n_items": len(items),
               "shard": [args.shard_index, args.num_shards], "configs": [c[0] for c in CONFIGS],
               "kv_rel_err_sample": STATE["err"]}, open(os.path.join(args.out_dir, "run_meta.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
