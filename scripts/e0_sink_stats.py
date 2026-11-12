"""E0: visual-sink existence and outlier statistics for Qwen3-VL (Thinking) models.

Prefill-only. For each COCO val2017 image we ask about one annotated object (its box
plays the role of V*'s target box) and record, per layer and per image token:
  x_uniform  n * r_j (attention received relative to uniform, within the image)
  ccos_pre / ccos_post   centered key cosine with the text-sink key (pre-/post-RoPE)
  rcos_pre   raw key cosine (pre-RoPE)
  hnorm, vnorm, knorm    hidden-state / value / key norms
  mratio     max_d |h_jd| / rms(h_j) over all dims (generic massive ratio)
  sdratio    same, restricted to the text sink's massive dims (Kang et al.-style)
Attention maps are reduced inside a forward hook and dropped, so peak memory stays at
one layer's attention even at ~3k image tokens.

Summaries use two sink criteria:
  x100   n * r >= 100   (2610.15855; depends on n, ~ r >= 3% at V*'s ~3300 tokens)
  abs3   r >= 0.03      (resolution-free equivalent)

Usage:
  python e0_sink_stats.py --model-dir M --model-tag T --coco-img DIR --coco-ann FILE \
      --n-images 200 --seed 0 --upscale 1 --out-dir OUT
"""
import argparse
import json
import os
import random
import time

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, InternVLForConditionalGeneration, Qwen3VLForConditionalGeneration
from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas

from transformers import AttentionInterface
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from sinkquant import sinkstats as S

# --- language-model attention: SDPA for the output, plus softmax weights for the question
# rows only ([H, |Q|, L] instead of [H, L, L]); the per-layer received attention a_j is reduced
# immediately. Validated against eager attention on the first image (--check-attn).
QSTATE = {"q_idx": None, "a": {}}


def query_rows_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    out, _ = sdpa_attention_forward(module, query, key, value, attention_mask,
                                    scaling=scaling, dropout=dropout, **kwargs)
    qi = QSTATE["q_idx"]
    if qi is not None:
        n_rep = query.shape[1] // key.shape[1]
        k = key.repeat_interleave(n_rep, dim=1) if n_rep > 1 else key      # == HF repeat_kv
        q = query[:, :, qi, :].float()
        sc = scaling if scaling is not None else q.shape[-1] ** -0.5
        logits = torch.matmul(q, k.float().transpose(-1, -2)) * sc          # [B, H, |Q|, L]
        L = key.shape[2]
        future = torch.arange(L, device=q.device)[None, :] > qi[:, None]
        p = torch.softmax(logits.masked_fill(future, float("-inf")), dim=-1)
        QSTATE["a"][module.layer_idx] = p[0].mean(dim=(0, 1)).cpu().numpy()
    return out, None


AttentionInterface.register("qrows_sdpa", query_rows_attention)

PAPER_TABLE1 = {  # arXiv 2610.15855 Table 1 (V*, peak layer): sinks/image, x uniform, share
    "Qwen3-VL-8B-Thinking": (3.9, 1259, 0.66),
    "Qwen3-VL-2B-Thinking": (2.7, 163, 0.12),
    "InternVL3_5-8B-HF": (1.5, 342, 0.24),
}
# InternVL3.5 thinking mode (model card); 2610.15855 runs InternVL3.5 under this prompt.
R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()
# InternVL3.5 default dynamic tiling (up to 12 tiles of 448^2), as in 2610.15855 App. A.1;
# the HF preprocessor_config ships with crop_to_patches=false, so it is set explicitly.
INTERNVL_TILE_KW = {"crop_to_patches": True, "max_patches": 12, "min_patches": 1}
FEATS = ["x_uniform", "ccos_pre", "ccos_post", "rcos_pre", "hnorm", "vnorm", "knorm", "mratio", "sdratio"]


def pick_images(ann_path, n, seed):
    ann = json.load(open(ann_path))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    imgs = {im["id"]: im for im in ann["images"]}
    by_img = {}
    for a in ann["annotations"]:
        im = imgs[a["image_id"]]
        frac = a["area"] / (im["width"] * im["height"])
        if not a["iscrowd"] and 0.02 <= frac <= 0.30:
            by_img.setdefault(a["image_id"], []).append(a)
    rng = random.Random(seed)
    ids = sorted(by_img)
    rng.shuffle(ids)
    out = []
    for iid in ids[:n]:
        a = rng.choice(by_img[iid])
        x, y, w, h = a["bbox"]
        out.append({"image_id": iid, "file": imgs[iid]["file_name"],
                    "wh": (imgs[iid]["width"], imgs[iid]["height"]),
                    "boxes": [(x, y, x + w, y + h)], "cat": cats[a["category_id"]],
                    "question": f"What color is the {cats[a['category_id']]}? Answer with one word."})
    return out


VSTAR_FORMAT = r"Put only the letter of the correct option inside \boxed{}."  # 2610.15855 Fig. 9


def pick_vstar(root, n):
    """V* bench: 191 multiple-choice questions; target boxes from the per-image json (xywh).

    As in 2610.15855 App. A.1 the closing answer-format sentence is replaced by the paper's
    thinking-model line. Multiple target objects (relative_position) -> union of boxes.
    """
    out = []
    for line in open(os.path.join(root, "test_questions.jsonl")):
        q = json.loads(line)
        meta = json.load(open(os.path.join(root, os.path.splitext(q["image"])[0] + ".json")))
        with Image.open(os.path.join(root, q["image"])) as im:
            wh = im.size
        lines = q["text"].split("\n")
        assert lines[-1].startswith("Answer with the option"), lines[-1]
        out.append({"image_id": f"q{q['question_id']}", "file": q["image"], "wh": wh,
                    "boxes": [(x, y, x + w, y + h) for x, y, w, h in meta["bbox"]],
                    "cat": "+".join(meta["target_object"]), "category": q["category"], "label": q["label"],
                    "question": "\n".join(lines[:-1] + [VSTAR_FORMAT])})
    return out[:n]


POPE_FORMAT = r"Answer yes or no. Put only the answer inside \boxed{}."


def pick_pope(root, n):
    """POPE subset (data/pope/subset/questions.jsonl, made by setup/make_pope_subset.py).
    POPE has no target boxes, so no image token is excluded as 'inside the target'."""
    out = []
    for line in open(os.path.join(root, "questions.jsonl")):
        q = json.loads(line)
        with Image.open(os.path.join(root, q["file"])) as im:
            wh = im.size
        out.append({**q, "wh": wh, "boxes": [], "cat": q["category"],
                    "question": q["question"] + "\n" + POPE_FORMAT})
    return out[:n]


def _inside(masks, n):
    return np.any(masks, axis=0) if len(masks) else np.zeros(n, dtype=bool)


def build_qwen3vl(proc, cfg, im_end, img, question, wh, boxes):
    msgs = [{"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    text += "\n</think>\n\n" if text.rstrip().endswith("<think>") else "<think>\n\n</think>\n\n"
    inputs = proc(text=[text], images=[img], return_tensors="pt").to("cuda")
    ids = inputs["input_ids"][0].cpu().numpy()
    img_mask = ids == cfg.image_token_id
    img_idx = np.nonzero(img_mask)[0]
    v_end = int(np.nonzero(ids == cfg.vision_end_token_id)[0][-1])
    q_end = v_end + 1 + int(np.nonzero(ids[v_end + 1:] == im_end)[0][0])
    q_idx = np.arange(v_end + 1, q_end)
    _, gh, gw = inputs["image_grid_thw"][0].tolist()
    assert len(img_idx) == (gh // 2) * (gw // 2), (len(img_idx), gh, gw)
    inside = _inside([S.token_boxes_inside(gh, gw, b, wh) for b in boxes], len(img_idx))
    return inputs, ids, img_mask, img_idx, q_idx, inside, {"grid": [gh, gw]}


def build_internvl(proc, cfg, im_end, img, question, wh, boxes):
    msgs = [{"role": "system", "content": [{"type": "text", "text": R1_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": img}, {"type": "text", "text": question}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt", **INTERNVL_TILE_KW).to("cuda", torch.bfloat16)
    ids = inputs["input_ids"][0].cpu().numpy()
    img_mask = ids == cfg.image_token_id
    img_idx = np.nonzero(img_mask)[0]
    last = int(img_idx[-1])
    assert np.all(np.diff(img_idx) == 1), "expected one contiguous image span"
    q_start = last + 2                                            # skip "</img>"
    q_end = q_start + int(np.nonzero(ids[q_start:] == im_end)[0][0])
    q_idx = np.arange(q_start, q_end)
    ncols, nrows = get_optimal_tiled_canvas((wh[1], wh[0]), (448, 448), 1, INTERNVL_TILE_KW["max_patches"])
    n_tiles = ncols * nrows + (1 if ncols * nrows > 1 else 0)
    assert inputs["pixel_values"].shape[0] == n_tiles, (inputs["pixel_values"].shape, ncols, nrows)
    xy, tid = S.internvl_token_centers(ncols, nrows, wh)
    assert len(xy) == len(img_idx), (len(xy), len(img_idx))
    inside = _inside([S.points_inside(xy, b) for b in boxes], len(img_idx))
    return inputs, ids, img_mask, img_idx, q_idx, inside, {"tiles_cols_rows": [ncols, nrows], "n_tiles": n_tiles}


def cache_kv(cache, i):
    if hasattr(cache, "layers"):
        return cache.layers[i].keys, cache.layers[i].values
    return cache.key_cache[i], cache.value_cache[i]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--family", choices=["qwen3vl", "internvl"], default="qwen3vl")
    ap.add_argument("--dataset", choices=["coco", "vstar", "pope"], default="coco")
    ap.add_argument("--pope-root")
    ap.add_argument("--coco-img")
    ap.add_argument("--coco-ann")
    ap.add_argument("--vstar-root")
    ap.add_argument("--n-images", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--upscale", type=int, default=1, help="bicubic upscale factor (diagnostic)")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(os.path.join(args.out_dir, "per_image"), exist_ok=True)
    torch.manual_seed(args.seed)

    assert torch.cuda.device_count() == 1, "expose exactly one GPU"
    gpu = torch.cuda.get_device_name(0)
    print("gpu:", gpu, "transformers:", transformers.__version__, "upscale:", args.upscale, flush=True)

    proc = AutoProcessor.from_pretrained(args.model_dir)
    # Vision tower on SDPA (its ~4x-larger patch sequence would make eager attention the
    # memory peak); only the language model needs explicit attention weights.
    cls = Qwen3VLForConditionalGeneration if args.family == "qwen3vl" else InternVLForConditionalGeneration
    model = cls.from_pretrained(
        args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
    lm_cfg = model.model.language_model.config
    lm_cfg._attn_implementation = "qrows_sdpa"
    print("family", args.family, "attn impl: text", lm_cfg._attn_implementation, flush=True)
    cfg = model.config
    tok = proc.tokenizer
    im_end = tok.convert_tokens_to_ids("<|im_end|>")
    layers = model.model.language_model.layers
    n_layers = len(layers)
    mid = n_layers // 2

    pre_keys = {}
    hooks = [layer.self_attn.k_norm.register_forward_hook(
        lambda m, i, o, idx=idx: pre_keys.__setitem__(idx, o.detach())) for idx, layer in enumerate(layers)]

    def eager_received_attention(inputs, q_idx):
        """Reference path for the check: eager attention weights, reduced the same way."""
        got = {}
        tmp = [layer.self_attn.register_forward_hook(
            lambda m, i, o, idx=idx: got.__setitem__(idx, o[1][0][:, q_idx, :].float().mean(dim=(0, 1)).cpu().numpy())
            or (o[0], None)) for idx, layer in enumerate(layers)]
        lm_cfg._attn_implementation = "eager"
        try:
            with torch.no_grad():
                model(**inputs, use_cache=False)
        finally:
            lm_cfg._attn_implementation = "qrows_sdpa"
            for h in tmp:
                h.remove()
        return np.stack([got[l] for l in range(n_layers)])

    attn_check = None

    if args.dataset == "coco":
        items = pick_images(args.coco_ann, args.n_images, args.seed)
        img_root = args.coco_img
    elif args.dataset == "vstar":
        items = pick_vstar(args.vstar_root, args.n_images)
        img_root = args.vstar_root
    else:
        items = pick_pope(args.pope_root, args.n_images)
        img_root = args.pope_root
    meta_f = open(os.path.join(args.out_dir, "meta.jsonl"), "w")
    t_start = time.time()
    for k, it in enumerate(items):
        img = Image.open(os.path.join(img_root, it["file"])).convert("RGB")
        wh, boxes = tuple(it["wh"]), it["boxes"]
        assert img.size == wh, (img.size, wh)
        if args.upscale > 1:
            img = img.resize((wh[0] * args.upscale, wh[1] * args.upscale), Image.BICUBIC)
            wh = (wh[0] * args.upscale, wh[1] * args.upscale)
            boxes = [tuple(v * args.upscale for v in b) for b in boxes]
        question = it["question"]
        if args.family == "qwen3vl":
            inputs, ids, img_mask, img_idx, q_idx, inside, layout = build_qwen3vl(proc, cfg, im_end, img, question, wh, boxes)
        else:
            inputs, ids, img_mask, img_idx, q_idx, inside, layout = build_internvl(proc, cfg, im_end, img, question, wh, boxes)

        QSTATE["q_idx"] = torch.as_tensor(q_idx, device="cuda")
        QSTATE["a"].clear()
        pre_keys.clear()
        with torch.no_grad():
            out = model(**inputs, output_hidden_states=True, use_cache=True)
        a_all = np.stack([QSTATE["a"][l] for l in range(n_layers)])         # [nl, L]
        assert np.allclose(a_all.sum(-1), 1.0, atol=1e-3), "received attention must sum to 1"
        if k == 0:
            QSTATE["q_idx"] = None
            a_ref = eager_received_attention(inputs, torch.as_tensor(q_idx, device="cuda"))
            attn_check = {"max_abs_diff": float(np.abs(a_ref - a_all).max()),
                          "max_rel_diff_top_token": float(np.max(np.abs(a_ref.max(-1) - a_all.max(-1)) / a_ref.max(-1))),
                          "same_argmax_all_layers": bool((a_ref.argmax(-1) == a_all.argmax(-1)).all())}
            print("attn check vs eager:", attn_check, flush=True)
            assert attn_check["max_abs_diff"] < 1e-2, attn_check
        t_star = S.text_sink_position(a_all[mid], img_mask)

        feats = {f: np.zeros((n_layers, len(img_idx)), np.float32) for f in FEATS}
        sd_record = {}
        for l in range(n_layers):
            r = S.in_image_share(a_all[l], img_idx)
            feats["x_uniform"][l] = len(img_idx) * r
            kp = pre_keys[l][0].float().transpose(0, 1).cpu().numpy()        # [Hkv, L, D]
            kc, vc = cache_kv(out.past_key_values, l)
            kc = kc[0].float().cpu().numpy()
            vc = vc[0].float().cpu().numpy()
            assert kp.shape == kc.shape and kp.shape[1] == len(ids), (kp.shape, kc.shape, len(ids))
            feats["ccos_pre"][l] = S.centered_key_cosine(kp, img_idx, t_star)
            feats["ccos_post"][l] = S.centered_key_cosine(kc, img_idx, t_star)
            feats["rcos_pre"][l] = S.raw_key_cosine(kp, img_idx, t_star)
            feats["knorm"][l] = np.linalg.norm(kp[:, img_idx], axis=-1).mean(0)
            feats["vnorm"][l] = np.linalg.norm(vc[:, img_idx], axis=-1).mean(0)
            hl = out.hidden_states[l][0].float()                               # layer input
            h = hl[img_idx]
            feats["hnorm"][l] = h.norm(dim=-1).cpu().numpy()
            rms = h.pow(2).mean(-1).sqrt()
            feats["mratio"][l] = (h.abs().max(-1).values / rms).cpu().numpy()
            dims = S.sink_dims(hl[t_star].cpu().numpy())
            feats["sdratio"][l] = S.sink_dim_ratio(h.cpu().numpy(), dims)
            if k == 0:
                sd_record[l] = {"dims": dims[:8].tolist(), "max_abs_tstar": float(hl[t_star].abs().max())}
        np.savez_compressed(os.path.join(args.out_dir, "per_image", f"{it['image_id']}.npz"),
                            inside=inside, **{f: v.astype(np.float16) for f, v in feats.items()})
        rec = {**it, "wh_used": wh, "n_img_tokens": int(len(img_idx)), "layout": layout,
               "t_star": t_star, "t_star_token": tok.decode([int(ids[t_star])]), "seq_len": int(len(ids))}
        if k == 0:
            rec["text_sink_dims_per_layer"] = sd_record
        meta_f.write(json.dumps(rec) + "\n")
        del out
        if k % 20 == 0:
            print(f"[{k}/{len(items)}] {time.time() - t_start:.0f}s n_img={len(img_idx)} "
                  f"t*={t_star}({tok.decode([int(ids[t_star])])!r}) "
                  f"mem={torch.cuda.max_memory_allocated() / 2**30:.1f}GB", flush=True)
    meta_f.close()
    for hk in hooks:
        hk.remove()

    summarize(args, n_layers, gpu, peak_mem=torch.cuda.max_memory_allocated() / 2**30,
              elapsed=time.time() - t_start, attn_check=attn_check)


def criterion_summary(recs, n_layers, mask_fn, seed):
    shares = np.zeros(n_layers)
    for l in range(n_layers):
        s = []
        for z in recs:
            xu = z["x_uniform"][l].astype(np.float64)
            r = xu / len(xu)
            s.append(r[mask_fn(r, z["inside"])].sum())
        shares[l] = np.mean(s)
    pk = S.peak_layer(shares)
    per_img, xu_sinks, labels, pool = [], [], [], {}
    for z in recs:
        xu = z["x_uniform"][pk].astype(np.float64)
        r = xu / len(xu)
        m = mask_fn(r, z["inside"])
        per_img.append(int(m.sum()))
        xu_sinks.extend(xu[m].tolist())
        labels.append(m)
        for f in FEATS[1:]:
            pool.setdefault(f, []).append(z[f][pk].astype(np.float64))
    labels = np.concatenate(labels)
    sign = {"vnorm": -1, "knorm": -1}  # paper scores -||v|| and -||k||
    aucs = {f: S.auc(sign.get(f, 1) * np.concatenate(v), labels) for f, v in pool.items()}
    rng = np.random.default_rng(seed)
    ccp = np.concatenate(pool["ccos_post"])
    return {
        "peak_layer": pk,
        "share_per_layer": [round(x, 4) for x in shares],
        "sinks_per_image": float(np.mean(per_img)),
        "frac_images_with_sink": float(np.mean(np.array(per_img) > 0)),
        "x_uniform_median_of_sinks": float(np.median(xu_sinks)) if xu_sinks else None,
        "share_at_peak": float(shares[pk]),
        "n_sink_tokens": int(labels.sum()),
        "auc_at_peak": aucs,
        "null_auc_permuted_labels": S.auc(ccp, rng.permutation(labels)),
        "ccos_post_gt_0.5_frac_of_sinks": float((ccp[labels] > 0.5).mean()) if labels.any() else None,
        "ccos_post_gt_0.5_frac_of_nonsinks": float((ccp[~labels] > 0.5).mean()),
    }


def summarize(args, n_layers, gpu, peak_mem, elapsed, attn_check=None):
    d = os.path.join(args.out_dir, "per_image")
    recs = [np.load(os.path.join(d, f)) for f in sorted(os.listdir(d))]
    n_tok = [z["x_uniform"].shape[1] for z in recs]
    conc = []
    for l in range(n_layers):
        top1 = [z["x_uniform"][l].astype(np.float64).max() / z["x_uniform"].shape[1] for z in recs]
        sdr = np.concatenate([z["sdratio"][l].astype(np.float64) for z in recs])
        conc.append({"layer": l, "median_top1_share": round(float(np.median(top1)), 4),
                     "p90_top1_share": round(float(np.percentile(top1, 90)), 4),
                     "sdratio_p50": round(float(np.median(sdr)), 2),
                     "sdratio_p99": round(float(np.percentile(sdr, 99)), 2)})
    summary = {
        "model_tag": args.model_tag, "gpu": gpu, "n_images": len(recs), "seed": args.seed,
        "upscale": args.upscale, "n_img_tokens_median": float(np.median(n_tok)), "n_layers": n_layers,
        "criteria": {
            "x100": criterion_summary(recs, n_layers, lambda r, ins: S.sink_mask(r, ins, 100.0), args.seed),
            "abs3": criterion_summary(recs, n_layers, lambda r, ins: S.sink_mask_abs(r, ins, 0.03), args.seed),
        },
        "paper_table1_vstar": PAPER_TABLE1.get(args.model_tag),
        "concentration_per_layer": conc,
        "peak_mem_gb": round(peak_mem, 2), "elapsed_s": round(elapsed, 1),
        "attn_check_vs_eager": attn_check,
    }
    json.dump(summary, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    print(json.dumps({k: v for k, v in summary.items() if k != "concentration_per_layer"}, indent=1), flush=True)


if __name__ == "__main__":
    main()
