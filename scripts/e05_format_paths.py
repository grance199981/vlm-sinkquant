"""E0.5: offline decomposition of activation-quantization error into P1 / P2.

For each linear-layer input of the language model (qkv, o_proj, gate/up, down) we capture
three groups of rows on V* images:
  sink    visual sinks (x100 criterion at the E0 peak layer, read from an E0 run directory)
  normal  random non-sink image tokens (seeded)
  tsink   the text-sink row t* (carries the massive activation)
and fake-quantize only the activations (weights stay BF16->FP32) with per-token INT4, MXFP4
and NVFP4, with and without a random orthogonal rotation (QuaRot-like). The output error
||delta w^T||^2 is split into P2 (delta on the row's massive channels, |x| >= 0.25 max|x|)
and P1 (the rest). For NVFP4 the per-tensor scale is the full-sequence amax; a second
variant 'nvfp4_xs' recomputes it without the sink rows (text sink + visual sinks) to measure
the cross-token effect of the global scale.

Usage:
  python e05_format_paths.py --family qwen3vl --model-dir M --model-tag T --vstar-root V \
      --e0-run-dir RUN/e0 --n-images 24 --n-normal 64 --seed 0 --out-dir OUT
"""
import argparse
import json
import os
import sys
import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoProcessor, InternVLForConditionalGeneration, Qwen3VLForConditionalGeneration

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from e0_sink_stats import build_internvl, build_qwen3vl, pick_vstar  # noqa: E402

from sinkquant import quant as Q  # noqa: E402
from sinkquant import sinkstats as S  # noqa: E402

from transformers import AttentionInterface  # noqa: E402
from transformers.integrations.sdpa_attention import sdpa_attention_forward  # noqa: E402

SITES = ["qkv", "o", "gateup", "down"]
KV_FMTS = ["int4_tok", "mxfp4", "nvfp4"]
KV_PARTS = ["K", "V", "KV"]
N_RAND = 5
# rotations evaluated on the linear-layer path: 'rot' = seeded random orthogonal (old name kept
# for comparability), 'had' = randomized full Hadamard (QuaRot), 'bhadB' = randomized
# block-diagonal Hadamard with block B (MR-GPTQ-style; 16 = NVFP4 block, 32 = MXFP4, 128)
ROTS = ["none", "rot", "had", "bhad16", "bhad32", "bhad128"]
PC_FMTS = [("int4_asym", 128), ("mxfp4", 32), ("nvfp4", 16)]   # per-channel K quant along tokens
MASS_CAP = 1024

# --- KV-path probe -------------------------------------------------------------------------
# Inside each language-model attention call (output still computed by SDPA, so the model is
# unchanged) we take the question rows and compare their attention output with exact K/V
# against K/V where only one target set of positions is fake-quantized (KV-cache style:
# per token and per KV head over head_dim; NVFP4 global scale = amax of the layer's K or V).
# Targets: visual sinks, N_RAND equal-size random ordinary visual tokens, the text sink, all
# image tokens. Metric per layer: sum ||O' - O||^2 and sum ||O||^2 over heads and question rows.
KV = {"active": False, "q_idx": None, "targets": None, "acc": None, "n_layers": 0}


def _kv_quant(x, fmt):
    if fmt == "nvfp4":
        return Q.nvfp4(x, tensor_amax=KV["_amax"])
    return Q.FORMATS[fmt](x)


def kvprobe_attention(module, query, key, value, attention_mask, scaling=None, dropout=0.0, **kwargs):
    out, _ = sdpa_attention_forward(module, query, key, value, attention_mask,
                                    scaling=scaling, dropout=dropout, **kwargs)
    if KV["active"]:
        qi = KV["q_idx"]
        n_rep = query.shape[1] // key.shape[1]
        sc = scaling if scaling is not None else query.shape[-1] ** -0.5
        q = query[0, :, qi, :].float()                                   # [H, |Q|, D]
        K = key[0].float()                                               # [Hkv, L, D]
        V = value[0].float()
        L = K.shape[1]
        future = torch.arange(L, device=q.device)[None, :] > qi[:, None]

        def attend(Kx, Vx, return_p=False):
            Kr = Kx.repeat_interleave(n_rep, 0) if n_rep > 1 else Kx
            Vr = Vx.repeat_interleave(n_rep, 0) if n_rep > 1 else Vx
            pr = torch.softmax((q @ Kr.transpose(-1, -2) * sc).masked_fill(future, float("-inf")), -1)
            return (pr @ Vr, pr) if return_p else pr @ Vr

        O, P = attend(K, V, return_p=True)
        recv = P.mean(dim=(0, 1))                                        # attention received per key [L]
        targets = dict(KV["targets"])
        if "sink" in targets and KV.get("img_idx") is not None:
            # attention-matched controls among non-sink image tokens, chosen in THIS layer:
            #   att_top : the len(sinks) most-attended non-sink image tokens (count-matched)
            #   att_mass: most-attended non-sink image tokens until their received attention
            #             reaches the sinks' (mass-matched; capped at MASS_CAP tokens)
            sinks = targets["sink"]
            cand = KV["nonsink_img"]
            order = torch.argsort(recv[cand], descending=True)
            ranked = cand[order]
            targets["att_top"] = ranked[:len(sinks)]
            cum = torch.cumsum(recv[ranked], 0)
            k = int(torch.searchsorted(cum, recv[sinks].sum()).item()) + 1
            targets["att_mass"] = ranked[:min(k, MASS_CAP, len(ranked))]
        if not KV.get("_checked"):
            ref = out[0, qi].float().transpose(0, 1)                     # sdpa output is [B, L, H, D]
            err = ((O - ref).norm() / ref.norm()).item()
            KV["check_rel_err_vs_sdpa"] = err
            assert err < 2e-2, f"probe attention does not match the model's SDPA output: {err}"
            KV["_checked"] = True
        den = O.pow(2).sum().item()
        l = module.layer_idx
        for fmt in KV_FMTS:
            for tname, idx in targets.items():
                if idx is None or len(idx) == 0:
                    continue
                KV["_amax"] = K.abs().amax()
                Kq = K.clone()
                Kq[:, idx] = _kv_quant(K[:, idx], fmt)
                KV["_amax"] = V.abs().amax()
                Vq = V.clone()
                Vq[:, idx] = _kv_quant(V[:, idx], fmt)
                for part, (Kx, Vx) in {"K": (Kq, V), "V": (K, Vq), "KV": (Kq, Vq)}.items():
                    num = (attend(Kx, Vx) - O).pow(2).sum().item()
                    a = KV["acc"].setdefault((fmt, tname.split("#")[0], part),
                                             {"num": np.zeros(KV["n_layers"]), "den": np.zeros(KV["n_layers"]),
                                              "cnt": np.zeros(KV["n_layers"]), "mass": np.zeros(KV["n_layers"]),
                                              "size": np.zeros(KV["n_layers"]), "rel_sum": np.zeros(KV["n_layers"])})
                    a["num"][l] += num
                    a["den"][l] += den
                    a["cnt"][l] += 1
                    a["mass"][l] += float(recv[idx].sum())
                    a["size"][l] += len(idx)
                    a["rel_sum"][l] += num / max(den, 1e-30)
        if "sink" in targets and KV.get("sink_local") is not None:
            _per_channel_probe(K, V, attend, O, den, l, targets)
    return out, None


def _per_channel_probe(K, V, attend, O, den, l, targets):
    """Cross-token damage under per-channel K quantization along tokens (KIVI/KVQuant-style).

    Only the image tokens' K is quantized (V and text stay FP). For a protected set S (visual
    sinks, or a count-matched random / top-||K|| non-sink control):
      in : S takes part in the group statistics, then S is restored to FP
      ex : S is excluded from the group statistics and kept in FP
    (in - ex) is the damage S does to the OTHER tokens through range expansion.
    'all' = every image token quantized, nothing excluded or restored.
    """
    img = KV["img_idx"]
    n = len(img)
    Kimg = K[:, img, :].transpose(1, 2)                                  # [Hkv, D, n]
    sink_l = KV["sink_local"]
    k_sink = int(sink_l.sum())
    rand_l = torch.zeros(n, dtype=torch.bool, device=K.device)
    rand_l[KV["pos2local"][targets["rand#0"]]] = True
    knorm = Kimg.norm(dim=1).mean(0).masked_fill(sink_l, -1.0)          # [n]
    knorm_l = torch.zeros(n, dtype=torch.bool, device=K.device)
    knorm_l[torch.topk(knorm, k_sink).indices] = True
    # count-matched most-attended non-sink image tokens of THIS layer (attention control for the
    # "own error" part of protecting S; the cross-token part in - ex is not attention-confounded)
    att_l = torch.zeros(n, dtype=torch.bool, device=K.device)
    att_l[KV["pos2local"][targets["att_top"]]] = True
    sets = {"sink": sink_l, "rand": rand_l, "knorm": knorm_l, "atttop": att_l}

    def err_of(Kimg_q):
        Kq = K.clone()
        Kq[:, img, :] = Kimg_q.transpose(1, 2)
        return (attend(Kq, V) - O).pow(2).sum().item()

    def acc(key, num):
        a = KV["pc"].setdefault(key, {"num": np.zeros(KV["n_layers"]), "den": np.zeros(KV["n_layers"])})
        a["num"][l] += num
        a["den"][l] += den

    for fmt, blk in PC_FMTS:
        q_all = Q.quant_along_tokens(Kimg, fmt, blk)
        acc((fmt, "all", "all"), err_of(q_all))
        for sname, smask in sets.items():
            acc((fmt, sname, "in"), err_of(torch.where(smask[None, None, :], Kimg, q_all)))
            q_ex = Q.quant_along_tokens(Kimg, fmt, blk, exclude=smask)
            acc((fmt, sname, "ex"), err_of(torch.where(smask[None, None, :], Kimg, q_ex)))
    # mechanism: in 128-token groups that contain a sink, how often the per-(head, channel)
    # abs max of the group sits on a sink token
    xp, _ = Q._pad_tokens(Kimg.abs(), 128)
    sp, _ = Q._pad_tokens(sink_l.float(), 128)
    arg = xp.reshape(*xp.shape[:-1], -1, 128).argmax(-1)                 # [Hkv, D, G]
    G = arg.shape[-1]
    owner = torch.arange(G, device=K.device)[None, None, :] * 128 + arg
    is_sink = sp[owner] > 0
    m = (sp.reshape(-1, 128).amax(-1) > 0)[None, None, :].expand_as(is_sink)
    st = KV["pc_stats"].setdefault("groupmax_from_sink", {"hit": np.zeros(KV["n_layers"]), "cnt": np.zeros(KV["n_layers"])})
    st["hit"][l] += float(is_sink[m].sum())
    st["cnt"][l] += float(m.sum())


AttentionInterface.register("kvprobe_sdpa", kvprobe_attention)
FMTS = ["int4_tok", "mxfp4", "nvfp4", "nvfp4_xs"]
GROUPS = ["sink", "normal", "tsink"]


def site_modules(layer):
    a, m = layer.self_attn, layer.mlp
    return {"qkv": (a.q_proj, [a.q_proj, a.k_proj, a.v_proj]), "o": (a.o_proj, [a.o_proj]),
            "gateup": (m.gate_proj, [m.gate_proj, m.up_proj]), "down": (m.down_proj, [m.down_proj])}


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["qwen3vl", "internvl"], required=True)
    ap.add_argument("--model-dir", required=True)
    ap.add_argument("--model-tag", required=True)
    ap.add_argument("--vstar-root", required=True)
    ap.add_argument("--e0-run-dir", required=True, help="E0 output dir with summary.json, meta.jsonl, per_image/")
    ap.add_argument("--criterion", default="x100")
    ap.add_argument("--n-images", type=int, default=24)
    ap.add_argument("--n-normal", type=int, default=64)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    assert torch.cuda.device_count() == 1
    rng = np.random.default_rng(args.seed)
    rng_kv = np.random.default_rng(args.seed + 1)   # separate stream: linear-path sampling unchanged

    e0 = json.load(open(os.path.join(args.e0_run_dir, "summary.json")))
    assert e0["model_tag"] == args.model_tag, (e0["model_tag"], args.model_tag)
    peak = e0["criteria"][args.criterion]["peak_layer"]
    meta = {m["image_id"]: m for m in map(json.loads, open(os.path.join(args.e0_run_dir, "meta.jsonl")))}

    proc = AutoProcessor.from_pretrained(args.model_dir)
    cls = Qwen3VLForConditionalGeneration if args.family == "qwen3vl" else InternVLForConditionalGeneration
    model = cls.from_pretrained(args.model_dir, dtype=torch.bfloat16, attn_implementation="sdpa",
                                device_map="cuda").eval()
    cfg = model.config
    im_end = proc.tokenizer.convert_tokens_to_ids("<|im_end|>")
    layers = model.model.language_model.layers
    n_layers = len(layers)
    model.model.language_model.config._attn_implementation = "kvprobe_sdpa"
    KV.update(acc={}, pc={}, pc_stats={}, n_layers=n_layers)
    hidden = model.model.language_model.config.hidden_size
    inter = layers[0].mlp.down_proj.in_features
    rots = {d: Q.random_orthogonal(d, seed=args.seed, device="cuda") for d in sorted({hidden, inter})}
    hads = {d: Q.random_hadamard(d, seed=args.seed, device="cuda") for d in sorted({hidden, inter})}

    def rotator(name, d):
        if name == "none":
            return None
        if name == "rot":
            return rots[d]
        if name == "had":
            return hads[d]
        blk = int(name[4:])
        return lambda t, blk=blk: Q.block_hadamard_rotate(t, blk, seed=args.seed)

    # capture: rows (bf16, CPU) and full-tensor amax (raw / rotated, with and without sink rows)
    cap = {}
    state = {"rows": None, "excl": None}

    def make_hook(l, site):
        def hook(mod, inp):
            x = inp[0][0]                                           # [L, d] bf16
            xf = x.float()
            R = rots[x.shape[-1]]
            xr = xf @ R
            keep = torch.ones(x.shape[0], dtype=torch.bool, device=x.device)
            keep[state["excl"]] = False
            rec = cap.setdefault((l, site), {"x": [], "grp": []})
            rec["x"].append(x[state["rows"]].cpu())
            rec["grp"].append(state["grp"])
            n = len(state["rows"])
            vals = [("amax", xf.abs().amax()), ("amax_xs", xf[keep].abs().amax()), ("amax_rot", xr.abs().amax()),
                    ("amax_had", (xf @ hads[x.shape[-1]]).abs().amax())]
            for b in (16, 32, 128):
                vals.append((f"amax_bhad{b}", Q.block_hadamard_rotate(xf, b, seed=args.seed).abs().amax()))
            for key, val in vals:
                rec.setdefault(key, []).append(torch.full((n,), float(val)))
        return hook

    hooks = []
    for l, layer in enumerate(layers):
        for site, (mod, _) in site_modules(layer).items():
            hooks.append(mod.register_forward_pre_hook(make_hook(l, site)))

    items = [it for it in pick_vstar(args.vstar_root, 10_000) if it["image_id"] in meta]
    items = [items[i] for i in rng.permutation(len(items))[:args.n_images]]
    per_img = []
    t0 = time.time()
    for k, it in enumerate(items):
        img = Image.open(os.path.join(args.vstar_root, it["file"])).convert("RGB")
        build = build_qwen3vl if args.family == "qwen3vl" else build_internvl
        inputs, ids, img_mask, img_idx, q_idx, inside, layout = build(proc, cfg, im_end, img, it["question"],
                                                                      tuple(it["wh"]), it["boxes"])
        z = np.load(os.path.join(args.e0_run_dir, "per_image", f"{it['image_id']}.npz"))
        assert z["x_uniform"].shape[1] == len(img_idx), (z["x_uniform"].shape, len(img_idx))
        xu = z["x_uniform"][peak].astype(np.float64)
        sink_local = S.sink_mask(xu / len(xu), z["inside"]) if args.criterion == "x100" else \
            S.sink_mask_abs(xu / len(xu), z["inside"])
        t_star = int(meta[it["image_id"]]["t_star"])
        sinks = img_idx[sink_local]
        pool = img_idx[~sink_local]
        normal = rng.choice(pool, size=min(args.n_normal, len(pool)), replace=False)
        rows = np.concatenate([sinks, normal, [t_star]]).astype(np.int64)
        grp = np.array([0] * len(sinks) + [1] * len(normal) + [2])
        state["rows"] = torch.as_tensor(rows, device="cuda")
        state["excl"] = torch.as_tensor(np.concatenate([sinks, [t_star]]).astype(np.int64), device="cuda")
        state["grp"] = torch.as_tensor(grp)
        targets = {"tsink": torch.as_tensor([t_star], device="cuda"),
                   "allimg": torch.as_tensor(img_idx, device="cuda")}
        if len(sinks):
            targets["sink"] = torch.as_tensor(sinks, device="cuda")
            for r in range(N_RAND):
                targets[f"rand#{r}"] = torch.as_tensor(rng_kv.choice(pool, size=len(sinks), replace=False),
                                                       device="cuda")
        nonsink = np.setdiff1d(img_idx, sinks)
        pos2local = torch.full((int(ids.shape[0]),), -1, dtype=torch.long, device="cuda")
        pos2local[torch.as_tensor(img_idx, device="cuda")] = torch.arange(len(img_idx), device="cuda")
        KV.update(active=True, q_idx=torch.as_tensor(q_idx, device="cuda"), targets=targets,
                  img_idx=torch.as_tensor(img_idx, device="cuda"),
                  nonsink_img=torch.as_tensor(nonsink, device="cuda"), pos2local=pos2local,
                  sink_local=torch.as_tensor(sink_local, device="cuda") if len(sinks) else None)
        with torch.no_grad():
            model(**inputs, use_cache=False)
        KV["active"] = False
        per_img.append({"image_id": it["image_id"], "n_sinks": int(len(sinks)), "n_img_tokens": int(len(img_idx))})
        print(f"[{k}/{len(items)}] {time.time() - t0:.0f}s sinks={len(sinks)} n_img={len(img_idx)}", flush=True)
    for h in hooks:
        h.remove()

    # analysis: per (layer, site) load FP32 weights once, evaluate all formats x {none, rot}
    acc = {}
    for l, layer in enumerate(layers):
        mods = site_modules(layer)
        for site in SITES:
            rec = cap[(l, site)]
            X = torch.cat(rec["x"]).float().cuda()
            G = torch.cat(rec["grp"]).cuda()
            W = torch.cat([m.weight.float() for m in mods[site][1]], 0)
            for rot_name in ROTS:
                R = rotator(rot_name, X.shape[-1])
                for fmt in FMTS:
                    if fmt == "nvfp4_xs" and rot_name != "none":
                        continue
                    if fmt.startswith("nvfp4"):
                        key = ("amax" + ("_xs" if fmt == "nvfp4_xs" else "")) if rot_name == "none" else f"amax_{rot_name}"
                        e = Q.path_errors(X, W, "nvfp4", rot=R, tensor_amax=torch.cat(rec[key]).cuda())
                    else:
                        e = Q.path_errors(X, W, fmt, rot=R)
                    for g, gname in enumerate(GROUPS):
                        m = G == g
                        if not m.any():
                            continue
                        a = acc.setdefault((rot_name, fmt, gname), {"total": np.zeros((n_layers, 4)), "p1": np.zeros((n_layers, 4)),
                                                                    "p2": np.zeros((n_layers, 4)), "ref": np.zeros((n_layers, 4)),
                                                                    "n": np.zeros((n_layers, 4)), "n_massive": np.zeros((n_layers, 4))})
                        j = SITES.index(site)
                        for kk in ["total", "p1", "p2", "ref", "n_massive"]:
                            a[kk][l, j] += float(e[kk][m].double().sum())
                        a["n"][l, j] += float(m.sum())
            del X, W
        torch.cuda.empty_cache()
        print(f"analysed layer {l}", flush=True)

    out = {"model_tag": args.model_tag, "family": args.family, "peak_layer_e0": peak, "criterion": args.criterion,
           "n_images": len(items), "n_sinks_total": int(sum(p["n_sinks"] for p in per_img)),
           "rotation": "random orthogonal (seeded), stands in for QuaRot Hadamard", "massive_frac": 0.25,
           "per_image": per_img, "results": {}}
    for (rot_name, fmt, gname), a in acc.items():
        tot, ref, p1, p2 = a["total"].sum(), a["ref"].sum(), a["p1"].sum(), a["p2"].sum()
        nrow = max(a["n"].sum(), 1)
        site_rel = {s: float(a["total"][:, j].sum() / max(a["ref"][:, j].sum(), 1e-30)) for j, s in enumerate(SITES)}
        out["results"][f"{rot_name}|{fmt}|{gname}"] = {
            "rel_err": float(tot / max(ref, 1e-30)),
            "p2_share": float(p2 / max(p1 + p2, 1e-30)),
            "rel_err_if_p2_removed": float(p1 / max(ref, 1e-30)),
            "rel_err_if_p1_removed": float(p2 / max(ref, 1e-30)),
            "mean_massive_channels": float(a["n_massive"].sum() / max(a["n"].sum(), 1)),
            "rel_err_by_site": site_rel,
            "rel_err_by_layer": [float(a["total"][l].sum() / max(a["ref"][l].sum(), 1e-30)) for l in range(n_layers)],
            "n_rows": int(a["n"][:, 0].sum()),
            "abs_err_per_row": float(tot / nrow),            # mean ||delta w^T||^2 per (row, layer, site)
            "ref_per_row": float(ref / nrow),                # mean ||x w^T||^2 per (row, layer, site)
            "abs_p1_per_row": float(p1 / nrow), "abs_p2_per_row": float(p2 / nrow),
            "abs_err_by_site": {s: float(a["total"][:, j].sum() / max(a["n"][:, j].sum(), 1)) for j, s in enumerate(SITES)},
        }
    for key in list(out["results"]):
        rot_name, fmt, gname = key.split("|")
        base = out["results"].get(f"{rot_name}|{fmt}|normal")
        if base:
            out["results"][key]["abs_ratio_vs_normal"] = out["results"][key]["abs_err_per_row"] / max(base["abs_err_per_row"], 1e-30)
            out["results"][key]["rel_ratio_vs_normal"] = out["results"][key]["rel_err"] / max(base["rel_err"], 1e-30)
    out["kv"] = {}
    for (fmt, tname, part), a in KV["acc"].items():
        out["kv"][f"{fmt}|{tname}|{part}"] = {
            "rel_change": float(a["num"].sum() / max(a["den"].sum(), 1e-30)),
            "rel_change_by_layer": [float(a["num"][l] / max(a["den"][l], 1e-30)) for l in range(n_layers)],
            "n_calls": int(a["cnt"].sum()),
            "mean_mass": float(a["mass"].sum() / max(a["cnt"].sum(), 1)),      # attention share received by the set
            "mean_size": float(a["size"].sum() / max(a["cnt"].sum(), 1)),
            # per-call relative change divided by per-call attention mass, averaged: "error per unit attention"
            "rel_change_per_call_mean": float(a["rel_sum"].sum() / max(a["cnt"].sum(), 1)),
            "rel_change_per_mass": float(a["rel_sum"].sum() / max(a["mass"].sum(), 1e-30)),
        }
    out["kv_check_rel_err_vs_sdpa"] = KV.get("check_rel_err_vs_sdpa")
    out["kv_pc"] = {f"{fmt}|{sname}|{var}": float(a["num"].sum() / max(a["den"].sum(), 1e-30))
                    for (fmt, sname, var), a in KV["pc"].items()}
    out["kv_pc_damage"] = {}
    for fmt, _ in PC_FMTS:
        allq = out["kv_pc"].get(f"{fmt}|all|all")
        for sname in ["sink", "rand", "knorm", "atttop"]:
            vin, vex = out["kv_pc"].get(f"{fmt}|{sname}|in"), out["kv_pc"].get(f"{fmt}|{sname}|ex")
            if vin is None:
                continue
            out["kv_pc_damage"][f"{fmt}|{sname}"] = {
                "rel_in": vin, "rel_ex": vex, "rel_all": allq,
                "damage_frac_of_in": (vin - vex) / max(vin, 1e-30),   # share of the error removed by excluding S
                "gain_vs_all": (allq - vex) / max(allq, 1e-30)}       # total gain of protecting S (own + cross)
    gs = KV["pc_stats"].get("groupmax_from_sink")
    out["kv_pc_groupmax_from_sink"] = float(gs["hit"].sum() / max(gs["cnt"].sum(), 1)) if gs else None
    out["kv_note"] = ("rel_change = sum ||O'-O||^2 / sum ||O||^2 over question rows, heads, layers; "
                      "'rand' pools %d equal-size random draws of ordinary visual tokens per image; "
                      "images without sinks contribute only to tsink/allimg; att_top / att_mass are count- and "
                      "attention-mass-matched non-sink image tokens chosen per layer (cap %d)" % (N_RAND, MASS_CAP))
    json.dump(out, open(os.path.join(args.out_dir, "summary.json"), "w"), indent=1)
    brief = {k: {kk: (round(v[kk], 5) if isinstance(v[kk], float) else v[kk]) for kk in
                 ["rel_err", "p2_share", "rel_err_if_p2_removed", "rel_err_if_p1_removed", "n_rows"]}
             for k, v in sorted(out["results"].items())}
    print(json.dumps(brief, indent=1), flush=True)
    print(json.dumps({k: round(v["rel_change"], 6) for k, v in sorted(out["kv"].items())}, indent=1), flush=True)
    print("peak_mem_gb", round(torch.cuda.max_memory_allocated() / 2**30, 2), "elapsed", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
