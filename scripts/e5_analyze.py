"""Analyze E5 teacher-forced runs (pre-registered criteria, incl. the amendment written before results).

Usage: PYTHONPATH=src python e5_analyze.py OUT_JSON items.jsonl:per_item.tar [items.jsonl:per_item.tar ...]

Per item, the analysed positions are those of the thinking segment (k < think_end; chains without
'</think>' use every position); items with rep32_last1k > 0.5 (loops) are excluded from the primary
analysis and summarised separately. Quarters are the first and last quarter of an item's analysed
positions.
  (a) per config: mean KL in the last quarter minus the first quarter (paired over items), raw and
      normalised as sum(KL) / sum(BF16 entropy) within the quarter. Primary: kv4_all, kv3_all, kv4had_all.
  (b) image dependence ratio R = mean dimg_q / mean dimg_bf16 per quarter, last minus first, for
      q in kv4_all, kv3_all (negative = quantization makes image dependence decay faster).
Curves (mean over items of per-decile means of each item's analysed positions) are saved for plots.
"""
import io
import json
import sys
import tarfile

import numpy as np

from sinkquant import evalstats as E

KL_CFGS = ["kv4_all", "kv3_all", "kv2_all", "kv4had_all", "kv4_img", "kv4_gen", "kv3_img", "kv3_gen"]
PRIMARY = ["kv4_all", "kv3_all", "kv4had_all"]


def load(pairs):
    items = []
    for pair in pairs:
        meta_f, tar_f = pair.split(":")
        meta = {}
        for line in open(meta_f, encoding="utf-8"):
            r = json.loads(line)
            meta[r["image_id"]] = r
        with tarfile.open(tar_f) as tf:
            for m in tf.getmembers():
                if m.name.endswith(".npz"):
                    iid = m.name.rsplit("/", 1)[-1][:-4]
                    if iid in meta:
                        z = np.load(io.BytesIO(tf.extractfile(m).read()))
                        items.append((meta[iid], {k: z[k] for k in z.files}))
    return items


def span(meta, n):
    te = meta["think_end"]
    return n if te is None or te < 0 else min(te, n)


def quarters(a):
    q = max(1, len(a) // 4)
    return a[:q], a[-q:]


def deciles(a):
    return np.array([c.mean() if len(c) else np.nan for c in np.array_split(a, 10)])


def main():
    out, pairs = sys.argv[1], sys.argv[2:]
    items = load(pairs)
    loops = [m for m, _ in items if (m.get("rep32_last1k") or 0) > 0.5]
    keep = [(m, z) for m, z in items if (m.get("rep32_last1k") or 0) <= 0.5 and span(m, len(z["ent_bf16"])) >= 16]
    res = {"n_items_total": len(items), "n_loops_excluded": len(loops), "n_analysed": len(keep),
           "top20_rate_median": float(np.median([m["top20_rate"] for m, _ in items])) if items else None,
           "top20_rate_min": float(np.min([m["top20_rate"] for m, _ in items])) if items else None,
           "think_len_median": float(np.median([span(m, len(z["ent_bf16"])) for m, z in keep])) if keep else None,
           "acc_bf16": float(np.mean([m["correct"] for m, _ in items])) if items else None,
           "a": {}, "b": {}, "curves": {}}
    for c in KL_CFGS:
        dr, dn, first, last = [], [], [], []
        curves = []
        for m, z in keep:
            s = span(m, len(z["ent_bf16"]))
            kl, ent = z[f"kl_{c}"][:s].astype(np.float64), z["ent_bf16"][:s].astype(np.float64)
            (k1, k2), (e1, e2) = quarters(kl), quarters(ent)
            dr.append(k2.mean() - k1.mean())
            dn.append(k2.sum() / max(e2.sum(), 1e-9) - k1.sum() / max(e1.sum(), 1e-9))
            first.append(k1.mean())
            last.append(k2.mean())
            curves.append(deciles(kl))
        res["a"][c] = {"first_q_mean": float(np.mean(first)), "last_q_mean": float(np.mean(last)),
                       "last_minus_first": E.paired_bootstrap(np.array(dr)),
                       "last_minus_first_normalised": E.paired_bootstrap(np.array(dn)), "primary": c in PRIMARY}
        res["curves"][f"kl_{c}"] = np.nanmean(np.stack(curves), 0).tolist()
    for q in ["kv4_all", "kv3_all"]:
        d, curves_q, curves_b = [], [], []
        for m, z in keep:
            s = span(m, len(z["ent_bf16"]))
            dq, db = z[f"dimg_{q}"][:s].astype(np.float64), z["dimg_bf16"][:s].astype(np.float64)
            (q1, q2), (b1, b2) = quarters(dq), quarters(db)
            d.append(q2.mean() / max(b2.mean(), 1e-9) - q1.mean() / max(b1.mean(), 1e-9))
            curves_q.append(deciles(dq))
            curves_b.append(deciles(db))
        res["b"][q] = {"ratio_last_minus_first": E.paired_bootstrap(np.array(d))}
        res["curves"][f"dimg_{q}"] = np.nanmean(np.stack(curves_q), 0).tolist()
        res["curves"]["dimg_bf16"] = np.nanmean(np.stack(curves_b), 0).tolist()
    for p in ["bf16", "kv4_all", "kv3_all"]:
        cs = [deciles(z[f"attn_{p}"].astype(np.float64).mean(0)[:span(m, z[f"attn_{p}"].shape[1])]) for m, z in keep]
        res["curves"][f"attn_img_{p}"] = np.nanmean(np.stack(cs), 0).tolist()
    json.dump(res, open(out, "w"), indent=1, default=float)
    sig = lambda t: " *" if t[1] > 0 or t[2] < 0 else ""
    f = lambda t: f"{t[0]:+.4f} [{t[1]:+.4f}, {t[2]:+.4f}]{sig(t)}"
    print({k: v for k, v in res.items() if k not in ("a", "b", "curves")})
    for c, t in res["a"].items():
        print(f"(a) {c:11s}{'*P' if t['primary'] else '  '} first={t['first_q_mean']:.4f} last={t['last_q_mean']:.4f} "
              f"diff={f(t['last_minus_first'])} norm={f(t['last_minus_first_normalised'])}")
    for q, t in res["b"].items():
        print(f"(b) {q}: ratio last-first = {f(t['ratio_last_minus_first'])}")
    for k in ["dimg_bf16", "attn_img_bf16", "attn_img_kv4_all"]:
        print(k, np.round(res["curves"][k], 4).tolist())


if __name__ == "__main__":
    main()
