"""Summarize E3 (activation-INT4 'blindness') per_item files.

Usage: PYTHONPATH=src python e3_analyze.py OUT_JSON DATASET per_item.jsonl [per_item.jsonl ...]
Per configuration: accuracy, agreement with FP16, KL(FP16 || cfg) over the options, share of items whose
top token is an option, median option mass; for POPE also yes rate, miss rate (label yes -> no) and
false-alarm rate (label no -> yes). Paired bootstrap 95% CIs for accuracy differences to FP16 and for
the key contrasts. Diagnostics are averaged over items per layer.
"""
import json
import sys

import numpy as np

from sinkquant import evalstats as E

out, ds, files = sys.argv[1], sys.argv[2], sys.argv[3:]
recs = {}
for f in files:
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        recs.setdefault(r["image_id"], r)
recs = list(recs.values())
cfgs = list(recs[0]["res"])
FP = "fp16"


def arr(c, fn):
    return np.array([fn(r, r["res"][c]) for r in recs], dtype=np.float64)


acc = {c: arr(c, lambda r, x: x["pred"] == r["label"]) for c in cfgs}
table = {}
for c in cfgs:
    t = {"acc": acc[c].mean(),
         "agree_fp16": arr(c, lambda r, x: x["pred"] == r["res"][FP]["pred"]).mean(),
         "kl_fp16": arr(c, lambda r, x: E.kl(r["res"][FP]["p"], x["p"])).mean(),
         "top_is_option": arr(c, lambda r, x: x["top_is_option"]).mean(),
         "opt_mass_median": float(np.median(arr(c, lambda r, x: x["opt_mass"]))),
         "acc_minus_fp16": E.paired_bootstrap(acc[c] - acc[FP])}
    if ds == "pope":
        yes = arr(c, lambda r, x: x["pred"] == "yes")
        lab = np.array([r["label"] == "yes" for r in recs])
        t.update(yes_rate=yes.mean(), miss_rate=1 - yes[lab].mean(), false_alarm=yes[~lab].mean(),
                 p_yes_mean=arr(c, lambda r, x: x["p"][r["options"].index("yes")]).mean())
    table[c] = t
tests = {}
for f in ["int4", "mxfp4", "nvfp4", "had_int3"]:
    if f"{f}_visonly" in acc:
        tests[f"{f} visonly-textonly acc"] = E.paired_bootstrap(acc[f"{f}_visonly"] - acc[f"{f}_textonly"])
if "int4_all" in acc:
    tests["had-all acc"] = E.paired_bootstrap(acc["int4_had"] - acc["int4_all"])
if "mxfp4_had" in acc:
    tests["mxfp4_had-mxfp4_all acc"] = E.paired_bootstrap(acc["mxfp4_had"] - acc["mxfp4_all"])
if ds == "pope" and "int4_visonly" in acc:
    lab = np.array([r["label"] == "yes" for r in recs])
    for c in ["int4_visonly", "int4_had", "int4_top4", "int4_top16"]:
        yc = arr(c, lambda r, x: x["pred"] == "yes")
        yf = arr(FP, lambda r, x: x["pred"] == "yes")
        tests[f"{c} miss-fp16 miss (yes items)"] = E.paired_bootstrap((1 - yc[lab]) - (1 - yf[lab]))
        tests[f"{c} fa-fp16 fa (no items)"] = E.paired_bootstrap(yc[~lab] - yf[~lab])
diag = {}
for m in ["q", "down"]:
    d = np.array([r["diag"][m] for r in recs], dtype=np.float64)      # [N, L, 6]
    diag[m] = {"hit_img": d[:, :, 0].mean(0).round(4).tolist(), "hit_txt": d[:, :, 1].mean(0).round(4).tolist(),
               "ratio_img": np.median(d[:, :, 2], 0).round(2).tolist(), "ratio_txt": np.median(d[:, :, 3], 0).round(2).tolist(),
               "zero_img": d[:, :, 4].mean(0).round(4).tolist(), "zero_txt": d[:, :, 5].mean(0).round(4).tolist()}
res = {"dataset": ds, "n_items": len(recs), "sources": files, "table": table, "tests": tests, "diag": diag}
json.dump(res, open(out, "w"), indent=1, default=float)
for c, t in table.items():
    print(f"{c:15s} " + " ".join(f"{k}={v:.3f}" if isinstance(v, float) else f"{k}=[{v[0]:+.3f},{v[1]:+.3f},{v[2]:+.3f}]"
                                 for k, v in t.items()))
for k, v in tests.items():
    print(f"{k:40s} {v[0]:+.3f} [{v[1]:+.3f}, {v[2]:+.3f}]" + (" *" if v[1] > 0 or v[2] < 0 else ""))
for m in diag:
    for k in ["hit_img", "zero_img", "zero_txt"]:
        print(m, k, "mean over layers", round(float(np.mean(diag[m][k])), 3), "max", round(float(np.max(diag[m][k])), 3))
