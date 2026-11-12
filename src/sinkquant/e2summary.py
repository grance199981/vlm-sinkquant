"""Summary statistics for the E2 protection kill test (numpy only; usable locally and remotely).

Records are the per-item dicts written by scripts/e2_protect.py. Primary metrics: KL(FP16 || q)
over the option letters and agreement of the predicted letter with FP16; accuracy and the
log-prob of the correct letter are reported too. Tests are paired bootstrap CIs over items.
"""
import numpy as np

from sinkquant import evalstats as E

FP = "fp16|None|none"


def _arrs(recs, key):
    acc = np.array([r["res"][key]["pred"] == r["label"] for r in recs], dtype=np.float64)
    lp = np.array([np.log(max(r["res"][key]["p"][r["options"].index(r["label"])], 1e-12)) for r in recs])
    klv = np.array([E.kl(r["res"][FP]["p"], r["res"][key]["p"]) for r in recs])
    agree = np.array([r["res"][key]["pred"] == r["res"][FP]["pred"] for r in recs], dtype=np.float64)
    return {"acc": acc, "logp": lp, "kl": klv, "agree": agree}


def summarize_records(recs, settings, formats, n_rand, protections=("vsink", "att", "tsink")):
    keys = [FP] + [f"{st}|{fm}|{pr}" for st in settings for fm in formats
                   for pr in ["none", *protections] + [f"rand#{r}" for r in range(n_rand)]]
    A = {k: _arrs(recs, k) for k in keys}
    table = {k: {"acc": float(a["acc"].mean()), "logp_correct": float(a["logp"].mean()),
                 "kl_vs_fp16": float(a["kl"].mean()), "agree_with_fp16": float(a["agree"].mean())}
             for k, a in A.items()}
    tests = {}
    for st in settings:
        for fm in formats:
            base = f"{st}|{fm}"
            none = A[f"{base}|none"]
            rand = {m: np.mean([A[f"{base}|rand#{r}"][m] for r in range(n_rand)], 0) for m in ["acc", "logp", "kl", "agree"]}
            t = {"degradation_agree(1-agree_none)": E.paired_bootstrap(1.0 - none["agree"]),
                 "degradation_acc(fp16-none)": E.paired_bootstrap(A[FP]["acc"] - none["acc"]),
                 "kl_none": E.paired_bootstrap(none["kl"])}
            for pr in protections:
                a = A[f"{base}|{pr}"]
                t[f"rand-{pr} kl"] = E.paired_bootstrap(rand["kl"] - a["kl"])        # > 0: protection lowers KL more than random
                t[f"none-{pr} kl"] = E.paired_bootstrap(none["kl"] - a["kl"])        # > 0: protection helps at all
                t[f"{pr}-rand agree"] = E.paired_bootstrap(a["agree"] - rand["agree"])
                t[f"{pr}-rand acc"] = E.paired_bootstrap(a["acc"] - rand["acc"])
            t["vsink-att kl"] = E.paired_bootstrap(A[f"{base}|att"]["kl"] - A[f"{base}|vsink"]["kl"])  # > 0: sinks better
            tests[base] = t
    return {"n_items": len(recs), "fp16_acc": table[FP]["acc"],
            "fp16_top_is_letter_rate": float(np.mean([r["res"][FP]["top_is_letter"] for r in recs])),
            "mean_k": float(np.mean([r["k"] for r in recs])), "table": table, "tests": tests,
            "note": "tests: (mean, lo95, hi95) of paired per-item differences; rand = mean of %d draws" % n_rand}
