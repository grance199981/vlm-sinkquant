"""MathVista testmini subset: N questions, stratified by (metadata.category, question_type) with
proportional allocation (largest remainder), seed S. Writes images/<pid>.png (original bytes of
decoded_image) and questions.jsonl. Usage: python make_mathvista_subset.py ROOT N SEED"""
import glob
import json
import os
import random
import sys

import pyarrow.parquet as pq

root, N, seed = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
out = os.path.join(root, "subset")
os.makedirs(os.path.join(out, "images"), exist_ok=True)
cols = ["pid", "question", "query", "choices", "unit", "precision", "answer", "question_type", "answer_type", "metadata"]
rows = []
for f in sorted(glob.glob(os.path.join(root, "data", "testmini-*.parquet"))):
    rows += pq.read_table(f, columns=cols).to_pylist()
assert len(rows) == 1000, len(rows)
strata = {}
for r in rows:
    strata.setdefault((r["metadata"]["category"], r["question_type"]), []).append(r)
keys = sorted(strata)
quota = {k: N * len(strata[k]) / len(rows) for k in keys}
alloc = {k: int(quota[k]) for k in keys}
for k in sorted(keys, key=lambda k: -(quota[k] - alloc[k]))[:N - sum(alloc.values())]:
    alloc[k] += 1
rng = random.Random(seed)
pick = []
for k in keys:
    pick += rng.sample(sorted(strata[k], key=lambda r: int(r["pid"])), alloc[k])
need = {r["pid"] for r in pick}
for f in sorted(glob.glob(os.path.join(root, "data", "testmini-*.parquet"))):
    for r in pq.read_table(f, columns=["pid", "decoded_image"]).to_pylist():
        if r["pid"] in need:
            open(os.path.join(out, "images", f"{r['pid']}.png"), "wb").write(r["decoded_image"]["bytes"])
with open(os.path.join(out, "questions.jsonl"), "w") as fo:
    for r in sorted(pick, key=lambda r: int(r["pid"])):
        fo.write(json.dumps({"image_id": f"m{r['pid']}", "file": f"images/{r['pid']}.png", "query": r["query"],
                             "question": r["question"], "choices": r["choices"], "unit": r["unit"],
                             "precision": r["precision"], "answer": r["answer"], "question_type": r["question_type"],
                             "answer_type": r["answer_type"], "category": r["metadata"]["category"],
                             "task": r["metadata"]["task"], "skills": r["metadata"]["skills"]}) + "\n")
print(len(pick), "questions;", {f"{k[0]}|{k[1]}": alloc[k] for k in keys})
