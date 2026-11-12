"""Merge E2 per_item.jsonl shards (dedupe by image_id) and write one summary.

Usage: PYTHONPATH=src python e2_merge.py OUT_DIR shard1/per_item.jsonl shard2/per_item.jsonl ...
"""
import json
import os
import sys

from sinkquant import e2summary

out_dir, files = sys.argv[1], sys.argv[2:]
recs = {}
for f in files:
    for line in open(f, encoding="utf-8"):
        r = json.loads(line)
        recs.setdefault(r["image_id"], r)
recs = [recs[k] for k in sorted(recs, key=lambda x: int(x.lstrip("qp")))]
summary = e2summary.summarize_records(recs, ["kv", "act"], ["int4_tok", "mxfp4"], 3)
summary["sources"] = files
os.makedirs(out_dir, exist_ok=True)
json.dump(summary, open(os.path.join(out_dir, "summary.json"), "w"), indent=1)
with open(os.path.join(out_dir, "per_item.jsonl"), "w", encoding="utf-8") as f:
    for r in recs:
        f.write(json.dumps(r) + "\n")
print(summary["n_items"], "items merged")
