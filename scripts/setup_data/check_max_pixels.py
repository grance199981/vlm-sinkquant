"""Check that capping image_processor longest_edge changes only oversized images (CPU only).
Usage: python check_max_pixels.py MODEL_DIR DATA_ROOT CAP id1 id2 ..."""
import json
import os
import sys

from PIL import Image
from transformers import AutoProcessor

md, root, cap, ids = sys.argv[1], sys.argv[2], int(sys.argv[3]), sys.argv[4:]
rows = {json.loads(l)["image_id"]: json.loads(l) for l in open(os.path.join(root, "subset", "questions.jsonl"))}
for capped in (False, True):
    proc = AutoProcessor.from_pretrained(md)
    if capped:
        proc.image_processor.size["longest_edge"] = cap
        proc.image_processor.max_pixels = cap
    for i in ids:
        img = Image.open(os.path.join(root, "subset", rows[i]["file"])).convert("RGB")
        out = proc.image_processor(images=[img], return_tensors="pt")
        thw = out["image_grid_thw"][0].tolist()
        print(f"capped={capped} {i} {img.size} grid_thw={thw} llm_tokens={thw[0] * thw[1] * thw[2] // 4}")
