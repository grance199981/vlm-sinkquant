"""Smoke test for InternVL3.5-HF with dynamic tiling (crop_to_patches, up to 12 tiles).

1. Layout check: a 4-colour quadrant image is tiled; using sinkstats.internvl_token_centers
   we assign each visual token to a quadrant and require that tokens are closer (cosine) to
   their own quadrant's mean feature than to any other quadrant's. A row/column or tile-order
   mistake in the mapping would break this.
2. Prefill with the R1 system prompt: eager attention weights from the language model,
   hidden states, KV cache shapes; then a short greedy generation.

Usage: CUDA_VISIBLE_DEVICES=<GPU UUID> python smoke_internvl.py <model_dir> <out_json>
"""
import json
import sys
import time

import numpy as np
import torch
import transformers
from PIL import Image
from transformers import AutoProcessor, InternVLForConditionalGeneration
from transformers.models.got_ocr2.image_processing_got_ocr2 import get_optimal_tiled_canvas

from sinkquant import sinkstats as S

R1_SYSTEM_PROMPT = """
You are an AI assistant that rigorously follows this response protocol:

1. First, conduct a detailed analysis of the question. Consider different angles, potential solutions, and reason through the problem step-by-step. Enclose this entire thinking process within <think> and </think> tags.

2. After the thinking section, provide a clear, concise, and direct answer to the user's question. Separate the answer from the think section with a newline.

Ensure that the thinking process is thorough but remains focused on the query. The final answer should be standalone and not reference the thinking section.
""".strip()
TILE_KW = {"crop_to_patches": True, "max_patches": 12, "min_patches": 1}


def main(model_dir, out_path):
    assert torch.cuda.device_count() == 1
    gpu = torch.cuda.get_device_name(0)
    proc = AutoProcessor.from_pretrained(model_dir)
    model = InternVLForConditionalGeneration.from_pretrained(
        model_dir, dtype=torch.bfloat16, attn_implementation="sdpa", device_map="cuda").eval()
    model.model.language_model.config._attn_implementation = "eager"
    cfg = model.config

    # 1) layout check. Colour regions split at x = W/2 and y = 0.375 H so that region borders
    # cut through the middle of tiles; classification is done *within* each tile (tile and
    # thumbnail features differ systematically, so a global comparison is not informative).
    W, H = 1344, 896
    yb = int(0.375 * H)
    arr = np.zeros((H, W, 3), np.uint8)
    arr[:yb, :W // 2] = (255, 0, 0)
    arr[:yb, W // 2:] = (0, 255, 0)
    arr[yb:, :W // 2] = (0, 0, 255)
    arr[yb:, W // 2:] = (255, 255, 0)
    img = Image.fromarray(arr)
    pix = proc.image_processor(images=[img], return_tensors="pt", **TILE_KW)["pixel_values"]
    ncols, nrows = get_optimal_tiled_canvas((H, W), (448, 448), 1, 12)
    with torch.no_grad():
        feats = model.model.get_image_features(
            pixel_values=pix.to("cuda", torch.bfloat16),
            vision_feature_layer=cfg.vision_feature_layer,
            vision_feature_select_strategy=cfg.vision_feature_select_strategy)
    feats = feats.reshape(-1, feats.shape[-1]).float().cpu().numpy()
    xy, tid = S.internvl_token_centers(ncols, nrows, (W, H))
    assert len(xy) == len(feats), (len(xy), feats.shape, ncols, nrows, pix.shape)
    f = feats / np.linalg.norm(feats, axis=1, keepdims=True)

    def within_tile_acc(xy_map):
        hits, total = 0, 0
        for t in np.unique(tid):
            m = tid == t
            p = xy_map[m]
            tok_w = (W / ncols / 16) if t >= 0 else (W / 16)
            tok_h = (H / nrows / 16) if t >= 0 else (H / 16)
            keep = (np.abs(p[:, 0] - W / 2) > 1.5 * tok_w) & (np.abs(p[:, 1] - yb) > 1.5 * tok_h)
            q = (p[:, 0] >= W / 2).astype(int) + 2 * (p[:, 1] >= yb).astype(int)
            regions = [k for k in range(4) if (keep & (q == k)).sum() >= 8]
            if len(regions) < 2:
                continue
            ft = f[m]
            means = np.stack([ft[keep & (q == k)].mean(0) for k in regions])
            sel = keep & np.isin(q, regions)
            pred = np.array(regions)[(ft[sel] @ means.T).argmax(1)]
            hits += int((pred == q[sel]).sum())
            total += int(sel.sum())
        return hits / max(total, 1), total

    layout_acc, n_checked = within_tile_acc(xy)
    xy_t = xy.copy()
    A, B = np.meshgrid(np.arange(16), np.arange(16), indexing="ij")
    for t in np.unique(tid):
        m = tid == t
        loc = xy[m].reshape(16, 16, 2)
        xs, ys = loc[0, :, 0], loc[:, 0, 1]
        # transposed token grid: token (row a, col b) placed at (x of col a, y of row b)
        xy_t[m] = np.stack([xs[A], ys[B]], -1).reshape(-1, 2)
    layout_acc_transposed, _ = within_tile_acc(xy_t)

    # 2) prefill + generation with the R1 system prompt on the same image
    msgs = [{"role": "system", "content": [{"type": "text", "text": R1_SYSTEM_PROMPT}]},
            {"role": "user", "content": [{"type": "image", "image": img},
                                         {"type": "text", "text": "What color is the top-left quadrant? Answer with one word."}]}]
    text = proc.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    inputs = proc(text=[text], images=[img], return_tensors="pt", **TILE_KW).to("cuda", torch.bfloat16)
    ids = inputs["input_ids"][0].cpu().numpy()
    n_img = int((ids == cfg.image_token_id).sum())
    got = {}
    hooks = [l.self_attn.register_forward_hook(
        lambda m, i, o, k=k: got.__setitem__(k, None if o[1] is None else tuple(o[1].shape)))
        for k, l in enumerate(model.model.language_model.layers)]
    with torch.no_grad():
        t0 = time.time()
        out = model(**inputs, output_hidden_states=True, use_cache=True)
        t_fwd = time.time() - t0
        for h in hooks:
            h.remove()
        gen = model.generate(**inputs, max_new_tokens=48, do_sample=False)
    answer = proc.batch_decode(gen[:, ids.shape[0]:], skip_special_tokens=True)[0]
    last_img = int(np.nonzero(ids == cfg.image_token_id)[0][-1])
    rec = {
        "gpu": gpu, "transformers": transformers.__version__,
        "grid_cols_rows": [ncols, nrows], "pixel_values": list(pix.shape), "n_visual_tokens": n_img,
        "layout_acc": layout_acc, "layout_n_checked": n_checked, "layout_acc_transposed_control": layout_acc_transposed,
        "attn_shape_layer0": got.get(0), "n_layers": len(got),
        "n_hidden_states": len(out.hidden_states), "seq_len": int(len(ids)),
        "tokens_after_last_image": proc.tokenizer.decode(ids[last_img + 1:last_img + 4].tolist()),
        "first_token": proc.tokenizer.decode([int(ids[0])]),
        "answer": answer, "forward_s": round(t_fwd, 2),
        "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
    }
    print(json.dumps(rec, indent=1), flush=True)
    json.dump(rec, open(out_path, "w"), indent=1)
    np.savez_compressed(out_path.replace(".json", "_layout_feats.npz"), feats=feats.astype(np.float16),
                        xy=xy, tid=tid, W=W, H=H, yb=yb, ncols=ncols, nrows=nrows)
    # decision on the layout is made offline from the saved features (alternative mappings)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
