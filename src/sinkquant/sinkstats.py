"""Pure-numpy helpers for E0 sink detection (no torch, unit-testable on CPU).

Definitions follow arXiv 2610.15855 §3.1 / Eq. 2:
  a_j   = attention token j receives, averaged over heads and question tokens
  r_j   = a_j / sum_{j' in image} a_j'          (share of in-image attention)
  sink  : image token with n * r_j >= 100 lying outside the target box
  t*    = argmax_{j not in image} a_j at the middle full-attention layer
  c~_j  = cos(k_j - mean_img_k, k_t* - mean_img_k), per KV head, then head-averaged
  peak  = layer with the largest sink share among the last three quarters of layers
"""
import numpy as np


def received_attention(attn, query_idx):
    """attn: [H, L, L] softmax probs of one layer; returns a: [L]."""
    return attn[:, query_idx, :].mean(axis=(0, 1))


def in_image_share(a, img_idx):
    ai = a[img_idx]
    return ai / ai.sum()


def text_sink_position(a_mid, img_mask):
    """Non-image position receiving the most attention at the middle layer."""
    masked = np.where(img_mask, -np.inf, a_mid)
    return int(np.argmax(masked))


def centered_key_cosine(keys, img_idx, t_star):
    """keys: [Hkv, L, D]. Returns head-averaged centered cosine for image tokens: [n]."""
    k_img = keys[:, img_idx, :]                       # [Hkv, n, D]
    mean = k_img.mean(axis=1, keepdims=True)          # [Hkv, 1, D]
    u = keys[:, t_star:t_star + 1, :] - mean          # [Hkv, 1, D]
    v = k_img - mean
    num = (v * u).sum(-1)
    den = np.linalg.norm(v, axis=-1) * np.linalg.norm(u, axis=-1) + 1e-12
    return (num / den).mean(axis=0)


def raw_key_cosine(keys, img_idx, t_star):
    k_img = keys[:, img_idx, :]
    u = keys[:, t_star:t_star + 1, :]
    num = (k_img * u).sum(-1)
    den = np.linalg.norm(k_img, axis=-1) * np.linalg.norm(u, axis=-1) + 1e-12
    return (num / den).mean(axis=0)


def sink_mask(r, inside_box, ratio=100.0):
    n = r.shape[0]
    return (n * r >= ratio) & (~inside_box)


def sink_mask_abs(r, inside_box, min_share=0.03):
    """Resolution-free variant: a token holding >= min_share of in-image attention.

    The paper's n*r >= 100 depends on n (V* has ~3300 image tokens, so it means r >= ~3%);
    at n ~ 260 (COCO) it would require r >= 38%.
    """
    return (r >= min_share) & (~inside_box)


def sink_dims(h_sink, frac=0.25):
    """Massive dims of the text-sink hidden state: |h_d| >= frac * max |h|."""
    a = np.abs(h_sink)
    return np.nonzero(a >= frac * a.max())[0]


def sink_dim_ratio(h, dims):
    """Kang et al.-style score: max over sink dims of |h_jd| / rms(h_j). h: [n, D]."""
    rms = np.sqrt((h.astype(np.float64) ** 2).mean(-1)) + 1e-12
    return np.abs(h[:, dims]).max(-1) / rms


def peak_layer(shares):
    """shares: [num_layers] mean sink share per layer; search the last 3/4."""
    n = len(shares)
    start = n // 4
    return start + int(np.argmax(np.asarray(shares)[start:]))


def auc(scores, labels):
    """ROC AUC via the rank-sum (Mann-Whitney U) formula, ties averaged."""
    scores = np.asarray(scores, dtype=np.float64)
    labels = np.asarray(labels, dtype=bool)
    n_pos, n_neg = labels.sum(), (~labels).sum()
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty(len(scores), dtype=np.float64)
    sorted_scores = scores[order]
    i = 0
    while i < len(scores):
        j = i
        while j + 1 < len(scores) and sorted_scores[j + 1] == sorted_scores[i]:
            j += 1
        ranks[order[i:j + 1]] = (i + j) / 2.0 + 1.0
        i = j + 1
    return float((ranks[labels].sum() - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg))


def internvl_token_centers(cols, rows, orig_wh, tile=448, per_side=16):
    """Token centres (in original-image pixels) for InternVL dynamic tiling.

    Tiles are row-major over a (cols*tile) x (rows*tile) resized canvas, each tile giving
    per_side x per_side row-major tokens; when there is more than one tile a thumbnail
    (the whole image resized to tile x tile) follows. Returns (xy [N, 2], tile_id [N]),
    tile_id = -1 for thumbnail tokens.
    """
    step = tile / per_side
    g = (np.arange(per_side) + 0.5) * step
    gy, gx = np.meshgrid(g, g, indexing="ij")
    sx, sy = orig_wh[0] / (cols * tile), orig_wh[1] / (rows * tile)
    xy, tid = [], []
    for i in range(cols * rows):
        c, r = i % cols, i // cols
        xy.append(np.stack([(c * tile + gx.ravel()) * sx, (r * tile + gy.ravel()) * sy], 1))
        tid.append(np.full(per_side * per_side, i))
    if cols * rows > 1:
        xy.append(np.stack([gx.ravel() * orig_wh[0] / tile, gy.ravel() * orig_wh[1] / tile], 1))
        tid.append(np.full(per_side * per_side, -1))
    return np.concatenate(xy), np.concatenate(tid)


def points_inside(xy, box_xyxy):
    x0, y0, x1, y1 = box_xyxy
    return (xy[:, 0] >= x0) & (xy[:, 0] < x1) & (xy[:, 1] >= y0) & (xy[:, 1] < y1)


def token_boxes_inside(grid_h, grid_w, box_xyxy, orig_wh, merge=2, patch=16):
    """Which merged visual tokens have their centre inside a box.

    grid_h/grid_w: patch grid from image_grid_thw (before the 2x2 merge).
    box_xyxy: box in original-image pixels. Returns bool [grid_h//merge * grid_w//merge],
    row-major, matching the order of image tokens in the sequence.
    """
    rh, rw = grid_h * patch, grid_w * patch            # resized image size
    sx, sy = rw / orig_wh[0], rh / orig_wh[1]
    x0, y0, x1, y1 = box_xyxy[0] * sx, box_xyxy[1] * sy, box_xyxy[2] * sx, box_xyxy[3] * sy
    th, tw = grid_h // merge, grid_w // merge
    cell = patch * merge
    cy = (np.arange(th) + 0.5) * cell
    cx = (np.arange(tw) + 0.5) * cell
    inside = ((cy[:, None] >= y0) & (cy[:, None] < y1)) & ((cx[None, :] >= x0) & (cx[None, :] < x1))
    return inside.reshape(-1)
