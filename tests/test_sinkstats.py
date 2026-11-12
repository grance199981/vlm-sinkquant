import numpy as np

from sinkquant import sinkstats as S


def test_auc_known_values():
    assert S.auc([0.1, 0.2, 0.8, 0.9], [0, 0, 1, 1]) == 1.0
    assert S.auc([0.9, 0.8, 0.2, 0.1], [0, 0, 1, 1]) == 0.0
    assert S.auc([0.5, 0.5, 0.5, 0.5], [0, 1, 0, 1]) == 0.5


def test_auc_random_labels_near_half():
    rng = np.random.default_rng(0)
    s = rng.normal(size=4000)
    y = rng.random(4000) < 0.1
    assert abs(S.auc(s, y) - 0.5) < 0.05


def test_received_attention_and_share():
    H, L = 2, 5
    attn = np.zeros((H, L, L))
    attn[:, :, 1] = 0.7          # every query puts 0.7 on token 1
    attn[:, :, 3] = 0.3
    a = S.received_attention(attn, np.array([3, 4]))
    assert np.allclose(a[[1, 3]], [0.7, 0.3])
    r = S.in_image_share(a, np.array([1, 2, 3]))
    assert np.isclose(r.sum(), 1.0) and np.isclose(r[0], 0.7)


def test_text_sink_position_skips_image():
    a = np.array([0.5, 0.9, 0.1, 0.2])
    img = np.array([False, True, False, False])
    assert S.text_sink_position(a, img) == 0


def test_centered_cosine_detects_aligned_token():
    rng = np.random.default_rng(1)
    Hkv, L, D = 2, 50, 16
    keys = rng.normal(size=(Hkv, L, D))
    keys[:, 0] = 10.0                      # text sink at position 0
    img_idx = np.arange(1, L)
    keys[:, 7] = keys[:, 1:].mean(1) + 5 * (keys[:, 0] - keys[:, 1:].mean(1))  # aligned image token
    c = S.centered_key_cosine(keys, img_idx, 0)
    assert np.argmax(c) == 6               # position 7 is index 6 within the image span
    assert c[6] > 0.9 and np.median(c) < 0.5


def test_sink_mask_and_peak():
    r = np.array([0.0, 0.5, 0.004, 0.496]) / 1.0
    inside = np.array([False, False, False, True])
    m = S.sink_mask(r, inside, ratio=1.5)  # n*r >= 1.5 -> r >= 0.375
    assert m.tolist() == [False, True, False, False]
    assert S.peak_layer([0.9, 0.1, 0.3, 0.2]) == 2   # first quarter (layer 0) excluded


def test_sink_mask_abs_is_resolution_free():
    r = np.array([0.05, 0.02, 0.93])
    inside = np.array([False, False, True])
    assert S.sink_mask_abs(r, inside, 0.03).tolist() == [True, False, False]


def test_sink_dims_and_ratio():
    h_sink = np.zeros(4096)
    h_sink[[3, 40]] = [1000.0, -800.0]
    h_sink[10] = 5.0
    dims = S.sink_dims(h_sink)
    assert dims.tolist() == [3, 40]
    h = np.ones((2, 4096))
    h[0, 40] = 200.0                       # token 0 is massive in a sink dim
    h[1, 10] = 200.0                       # token 1 is massive in a non-sink dim
    r = S.sink_dim_ratio(h, dims)
    assert r[0] > 20 and r[1] < 2


def test_internvl_token_centers_layout():
    # 2 cols x 1 row over a 896x448 original: tile 0 = left half, tile 1 = right half, + thumbnail
    xy, tid = S.internvl_token_centers(2, 1, (896, 448))
    assert len(xy) == 3 * 256 and (tid == -1).sum() == 256
    assert xy[tid == 0][:, 0].max() < 448 and xy[tid == 1][:, 0].min() > 448
    # row-major inside a tile: token 1 is right of token 0, token 16 is below token 0
    t0 = xy[tid == 0]
    assert t0[1, 0] > t0[0, 0] and t0[1, 1] == t0[0, 1] and t0[16, 1] > t0[0, 1]
    # thumbnail spans the whole image
    th = xy[tid == -1]
    assert th[:, 0].min() < 30 and th[:, 0].max() > 866
    inside = S.points_inside(xy, (0, 0, 448, 448))
    assert inside[tid == 0].all() and not inside[tid == 1].any()
    # single tile: no thumbnail
    xy1, tid1 = S.internvl_token_centers(1, 1, (448, 448))
    assert len(xy1) == 256 and (tid1 == -1).sum() == 0


def test_token_boxes_inside_row_major():
    # 4x4 merged tokens over a 128x128 resized image (grid 8x8 patches of 16px)
    inside = S.token_boxes_inside(8, 8, (0, 0, 64, 32), (128, 128))
    grid = inside.reshape(4, 4)
    assert grid[0, :2].all() and not grid[0, 2:].any() and not grid[1:].any()
