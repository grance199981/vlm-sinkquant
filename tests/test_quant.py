import torch

from sinkquant import quant as Q


def _sink_row(d=4096, big=300.0, ch=1000):
    torch.manual_seed(0)
    x = torch.empty(1, d).uniform_(-1, 1)
    x[0, ch] = big
    return x


def test_int4_per_token_example_from_readme():
    x = _sink_row()
    q = Q.int4_per_token(x)
    assert torch.isclose(q[0, 1000], torch.tensor(300.0))          # max exactly represented
    others = torch.cat([q[0, :1000], q[0, 1001:]])
    assert (others == 0).all()                                     # whole row zeroed (P1)


def test_mxfp4_example_from_readme():
    x = _sink_row()
    q = Q.mxfp4(x)
    assert torch.isclose(q[0, 1000], torch.tensor(256.0))           # 300/64 = 4.69 -> 4 -> 256
    blk = q[0, 992:1024].clone()
    blk[1000 - 992] = 0
    assert (blk == 0).all()                                        # 31 neighbours zeroed
    assert (q[0, :992] != 0).float().mean() > 0.7                  # other blocks keep info


def test_mxfp4_grid_and_saturation():
    x = torch.tensor([[7.9, 1.0, 0.24, 0.26] + [0.0] * 28])
    q = Q.mxfp4(x)                                                 # amax 7.9 -> scale 2^(2-2)=1
    assert q[0, 0] == 6.0 and q[0, 1] == 1.0 and q[0, 2] == 0.0 and q[0, 3] == 0.5


def test_nvfp4_tensor_max_exact_and_small_block_error():
    x = _sink_row()
    q = Q.nvfp4(x)
    assert torch.isclose(q[0, 1000], torch.tensor(300.0), rtol=1e-3)
    x2 = torch.zeros(1, 32)
    x2[0, :16] = torch.linspace(-1, 1, 16)
    x2[0, 16] = 100.0                                              # tensor max in another block
    q2 = Q.nvfp4(x2)
    rel = ((q2[0, :16] - x2[0, :16]).abs().max() / x2[0, :16].abs().max()).item()
    assert rel < 0.2


def test_rotation_preserves_output_and_spreads_outlier():
    torch.manual_seed(1)
    d, o = 256, 64
    x = torch.randn(4, d)
    x[0, 7] = 500.0
    w = torch.randn(o, d) / d ** 0.5
    R = Q.random_orthogonal(d, seed=0)
    assert torch.allclose((x @ R) @ (w @ R).T, x @ w.T, atol=1e-2)
    assert (x @ R)[0].abs().max() < 0.3 * 500.0


def test_path_errors_split_int4_vs_mxfp4():
    torch.manual_seed(2)
    d, o = 4096, 128
    x = _sink_row(d)
    w = torch.randn(o, d) / d ** 0.5
    e_int = Q.path_errors(x, w, "int4_tok")
    e_mx = Q.path_errors(x, w, "mxfp4")
    # INT4 per token: P1 dominates (max exact -> p2 == 0); MXFP4: P2 dominates for this row
    assert e_int["p2"][0] == 0 and e_int["p1"][0] > 0
    assert e_mx["p2"][0] > e_mx["p1"][0]
    assert e_mx["p1"][0] < e_int["p1"][0]


def test_nvfp4_per_row_tensor_amax():
    x = torch.zeros(2, 16)
    x[:, 0] = 1.0
    x[:, 1] = 0.3
    big = Q.nvfp4(x, tensor_amax=torch.tensor([1.0, 17408.0]))
    # a huge tensor max (text sink) leaves E4M3 block scales near their lower range: still finite
    assert torch.isfinite(big).all()
    assert torch.isclose(big[0, 0], torch.tensor(1.0))


def test_hadamard_orders_are_orthogonal():
    for d in [16, 32, 128, 12, 192, 6144]:
        h = Q.hadamard_matrix(d)
        assert torch.allclose(h @ h.T, d * torch.eye(d, dtype=h.dtype)), d
        r = Q.random_hadamard(d, seed=3)
        assert torch.allclose(r @ r.T, torch.eye(d), atol=1e-5), d


def test_hadamard_spreads_outlier_and_preserves_output():
    torch.manual_seed(4)
    d, o = 192, 32
    x = torch.randn(3, d)
    x[0, 5] = 400.0
    w = torch.randn(o, d) / d ** 0.5
    R = Q.random_hadamard(d, seed=0)
    assert torch.allclose((x @ R) @ (w @ R).T, x @ w.T, atol=1e-2)
    assert (x @ R)[0].abs().max() < 0.2 * 400.0
    xb = Q.block_hadamard_rotate(x, 32)
    hb = Q.random_hadamard(32, seed=0)
    assert torch.allclose(xb[:, :32], x[:, :32] @ hb, atol=1e-4)


def test_quant_along_tokens_exclusion_removes_range_expansion():
    torch.manual_seed(5)
    x = torch.randn(4, 64) * 0.1                                      # [channels, tokens]
    x[:, 10] = 50.0                                                   # an outlier token
    excl = torch.zeros(64, dtype=torch.bool)
    excl[10] = True
    others = torch.ones(64, dtype=torch.bool)
    others[10] = False
    for fmt, blk in [("int4_asym", 32), ("mxfp4", 32), ("nvfp4", 16)]:
        e_in = (Q.quant_along_tokens(x, fmt, blk) - x)[:, others].pow(2).sum()
        e_ex = (Q.quant_along_tokens(x, fmt, blk, exclude=excl) - x)[:, others].pow(2).sum()
        assert e_ex < 0.2 * e_in, (fmt, float(e_in), float(e_ex))


def test_int4_keep_topc_recovers_small_values():
    x = _sink_row()
    assert Q.int4_zero_frac(x)[0] > 0.99                            # one big value zeroes the row
    q = Q.int4_keep_topc(x, 1)
    assert q[0, 1000] == x[0, 1000]                                 # outlier kept exactly
    rest = torch.cat([q[0, :1000] - x[0, :1000], q[0, 1001:] - x[0, 1001:]])
    assert rest.abs().max() <= 1.0 / 14 + 1e-6                     # scale now set by |x| <= 1
    assert torch.equal(Q.int4_keep_topc(x, 0), Q.int4_per_token(x))


def test_int4_keep_topc_batched_rows_independent():
    torch.manual_seed(1)
    x = torch.randn(2, 3, 64)
    x[0, 1, 5] = 50.0
    q = Q.int4_keep_topc(x, 2)
    for i in range(2):
        for j in range(3):
            assert torch.allclose(q[i, j], Q.int4_keep_topc(x[i, j][None], 2)[0])


def test_int_per_token_bits():
    torch.manual_seed(2)
    x = torch.randn(4, 64)
    assert torch.equal(Q.int_per_token(x, 4), Q.int4_per_token(x))
    q3 = Q.int_per_token(x, 3)
    lv = (q3 / (x.abs().amax(-1, keepdim=True) / 3)).round()
    assert lv.abs().max() <= 3 and len(torch.unique(lv)) <= 7
