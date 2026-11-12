"""Activation fake-quantizers for E0.5 (torch, operate on the last dimension).

int4_per_token : symmetric INT4, one scale per row, scale = clip * max|x| / 7, range [-7, 7]
mxfp4          : OCP MX v1.0 — blocks of 32, shared E8M0 scale 2^(floor(log2 amax) - 2),
                 elements FP4 E2M1 {0, .5, 1, 1.5, 2, 3, 4, 6}, saturating at 6
nvfp4          : NVIDIA NVFP4 — blocks of 16, block scale amax/6 stored in FP8 E4M3,
                 times a per-tensor FP32 scale amax_tensor / (448 * 6)
Rounding to the E2M1 grid is round-to-nearest with ties to the lower magnitude (the spec
requires ties-to-even; the difference only affects exact midpoints).
"""
import torch

E2M1 = torch.tensor([0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0])


def _round_e2m1(y):
    """Round |y| (already scaled) to the E2M1 grid, keep sign, saturate at 6."""
    grid = E2M1.to(y.device, y.dtype)
    a = y.abs().clamp(max=6.0)
    idx = torch.bucketize(a, (grid[1:] + grid[:-1]) / 2, right=False)
    return torch.sign(y) * grid[idx]


def int4_per_token(x, clip=1.0):
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    scale = clip * amax / 7.0
    return (x / scale).round().clamp(-7, 7) * scale


def int_per_token(x, bits):
    """Symmetric per-token INT-b, range [-(2^(b-1)-1), 2^(b-1)-1]; bits=4 equals int4_per_token."""
    qmax = 2 ** (bits - 1) - 1
    amax = x.abs().amax(-1, keepdim=True).clamp(min=1e-12)
    scale = amax / qmax
    return (x / scale).round().clamp(-qmax, qmax) * scale


def int4_keep_topc(x, c):
    """Per-token INT4 where the c largest-|x| entries of each row stay exact and the row scale
    is set by the remaining entries (dynamic per-row outlier split)."""
    if c <= 0:
        return int4_per_token(x)
    a = x.abs()
    keep = torch.zeros_like(x, dtype=torch.bool).scatter_(-1, a.topk(c, dim=-1).indices, True)
    q = int4_per_token(x.masked_fill(keep, 0.0))
    return torch.where(keep, x, q)


def int4_zero_frac(x):
    """Share of entries per row that per-token INT4 rounds to zero (|x| < 0.5 * max|x| / 7)."""
    a = x.abs()
    return (a < a.amax(-1, keepdim=True) / 14.0).float().mean(-1)


def _blocks(x, b):
    d = x.shape[-1]
    assert d % b == 0, (d, b)
    return x.reshape(*x.shape[:-1], d // b, b)


def mxfp4(x, block=32):
    xb = _blocks(x, block)
    amax = xb.abs().amax(-1, keepdim=True)
    e = torch.floor(torch.log2(amax.clamp(min=2.0 ** -126))) - 2.0      # emax(E2M1) = 2
    scale = torch.pow(2.0, e)
    q = _round_e2m1(xb / scale) * scale
    q = torch.where(amax > 0, q, torch.zeros_like(q))
    return q.reshape(x.shape)


def nvfp4(x, block=16, tensor_amax=None):
    """tensor_amax: scalar, or [n] per-row values (rows taken from different tensors)."""
    xb = _blocks(x, block)
    if tensor_amax is None:
        t_amax = x.abs().amax()
    else:
        t_amax = torch.as_tensor(tensor_amax, device=x.device, dtype=x.dtype)
        if t_amax.ndim == 1:
            t_amax = t_amax.reshape(-1, *([1] * (xb.ndim - 1)))
    s_g = (t_amax / (448.0 * 6.0)).clamp(min=1e-30)
    amax = xb.abs().amax(-1, keepdim=True)
    s_b = (amax / 6.0 / s_g).clamp(max=448.0).to(torch.float8_e4m3fn).to(x.dtype)
    scale = s_b * s_g
    safe = torch.where(scale > 0, scale, torch.ones_like(scale))
    q = _round_e2m1(xb / safe) * scale
    return q.reshape(x.shape)


FORMATS = {"int4_tok": int4_per_token, "mxfp4": mxfp4, "nvfp4": nvfp4}


def random_orthogonal(d, seed=0, device="cpu"):
    """Seeded random orthogonal matrix (QR of a Gaussian; sign-fixed). Stands in for QuaRot's
    Hadamard rotation, which has no simple construction for d = 12288 (down_proj input)."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    a = torch.randn(d, d, generator=g, dtype=torch.float32).to(device)
    q, r = torch.linalg.qr(a.double() if device == "cpu" else a)
    q = q * torch.sign(torch.diagonal(r))
    return q.to(dtype=torch.float32)


def massive_mask(x, frac=0.25):
    """Per-row 'massive channels': |x_d| >= frac * max_d |x_d|."""
    a = x.abs()
    return a >= frac * a.amax(-1, keepdim=True)


def path_errors(x, w, fmt, rot=None, frac=0.25, tensor_amax=None):
    """Output-error decomposition for rows x [n, d] through y = x w^T (w [o, d]).

    With rotation R (orthogonal matrix, or a callable applying one along the last dim):
    quantize x R, multiply by (w R)^T, so the exact output is unchanged. The massive mask is taken in the space where quantization happens.
    Returns per-row tensors: total, p2 (error carried by massive channels), p1 (the rest),
    and ref = ||x w^T||^2 for normalization.
    """
    if rot is None:
        xr, wr = x, w
    elif callable(rot):
        xr, wr = rot(x), rot(w)
    else:
        xr, wr = x @ rot, w @ rot
    q = nvfp4(xr, tensor_amax=tensor_amax) if fmt == "nvfp4" else FORMATS[fmt](xr)
    delta = q - xr
    m = massive_mask(xr, frac)
    d_m = torch.where(m, delta, torch.zeros_like(delta))
    d_r = delta - d_m
    tot = (delta @ wr.T).pow(2).sum(-1)
    p2 = (d_m @ wr.T).pow(2).sum(-1)
    p1 = (d_r @ wr.T).pow(2).sum(-1)
    ref = (x @ w.T).pow(2).sum(-1)
    return {"total": tot, "p2": p2, "p1": p1, "ref": ref, "n_massive": m.sum(-1)}



# --- Hadamard rotations (QuaRot-style) ----------------------------------------------------

def _sylvester(n):
    h = torch.ones(1, 1, dtype=torch.float64)
    while h.shape[0] < n:
        h = torch.cat([torch.cat([h, h], 1), torch.cat([h, -h], 1)], 0)
    assert h.shape[0] == n, "Sylvester construction needs a power of two"
    return h


def _paley12():
    """Hadamard matrix of order 12 (Paley construction I, q = 11)."""
    q = 11
    residues = {(i * i) % q for i in range(1, q)}
    chi = lambda a: 0 if a % q == 0 else (1 if a % q in residues else -1)
    Qm = torch.tensor([[chi(j - i) for j in range(q)] for i in range(q)], dtype=torch.float64)
    S = torch.zeros(q + 1, q + 1, dtype=torch.float64)
    S[0, 1:] = 1
    S[1:, 0] = -1
    S[1:, 1:] = Qm
    return torch.eye(q + 1, dtype=torch.float64) + S


def hadamard_matrix(d):
    """Unnormalized +-1 Hadamard of order d = 2^k or 12 * 2^k."""
    if d & (d - 1) == 0:
        return _sylvester(d)
    assert d % 12 == 0 and (d // 12) & (d // 12 - 1) == 0, f"unsupported Hadamard order {d}"
    return torch.kron(_paley12(), _sylvester(d // 12))


def random_hadamard(d, seed=0, device="cpu"):
    """Randomized Hadamard H * diag(s) / sqrt(d) (orthogonal), as in QuaRot."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    s = torch.randint(0, 2, (d,), generator=g).double() * 2 - 1
    h = hadamard_matrix(d) * s[None, :] / d ** 0.5
    return h.to(device=device, dtype=torch.float32)


def block_hadamard_rotate(x, block, seed=0):
    """Apply a randomized block-diagonal Hadamard (size `block`) along the last dim."""
    d = x.shape[-1]
    assert d % block == 0
    hb = random_hadamard(block, seed=seed, device=x.device)
    return (x.reshape(*x.shape[:-1], d // block, block) @ hb).reshape(x.shape)


# --- quantization along the token axis with excluded positions (per-channel KV) -----------

def _pad_tokens(x, block):
    n = x.shape[-1]
    pad = (-n) % block
    if pad:
        x = torch.nn.functional.pad(x, (0, pad))
    return x, n


def quant_along_tokens(x, fmt, block, exclude=None):
    """Fake-quantize x [..., n_tokens] in contiguous token groups of size `block`, per channel.

    exclude: bool [n_tokens]; excluded tokens do not contribute to the group statistics
    (they are still quantized here; callers restore them to FP if they are kept in FP).
    fmt: 'int4_asym' (KIVI-style min-max, 16 levels), 'mxfp4', 'nvfp4'.
    """
    mask = torch.zeros(x.shape[-1], dtype=torch.bool, device=x.device) if exclude is None else exclude
    xp, n = _pad_tokens(x, block)
    mp, _ = _pad_tokens(mask.to(x.dtype), block)
    mp = mp.bool()
    pad_mask = torch.zeros_like(mp)
    pad_mask[n:] = True
    ignore = (mp | pad_mask).reshape(-1, block)                         # [G, block]
    xb = xp.reshape(*xp.shape[:-1], -1, block)                          # [..., G, block]
    if fmt == "int4_asym":
        hi = xb.masked_fill(ignore, float("-inf")).amax(-1, keepdim=True)
        lo = xb.masked_fill(ignore, float("inf")).amin(-1, keepdim=True)
        valid = torch.isfinite(hi)
        hi = torch.where(valid, hi, torch.zeros_like(hi))
        lo = torch.where(valid, lo, torch.zeros_like(lo))
        scale = ((hi - lo) / 15.0).clamp(min=1e-12)
        q = ((xb - lo) / scale).round().clamp(0, 15) * scale + lo
    elif fmt in ("mxfp4", "nvfp4"):
        amax = xb.masked_fill(ignore, 0.0).abs().amax(-1, keepdim=True)
        if fmt == "mxfp4":
            e = torch.floor(torch.log2(amax.clamp(min=2.0 ** -126))) - 2.0
            scale = torch.pow(2.0, e)
        else:
            t_amax = x.masked_fill(mask, 0.0).abs().amax()
            s_g = (t_amax / (448.0 * 6.0)).clamp(min=1e-30)
            s_b = (amax / 6.0 / s_g).clamp(max=448.0).to(torch.float8_e4m3fn).to(x.dtype)
            scale = s_b * s_g
        safe = torch.where(scale > 0, scale, torch.ones_like(scale))
        q = _round_e2m1(xb / safe) * scale
    else:
        raise ValueError(fmt)
    return q.reshape(xp.shape)[..., :n]
