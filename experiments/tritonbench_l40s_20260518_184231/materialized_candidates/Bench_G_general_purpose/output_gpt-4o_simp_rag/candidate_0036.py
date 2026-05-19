import triton
import triton.language as tl
import torch

@triton.jit
def _triton_rope(
    Q, K, COS, SIN,  # Pointers to input and output matrices
    CU_SEQLENS, SEQLEN_OFFSETS,
    SEQLEN, NHEADS, ROTARY_DIM, SEQLEN_RO, CACHE_KEY_SEQLEN,
    STRIDE_Q_BATCH, STRIDE_Q_SEQLEN, STRIDE_Q_NHEADS, STRIDE_Q_HEADDIM,
    STRIDE_K_BATCH, STRIDE_K_SEQLEN, STRIDE_K_NHEADS, STRIDE_K_HEADDIM,
    BLOCK_K: tl.constexpr, IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr, INTERLEAVED: tl.constexpr, CONJUGATE: tl.constexpr,
    BLOCK_M: tl.constexpr
):
    pid_m = tl.program_id(axis=0)
    pid_batch = tl.program_id(axis=1)
    pid_head = tl.program_id(axis=2)
    rotary_dim_half = ROTARY_DIM // 2

    if not IS_VARLEN:
        Q = Q + pid_batch * STRIDE_Q_BATCH + pid_head * STRIDE_Q_NHEADS
        K = K + pid_batch * STRIDE_K_BATCH + pid_head * STRIDE_K_NHEADS
    else:
        start_idx = tl.load(CU_SEQLENS + pid_batch)
        SEQLEN = tl.load(CU_SEQLENS + pid_batch + 1) - start_idx
        Q = Q + start_idx * STRIDE_Q_SEQLEN + pid_head * STRIDE_Q_NHEADS
        K = K + start_idx * STRIDE_K_SEQLEN + pid_head * STRIDE_K_NHEADS

    if pid_m * BLOCK_M >= SEQLEN:
        return

    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    if not IS_SEQLEN_OFFSETS_TENSOR:
        rm_cs = rm + SEQLEN_OFFSETS
    else:
        rm_cs = rm + tl.load(SEQLEN_OFFSETS + pid_batch)
    rk_half = tl.arange(0, BLOCK_K // 2)

    Q = Q + (rm[:, None] * STRIDE_Q_SEQLEN + rk_half[None, :] * STRIDE_Q_HEADDIM)
    K = K + (rm[:, None] * STRIDE_K_SEQLEN + rk_half[None, :] * STRIDE_K_HEADDIM)
    COS = COS + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])
    SIN = SIN + (rm_cs[:, None] * rotary_dim_half + rk_half[None, :])

    cos = tl.load(COS, mask=(rm_cs[:, None] < SEQLEN_RO) & (rk_half[None, :] < rotary_dim_half), other=1.0).to(tl.float32)
    sin = tl.load(SIN, mask=(rm_cs[:, None] < SEQLEN_RO) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)

    q0 = tl.load(Q, mask=(rm[:, None] < SEQLEN) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
    q1 = tl.load(Q + rotary_dim_half * STRIDE_Q_HEADDIM, mask=(rm[:, None] < SEQLEN) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
    k0 = tl.load(K, mask=(rm[:, None] < SEQLEN) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)
    k1 = tl.load(K + rotary_dim_half * STRIDE_K_HEADDIM, mask=(rm[:, None] < SEQLEN) & (rk_half[None, :] < rotary_dim_half), other=0.0).to(tl.float32)

    if CONJUGATE:
        sin = -sin

    q_out0 = q0 * cos - q1 * sin
    q_out1 = q0 * sin + q1 * cos
    k_out0 = k0 * cos - k1 * sin
    k_out1 = k0 * sin + k1 * cos

    Q = Q + (rm[:, None] * STRIDE_Q_SEQLEN + rk_half[None, :] * STRIDE_Q_HEADDIM)
    K = K + (rm[:, None] * STRIDE_K_SEQLEN + rk_half[None, :] * STRIDE_K_HEADDIM)

    tl.store(Q, q_out0, mask=(rm[:, None] < SEQLEN) & (rk_half[None, :] < rotary_dim_half))
    tl.store(Q + rotary_dim_half * STRIDE_Q_HEADDIM, q_out1, mask=(rm[:, None] < SEQLEN) & (rk_half[None, :] < rotary_dim_half))
    tl.store(K, k_out0, mask=(rm[:, None] < SEQLEN) & (rk_half[None, :] < rotary_dim_half))
    tl.store(K + rotary_dim_half * STRIDE_K_HEADDIM, k_out1, mask=(rm[:, None] < SEQLEN) & (rk_half[None, :] < rotary_dim_half))


def rope_forward(
    q: torch.Tensor,
    k: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    seqlen_offsets: Union[int, torch.Tensor] = 0,
    cu_seqlens: Optional[torch.Tensor] = None,
    max_seqlen: Optional[int] = None,
    interleaved=False,
    inplace=False,
    conjugate=False,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    is_varlen = cu_seqlens is not None
    if not is_varlen:
        batch, seqlen, nheads, headdim = q.shape
    else:
        assert max_seqlen is not None, "If cu_seqlens is passed in, then max_seqlen must be passed"
        total_seqlen, nheads, headdim = q.shape
        batch_p_1 = cu_seqlens.shape[0]
        batch = batch_p_1 - 1
        seqlen = max_seqlen

    seqlen_ro, rotary_dim = cos.shape
    assert sin.shape == cos.shape
    rotary_dim *= 2
    assert rotary_dim <= headdim, "rotary_dim must be <= headdim"
    assert headdim <= 256, "Only support headdim <= 256"
    assert seqlen_ro >= seqlen, "seqlen_ro must be >= seqlen"

    assert cos.dtype == sin.dtype, f"cos and sin must have the same dtype, got {cos.dtype} and {sin.dtype}"
    assert q.dtype == cos.dtype, f"Input and cos/sin must have the same dtype, got {q.dtype} and {cos.dtype}"

    cos, sin = cos.contiguous(), sin.contiguous()
    if isinstance(seqlen_offsets, torch.Tensor):
        assert seqlen_offsets.shape == (batch,)
        assert seqlen_offsets.dtype in [torch.int32, torch.int64]
        seqlen_offsets = seqlen_offsets.contiguous()
    else:
        assert seqlen_offsets + seqlen <= seqlen_ro

    q_out = torch.empty_like(q) if not inplace else q
    k_out = torch.empty_like(k) if not inplace else k
    if rotary_dim < headdim and not inplace:
        q_out[..., rotary_dim:].copy_(q[..., rotary_dim:])
        k_out[..., rotary_dim:].copy_(k[..., rotary_dim:])

    BLOCK_K = 32 if rotary_dim <= 32 else (64 if rotary_dim <= 64 else (128 if rotary_dim <= 128 else 256))
    def grid(META): return (triton.cdiv(seqlen, META["BLOCK_M"]), batch, nheads)
    BLOCK_M = 4 if interleaved else (8 if rotary_dim <= 64 else 4)

    with torch.cuda.device(q.device.index):
        _triton_rope[grid](
            q_out, k_out, cos, sin, cu_seqlens, seqlen_offsets, seqlen, nheads, rotary_dim, seqlen_ro, seqlen // 128,
            q_out.stride(0) if not is_varlen else 0, q_out.stride(-3), q_out.stride(-2), q_out.stride(-1),
            k_out.stride(0) if not is_varlen else 0, k_out.stride(-3), k_out.stride(-2), k_out.stride(-1),
            BLOCK_K, isinstance(seqlen_offsets, torch.Tensor), is_varlen, interleaved, conjugate, BLOCK_M,
        )

    return q_out, k_out, cos, sin
