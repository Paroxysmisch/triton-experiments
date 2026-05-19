import triton
import triton.language as tl
import math
import torch


# Define tuning parameters
BLOCK_M = 128
BLOCK_N = 128
BLOCK_DMODEL = 64


@triton.jit
def _attn_fwd_inner(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    M, N, D,
    qk_scale,
    # Optional parameters for features like a sliding window mask can go here.
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # -----
    # Program IDs
    bid_m = tl.program_id(0)
    bid_b = tl.program_id(1)

    # Compute block starting indices
    m_start = bid_m * BLOCK_M
    # Optionally handle a sliding window or other context specifics in N
    n_start = 0

    # Create ranges for block loading
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_n = n_start + tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Prepare accumulators for partial sums
    m_i = tl.full([BLOCK_M], float('-inf'), dtype=tl.float32)
    l_i = tl.zeros([BLOCK_M], dtype=tl.float32)

    # Tile loop over N
    # Each iteration processes a BLOCK_N portion of the K, V data
    # to compute interactions with Q.
    for n_offset in range(0, N, BLOCK_N):
        # Offsets for the current sub-block
        cur_offs_n = n_offset + tl.arange(0, BLOCK_N)

        # Load Q sub-block [BLOCK_M x BLOCK_DMODEL]
        # shape: (BLOCK_M, BLOCK_DMODEL)
        q = tl.load(
            Q_ptr + (bid_b * M + offs_m[:, None]) * stride_qm
            + (offs_d[None, :]) * stride_qd,
            mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
            other=0.0
        )

        # Load K sub-block [BLOCK_N x BLOCK_DMODEL]
        # shape: (BLOCK_N, BLOCK_DMODEL)
        k = tl.load(
            K_ptr + (bid_b * N + cur_offs_n[:, None]) * stride_km
            + (offs_d[None, :]) * stride_kd,
            mask=(cur_offs_n[:, None] < N) & (offs_d[None, :] < D),
            other=0.0
        )

        # QK^T for the current block: [BLOCK_M x BLOCK_N]
        # Each q_i and k_j are vectors of size BLOCK_DMODEL
        qk = tl.dot(q, k, trans_b=True) * qk_scale

        # Optional: apply sliding window or other mask if needed

        # Perform the iterative softmax update
        # Compare old maxima with the new block and combine
        mha_max = tl.maximum(tl.max(qk, 1), m_i)
        alpha = tl.exp2(qk - mha_max[:, None])
        alpha_old = tl.exp2(m_i - mha_max)
        alpha_old = alpha_old[:, None] * l_i[:, None]
        l_new = tl.sum(alpha, 1) + alpha_old[:, 0]
        m_i = mha_max
        l_i = l_new

        # Load V sub-block: [BLOCK_N x BLOCK_DMODEL]
        v = tl.load(
            V_ptr + (bid_b * N + cur_offs_n[:, None]) * stride_vm
            + (offs_d[None, :]) * stride_vd,
            mask=(cur_offs_n[:, None] < N) & (offs_d[None, :] < D),
            other=0.0
        )

        # Weighted sum for partial output
        w = alpha / l_new[:, None]
        out_update = tl.dot(w, v)
        out_val = tl.load(
            Out_ptr + (bid_b * M + offs_m[:, None]) * stride_om
            + (offs_d[None, :]) * stride_od,
            mask=(offs_m[:, None] < M) & (offs_d[None, :] < D),
            other=0.0
        )
        out_val += out_update
        tl.store(
            Out_ptr + (bid_b * M + offs_m[:, None]) * stride_om
            + (offs_d[None, :]) * stride_od,
            out_val,
            mask=(offs_m[:, None] < M) & (offs_d[None, :] < D)
        )


@triton.jit
def _attn_fwd(
    Q_ptr, K_ptr, V_ptr, Out_ptr,
    B, M, N, D,
    stride_qm, stride_qd,
    stride_km, stride_kd,
    stride_vm, stride_vd,
    stride_om, stride_od,
    qk_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    # Program ID for block along M dimension and batch dimension
    bid_m = tl.program_id(0)
    bid_b = tl.program_id(1)

    # Compute start index of each block
    m_start = bid_m * BLOCK_M

    # Offsets for M dimension
    offs_m = m_start + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    # Zero-initialize the output block
    out_init = tl.zeros([BLOCK_M, BLOCK_DMODEL], dtype=tl.float32)
    tl.store(
        Out_ptr + (bid_b * M + offs_m[:, None]) * stride_om
        + (offs_d[None, :]) * stride_od,
        out_init,
        mask=(offs_m[:, None] < M) & (offs_d[None, :] < D)
    )

    # Now call the inner kernel that processes N in sub-blocks (BLOCK_N)
    # This is done in a separate kernel to illustrate modular design, but
    # you can also inline or parametrize it in a single kernel if desired.
    _attn_fwd_inner(
        Q_ptr, K_ptr, V_ptr, Out_ptr,
        stride_qm, stride_qd,
        stride_km, stride_kd,
        stride_vm, stride_vd,
        stride_om, stride_od,
        M, N, D,
        qk_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL
    )


def _forward(Q, K, V, qk_scale, blocksizes=(BLOCK_M, BLOCK_N, BLOCK_DMODEL)):
    # Q, K, V: [B, M, D], [B, N, D], [B, N, D]
    # qk_scale: scalar float
    # blocksizes: (BLOCK_M, BLOCK_N, BLOCK_DMODEL)
    B, M, DQ = Q.shape
    _, N, DK = K.shape
    Out = torch.zeros((B, M, DQ), device=Q.device, dtype=Q.dtype)

    stride_qm = Q.stride(0)
    stride_qd = Q.stride(2)
    stride_km = K.stride(0)
    stride_kd = K.stride(2)
    stride_vm = V.stride(0)
    stride_vd = V.stride(2)
    stride_om = Out.stride(0)
    stride_od = Out.stride(2)

    grid = lambda META: (math.ceil(M / META['BLOCK_M']), B)
    try:
        _attn_fwd[grid](
            Q, K, V, Out,
            B, M, N, DQ,
            stride_qm, stride_qd,
            stride_km, stride_kd,
            stride_vm, stride_vd,
            stride_om, stride_od,
            qk_scale,
            BLOCK_M=blocksizes[0],
            BLOCK_N=blocksizes[1],
            BLOCK_DMODEL=blocksizes[2]
        )
    except triton.OutOfResources as e:
        raise RuntimeError("Triton kernel out of resources, try smaller block sizes") from e

    return Out
