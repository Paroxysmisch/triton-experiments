import torch
import triton
import triton.language as tl

@triton.jit
def _triton_rope(
    Q,
    K,
    COS,
    SIN,
    ROW_STRIDE_Q,
    ROW_STRIDE_K,
    COS_PHI,
    SIN_PHI,
    CU_SEQLENS,
    BLOCK_M: tl.constexpr,
    NHEADS: tl.constexpr,
    D_HEAD: tl.constexpr,
    SEQLEN_OFFSETS: tl.constexpr,
    IS_SEQLEN_OFFSETS_TENSOR: tl.constexpr,
    IS_CU_SEQLENS_TENSOR: tl.constexpr,
    IS_VARLEN: tl.constexpr,
    BACKWARD_PASS: tl.constexpr,
):
    start_idx = tl.program_id(0)
    if BACKWARD_PASS:
        # forward pass is opposite to backward pass
        start_idx = tl.num_programs(0) - 1 - start_idx
    pid_batch = tl.math.floordiv(start_idx, NHEADS)
    pid_head = start_idx % NHEADS
    if not IS_VARLEN:
        q_offset = pid_batch * ROW_STRIDE_Q + pid_head * D_HEAD
        k_offset = pid_batch * ROW_STRIDE_K + pid_head * D_HEAD
    else:
        start_idx_batch = tl.math.minimum(pid_batch + 1, tl.num_programs(0))
        start_idx_batch = tl.math.maximum(0, start_idx_batch - 1)
        row_offset = tl.load(
            CU_SEQLENS + start_idx_batch, mask=(pid_batch <
                                                CU_SEQLENS.shape[0] - 1), other=0)
        q_rows = tl.load(
            CU_SEQLENS + pid_batch + 1, mask=pid_batch <
            CU_SEQLENS.shape[0] - 1, other=0
        ) - row_offset
        seqlen_offset = tl.load(
            CU_SEQLENS + pid_batch, mask=pid_batch < CU_SEQLENS.shape[0] - 1, other=0)
        q_offset = (row_offset + seqlen_offset * NHEADS + pid_head * D_HEAD) * ROW_STRIDE_Q
        k_offset = (seqlen_offset * NHEADS + pid_head * D_HEAD) * ROW_STRIDE_K
    # tl.print(
    #     f"offset {q_offset = } {pid_batch = } {pid_head = } {NHEADS = }",
    #     flush=True,
    # )
    # tl.print(tl.libdevice.exp2(20), flush=True)
    BLOCK_M_TRUNC = tl.where(pid_batch * NHEADS + pid_head ==
                             tl.num_programs(0) - 1, tl.num_programs(1) - (start_idx + 1) * BLOCK_M, BLOCK_M)

    rm = tl.arange(0, BLOCK_M_TRUNC) + start_idx * BLOCK_M
    rn = tl.arange(0, BLOCK_M_TRUNC)
    q_ptrs = Q + q_offset + ROW_STRIDE_Q * rm[:, None] + rn[None, :]
    k_ptrs = K + k_offset + ROW_STRIDE_K * rm[:, None] + rn[None, :]
    ro = rm % BLOCK_M

    if not IS_SEQLEN_OFFSETS_TENSOR:
        ro_cs = ro + SEQLEN_OFFSETS
    else:
        ro_cs = ro + tl.load(SEQLEN_OFFSETS + pid_batch +
                            0 * tl.arange(0, BLOCK_M_TRUNC)).to(tl.int32)

    d_head_half = D_HEAD // 2
    cos = tl.load(
        COS + ro_cs * d_head_half + tl.arange(
            0, BLOCK_M_TRUNC) % d_head_half,
        mask=(rm < q_rows) & (rn < d_head_half),
        other=1.0,
    ).to(tl.float32)

    sin = tl.load(
        SIN + ro_cs * d_head_half + tl.arange(
            0, BLOCK_M_TRUNC) % d_head_half,
        mask=(rm < q_rows) & (rn < d_head_half),
        other=0.0,
    ).to(tl.float32)
    if not BACKWARD_PASS:
        cos_k, sin_k = cos, sin
    else:
        cos_k = -cos
        sin_k = -sin
    q = tl.load(q_ptrs, mask=(rm < q_rows) & (rn < d_head_half), other=0.0).to(
        tl.float32
    )
    k = tl.load(k_ptrs, mask=(rm < q_rows) & (rn < d_head_half), other=0.0).to(
        tl.float32
    )
    q_rot = q * cos - k * sin_k
    k_rot = q * sin_k + k * cos_k

    tl.store(q_ptrs, q_rot, mask=(rm < q_rows) & (rn < d_head_half))
    tl.store(k_ptrs, k_rot, mask=(rm < q_rows) & (rn < d_head_half))
    return

def rope_forward(q,
                k,
                cos,
                sin,
                cu_seqlens,
                seqlen_offsets,
                max_seqlen,
                cos_phi,
                sin_phi,
                batch_size,
                seq_len,
                nheads,
                d_head):
    BACKWARD_PASS = False
    IS_VARLEN = cu_seqlens is not None
    # traspose to (batch_size, seq_len, nheads, d_head)
    q = q.view(batch_size, seq_len, nheads, d_head)

    seqlen_offsets = seqlen_offsets.flatten().to(torch.int32)
    cu_seqlens = cu_seqlens.to(torch.int32) if cu_seqlens is not None else None
    k = k.view(batch_size, seq_len, nheads, d_head).contiguous()

    # Compute the required padding
    q = tl.compiler.pad(q, (0, 0, 0, (cu_seqlens[-1] + 1) * nheads - q.shape[1]))
    k = tl.compiler.pad(k, (0, 0, 0, cu_seqlens[-1] * nheads - k.shape[0]))

    row_num = (batch_size * seq_len).to(torch.int32)

    k = tl.trans(k)
    _triton_rope[(
        row_num,
        nheads,
    )](q, k, cos, sin, *tl.xform_strides(q), cos_phi, sin_phi, cu_seqlens,
        NHEADS=nheads,
        D_HEAD=d_head,
        SEQLEN_OFFSETS=seqlen_offsets,
        IS_SEQLEN_OFFSETS_TENSOR=(
            seqlen_offsets.ndim == 2),
        IS_CU_SEQLENS_TENSOR=(cu_seqlens is not None),
        IS_VARLE
