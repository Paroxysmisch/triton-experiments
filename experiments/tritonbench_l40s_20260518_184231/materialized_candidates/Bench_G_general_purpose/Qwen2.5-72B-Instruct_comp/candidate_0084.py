import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_qb, stride_qh, stride_qm,
    stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn,
    stride_ob, stride_oh, stride_om,
    stride_b0b, stride_b0h, stride_b0n,
    nheads, nctx, sm_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_DMODEL: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = pid // nheads
    hid = pid % nheads

    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (bid * stride_qb + hid * stride_qh) + offs_m[:, None] * stride_qm + offs_d[None, :]
    k_ptrs = K + (bid * stride_kb + hid * stride_kh) + offs_d[:, None] * stride_kn + offs_n[None, :]
    v_ptrs = V + (bid * stride_vb + hid * stride_vh) + offs_n[:, None] * stride_vn + offs_d[None, :]
    b0_ptrs = B0 + (bid * stride_b0b + hid * stride_b0h) + offs_m[:, None] * stride_b0n + offs_n[None, :]

    acc = tl.zeros((BLOCK_M, BLOCK_DMODEL), dtype=tl.float32)

    for start_n in range(0, nctx, BLOCK_N):
        k_block_ptr = k_ptrs + start_n * stride_kn
        v_block_ptr = v_ptrs + start_n * stride_vn
        b0_block_ptr = b0_ptrs + start_n * stride_b0n

        k = tl.load(k_block_ptr)
        v = tl.load(v_block_ptr)
        b0 = tl.load(b0_block_ptr)

        q = tl.load(q_ptrs)
        q = q * sm_scale

        qk = tl.dot(q, k, trans_b=True)
        qk = qk + b0

        m = tl.max(qk, 1)
        qk = qk - m[:, None]
        qk = tl.exp2(qk)

        z = tl.sum(qk, 1)
        qk = qk / z[:, None]

        acc += tl.dot(qk, v)

    out_ptrs = Out + (bid * stride_ob + hid * stride_oh) + offs_m[:, None] * stride_om + offs_d[None, :]
    tl.store(out_ptrs, acc)

import torch
import triton
import triton.language as tl

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, sm_scale, BLOCK_M, BLOCK_N, BLOCK_DMODEL, num_warps, num_stages):
    # Validate input tensor shapes and types
    assert q.shape == k.shape == v.shape, "Q, K, and V must have the same shape"
    assert q.dtype == k.dtype == v.dtype, "Q, K, and V must have the same data type"
    assert q.dim() == 4, "Q, K, and V must be 4D tensors"
    assert b0.shape[0] == q.shape[0], "B0 batch size must match Q batch size"
    assert b0.shape[1] == q.shape[1], "B0 head size must match Q head size"
    assert b0.shape[2] == q.shape[2], "B0 sequence length must match Q sequence length"
    assert b0.shape[3] == q.shape[2], "B0 sequence length must match Q sequence length"

    B, H, N_CTX, D = q.shape
    OUT_DTYPE = q.dtype
    BIAS_LAST_SIZE = b0.shape[-1]

    # Configure the grid
    grid = (B * H,)

    # Allocate output tensor
    out = torch.empty_like(q, dtype=OUT_DTYPE, device=q.device)

    # Launch the kernel
    _fwd_kernel_aligned[grid](
        q, k, v, b0, out,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        b0.stride(0), b0.stride(1), b0.stride(2),
        H, N_CTX, sm_scale,
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_DMODEL=BLOCK_DMODEL,
        num_warps=num_warps, num_stages=num_stages
    )

    return out
