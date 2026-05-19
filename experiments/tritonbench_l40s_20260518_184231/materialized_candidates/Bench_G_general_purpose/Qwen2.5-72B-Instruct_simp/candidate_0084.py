import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_aligned(
    Q, K, V, B0, Out,
    stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_kn,
    stride_vb, stride_vh, stride_vn, stride_b0b, stride_b0h, stride_b0m, stride_b0n,
    stride_ob, stride_oh, stride_om,
    N_CTX, H, BLOCK_M, BLOCK_N, BLOCK_DMODEL, SOFTMAX_SCALE
):
    pid = tl.program_id(axis=0)
    pid_m = tl.program_id(axis=1)
    num_pid_m = tl.num_programs(axis=1)
    num_pid_n = tl.num_programs(axis=0)
    block_m = pid_m * BLOCK_M
    block_n = pid * BLOCK_N

    # Offsets for Q, K, V, B0, and Out
    offs_qm = block_m + tl.arange(0, BLOCK_M)
    offs_kn = block_n + tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)

    q_ptrs = Q + (offs_qm[:, None] * stride_qm + offs_d[None, :] * stride_qh)
    k_ptrs = K + (offs_kn[None, :] * stride_kn + offs_d[:, None] * stride_kh)
    v_ptrs = V + (offs_kn[None, :] * stride_vn + offs_d[:, None] * stride_vh)
    b0_ptrs = B0 + (offs_qm[:, None] * stride_b0m + offs_kn[None, :] * stride_b0n)

    # Load Q, K, V, and B0
    q = tl.load(q_ptrs)
    k = tl.load(k_ptrs)
    v = tl.load(v_ptrs)
    b0 = tl.load(b0_ptrs)

    # Compute scaled dot-product
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    qk = tl.dot(q, k, trans_b=True)
    qk *= SOFTMAX_SCALE
    qk += b0

    # Compute softmax
    qk = tl.exp2(qk)
    qk /= tl.sum(qk, axis=1)[:, None]

    # Compute output
    out = tl.dot(qk, v)

    # Store output
    out_ptrs = Out + (offs_qm[:, None] * stride_om + offs_d[None, :] * stride_oh)
    tl.store(out_ptrs, out)

import torch
import triton
import triton.language as tl

def _attention_rel_h_rel_w_kernel_aligned_device(q, k, v, b0, out, BLOCK_M, BLOCK_N, BLOCK_DMODEL, SOFTMAX_SCALE):
    # Validate input shapes
    assert q.shape == k.shape == v.shape, "Q, K, and V must have the same shape"
    assert q.shape[0] == b0.shape[0], "Batch dimension of Q and B0 must match"
    assert q.shape[1] == b0.shape[1], "Head dimension of Q and B0 must match"
    assert q.shape[2] == b0.shape[2], "Sequence length of Q and B0 must match"
    assert q.shape[3] == b0.shape[3], "Sequence length of K and B0 must match"

    # Calculate grid dimensions
    grid = (q.shape[2] // BLOCK_M, q.shape[0] * q.shape[1])

    # Launch the kernel
    _fwd_kernel_aligned[grid](
        q, k, v, b0, out,
        q.stride(0), q.stride(1), q.stride(2), k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2), b0.stride(0), b0.stride(1), b0.stride(2), b0.stride(3),
        out.stride(0), out.stride(1), out.stride(2),
        q.shape[2], q.shape[1], BLOCK_M, BLOCK_N, BLOCK_DMODEL, SOFTMAX_SCALE
    )
