import triton
import triton.language as tl

@triton.jit
def block_sparse_attention_kernel(
    Q, K, V, out, row_ptr, col_idx,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_om, stride_od,
    nheads, nblocks_q, nblocks_k, d, softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, NUM_D_BLOCKS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    head_id = tl.program_id(axis=2)

    # Compute the block indices
    block_m = pid * BLOCK_M
    block_n = bid * BLOCK_N

    # Pointers to the Q, K, V matrices
    Q_block_ptr = Q + head_id * stride_qb + block_m * stride_qm
    K_block_ptr = K + head_id * stride_kb + block_n * stride_kn
    V_block_ptr = V + head_id * stride_vb + block_n * stride_vn
    out_block_ptr = out + head_id * stride_ob + block_m * stride_om

    # Load the Q, K, V blocks
    Q_block = tl.load(Q_block_ptr, mask=block_m + tl.arange(0, BLOCK_M) < nblocks_q, other=0.0)
    K_block = tl.load(K_block_ptr, mask=block_n + tl.arange(0, BLOCK_N) < nblocks_k, other=0.0)
    V_block = tl.load(V_block_ptr, mask=block_n + tl.arange(0, BLOCK_N) < nblocks_k, other=0.0)

    # Compute the query-key products
    QK = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, d, BLOCK_D):
        QK += tl.dot(Q_block[:, i:i+BLOCK_D], K_block[:, i:i+BLOCK_D].T)
    QK *= softmax_scale

    # Apply the softmax
    QK = tl.softmax(QK, axis=1)

    # Compute the weighted value sums
    out_block = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for i in range(0, BLOCK_N, NUM_D_BLOCKS):
        out_block += tl.dot(QK[:, i:i+NUM_D_BLOCKS], V_block[i:i+NUM_D_BLOCKS, :])

    # Store the output
    tl.store(out_block_ptr, out_block, mask=block_m + tl.arange(0, BLOCK_M) < nblocks_q)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 16, 'BLOCK_N': 16, 'BLOCK_D': 16, 'NUM_D_BLOCKS': 1}, num_warps=4),
        triton.Config({'BLOCK_M': 32, 'BLOCK_N': 32, 'BLOCK_D': 32, 'NUM_D_BLOCKS': 1}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_D': 64, 'NUM_D_BLOCKS': 1}, num_warps=16),
    ],
    key=['nheads', 'nblocks_q', 'nblocks_k', 'd']
)
@triton.jit
def block_sparse_attention_kernel(
    Q, K, V, out, row_ptr, col_idx,
    stride_qb, stride_qm, stride_qd,
    stride_kb, stride_kn, stride_kd,
    stride_vb, stride_vn, stride_vd,
    stride_ob, stride_om, stride_od,
    nheads, nblocks_q, nblocks_k, d, softmax_scale,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_D: tl.constexpr, NUM_D_BLOCKS: tl.constexpr
):
    pid = tl.program_id(axis=0)
    bid = tl.program_id(axis=1)
    head_id = tl.program_id(axis=2)

    # Compute the block indices
    block_m = pid * BLOCK_M
    block_n = bid * BLOCK_N

    # Pointers to the Q, K, V matrices
    Q_block_ptr = Q + head_id * stride_qb + block_m * stride_qm
    K_block_ptr = K + head_id * stride_kb + block_n * stride_kn
    V_block_ptr = V + head_id * stride_vb + block_n * stride_vn
    out_block_ptr = out + head_id * stride_ob + block_m * stride_om

    # Load the Q, K, V blocks
    Q_block = tl.load(Q_block_ptr, mask=block_m + tl.arange(0, BLOCK_M) < nblocks_q, other=0.0)
    K_block = tl.load(K_block_ptr, mask=block_n + tl.arange(0, BLOCK_N) < nblocks_k, other=0.0)
    V_block = tl.load(V_block_ptr, mask=block_n + tl.arange(0, BLOCK_N) < nblocks_k, other=0.0)

    # Compute the query-key products
    QK = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for i in range(0, d, BLOCK_D):
        QK += tl.dot(Q_block[:, i:i+BLOCK_D], K_block[:, i:i+BLOCK_D].T)
    QK *= softmax_scale

    # Apply the softmax
    QK = tl.softmax(QK, axis=1)

    # Compute the weighted value sums
    out_block = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)
    for i in range(0, BLOCK_N, NUM_D_BLOCKS):
        out_block += tl.dot(QK[:, i:i+NUM_D_BLOCKS], V_block[i:i+NUM_D_BLOCKS, :])

    # Store the output
    tl.store(out_block_ptr, out_block, mask=block_m + tl.arange(0, BLOCK_M) < nblocks_q)

def block_sparse_attention(Q, K, V, row_ptr, col_idx, nheads, nblocks_q, nblocks_k, d, softmax_scale):
    # Define the grid and block sizes
    grid = (nblocks_q, nblocks_k, nheads)
    block = (16, 16, 1)

    # Allocate the output tensor
    out = torch.empty_like(Q)

    # Launch the kernel
    block_sparse_attention_kernel[grid, block](
        Q, K, V, out, row_ptr, col_idx,
        Q.stride(0), Q.stride(1), Q.stride(2),
        K.stride(0), K.stride(1), K.stride(2),
        V.stride(0), V.stride(1), V.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        nheads, nblocks_q, nblocks_k, d, softmax_scale
    )

    return out
