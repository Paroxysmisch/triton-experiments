import triton
import triton.language as tl
import torch

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q_ptr, k_ptr, g_ptr, A_ptr,
    Q_STRIDE, K_STRIDE, G_STRIDE, A_STRIDE,
    N_CTX, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    i = pid // N_CTX
    j = pid % N_CTX

    if i > j:
        return

    q = tl.load(q_ptr + i * Q_STRIDE)
    k = tl.load(k_ptr + j * K_STRIDE)
    g = tl.load(g_ptr + i * G_STRIDE)

    # Compute the block
    qk = tl.dot(q, k.T)
    qk_scaled = qk * g
    qk_exp = tl.exp(qk_scaled)
    
    # Accumulate into A
    b_A = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    b_A += qk_exp
    tl.store(A_ptr + i * A_STRIDE + j * BLOCK_SIZE, b_A)


@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q_ptr, k_ptr, g_ptr, A_ptr,
    Q_STRIDE, K_STRIDE, G_STRIDE, A_STRIDE,
    N_CTX, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    i = pid // N_CTX
    j = pid % N_CTX

    if i != j:
        return

    q = tl.load(q_ptr + i * Q_STRIDE)
    k = tl.load(k_ptr + j * K_STRIDE)
    g = tl.load(g_ptr + i * G_STRIDE)

    # Compute the block
    qk = tl.dot(q, k.T)
    qk_scaled = qk * g
    qk_exp = tl.exp(qk_scaled)
    
    # Accumulate into A
    b_A = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    b_A += qk_exp
    tl.store(A_ptr + i * A_STRIDE + j * BLOCK_SIZE, b_A)


@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q_ptr, k_ptr, g_ptr, A_intra_ptr,
    Q_STRIDE, K_STRIDE, G_STRIDE, A_INTRA_STRIDE,
    N_CTX, BLOCK_SIZE: tl.constexpr, K_SPLIT: tl.constexpr
):
    pid = tl.program_id(0)
    i = pid // N_CTX
    j = pid % N_CTX

    q = tl.load(q_ptr + i * Q_STRIDE)
    k = tl.load(k_ptr + j * K_STRIDE)
    g = tl.load(g_ptr + i * G_STRIDE)

    # Compute the block with splitting
    b_A_intra = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k_split in range(K_SPLIT):
        qk = tl.dot(q[:, k_split * BLOCK_SIZE:(k_split + 1) * BLOCK_SIZE], 
                    k[:, k_split * BLOCK_SIZE:(k_split + 1) * BLOCK_SIZE].T)
        qk_scaled = qk * g
        qk_exp = tl.exp(qk_scaled)
        b_A_intra += qk_exp

    tl.store(A_intra_ptr + i * A_INTRA_STRIDE + j * BLOCK_SIZE, b_A_intra)


@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(
    A_intra_ptr, A_ptr,
    A_INTRA_STRIDE, A_STRIDE,
    N_CTX, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    i = pid // N_CTX
    j = pid % N_CTX

    b_A = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)
    for k_split in range(N_CTX):
        b_A += tl.load(A_intra_ptr + i * A_INTRA_STRIDE + j * BLOCK_SIZE)

    tl.store(A_ptr + i * A_STRIDE + j * BLOCK_SIZE, b_A)


@triton.jit
def chunk_gla_fwd_kernel_o(
    q_ptr, k_ptr, g_ptr, A_ptr, o_ptr,
    Q_STRIDE, K_STRIDE, G_STRIDE, A_STRIDE, O_STRIDE,
    N_CTX, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    i = pid // N_CTX
    j = pid % N_CTX

    q = tl.load(q_ptr + i * Q_STRIDE)
    k = tl.load(k_ptr + j * K_STRIDE)
    g = tl.load(g_ptr + i * G_STRIDE)
    A = tl.load(A_ptr + i * A_STRIDE + j * BLOCK_SIZE)

    # Compute the output
    o = tl.dot(A, k)
    o = o * g
    tl.store(o_ptr + i * O_STRIDE + j * BLOCK_SIZE, o)


def chunk_fwd_intra_gated_gk_fn(q, k, g, A, N_CTX, BLOCK_SIZE, K_SPLIT):
    grid = (N_CTX, N_CTX)
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](q, k, g, A, q.stride(0), k.stride(0), g.stride(0), A.stride(0), N_CTX, BLOCK_SIZE)
    chunk_gla_fwd_A_kernel_intra_sub_intra[grid](q, k, g, A, q.stride(0), k.stride(0), g.stride(0), A.stride(0), N_CTX, BLOCK_SIZE)
    chunk_gla_fwd_A_kernel_intra_sub_intra_split[grid](q, k, g, A, q.stride(0), k.stride(0), g.stride(0), A.stride(0), N_CTX, BLOCK_SIZE, K_SPLIT)
    chunk_gla_fwd_A_kernel_intra_sub_intra_merge[grid](A, A, A.stride(0), A.stride(0), N_CTX, BLOCK_SIZE)


def chunk_fwd_o_gated_gk_fn(q, k, g, A, o, N_CTX, BLOCK_SIZE):
    grid = (N_CTX, N_CTX)
    chunk_gla_fwd_kernel_o[grid](q, k, g, A, o, q.stride(0), k.stride(0), g.stride(0), A.stride(0), o.stride(0), N_CTX, BLOCK_SIZE)
