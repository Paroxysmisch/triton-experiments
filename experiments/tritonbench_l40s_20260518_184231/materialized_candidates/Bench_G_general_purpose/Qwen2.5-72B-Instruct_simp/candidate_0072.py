import triton
import triton.language as tl

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_inter(
    q_ptr, k_ptr, g_ptr, A_ptr,
    q_stride0, q_stride1, k_stride0, k_stride1, g_stride0, g_stride1, A_stride0, A_stride1,
    BLOCK_SIZE: tl.constexpr, SCALE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    q_offsets = q_stride0 * offsets
    k_offsets = k_stride0 * offsets
    g_offsets = g_stride0 * offsets
    A_offsets = A_stride0 * offsets

    q = tl.load(q_ptr + q_offsets)
    k = tl.load(k_ptr + k_offsets)
    g = tl.load(g_ptr + g_offsets)

    dot_product = tl.dot(q, k, allow_tf32=True)
    scaled_dot_product = dot_product * SCALE
    adjusted_dot_product = scaled_dot_product + g

    tl.store(A_ptr + A_offsets, adjusted_dot_product)

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra(
    q_ptr, k_ptr, g_ptr, A_ptr,
    q_stride0, q_stride1, k_stride0, k_stride1, g_stride0, g_stride1, A_stride0, A_stride1,
    BLOCK_SIZE: tl.constexpr, SCALE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    q_offsets = q_stride0 * offsets
    k_offsets = k_stride0 * offsets
    g_offsets = g_stride0 * offsets
    A_offsets = A_stride0 * offsets

    q = tl.load(q_ptr + q_offsets)
    k = tl.load(k_ptr + k_offsets)
    g = tl.load(g_ptr + g_offsets)

    dot_product = tl.dot(q, k, allow_tf32=True)
    scaled_dot_product = dot_product * SCALE
    adjusted_dot_product = scaled_dot_product + g

    tl.store(A_ptr + A_offsets, adjusted_dot_product)

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_split(
    q_ptr, k_ptr, g_ptr, A_ptr, A_intra_ptr,
    q_stride0, q_stride1, k_stride0, k_stride1, g_stride0, g_stride1, A_stride0, A_stride1, A_intra_stride0, A_intra_stride1,
    BLOCK_SIZE: tl.constexpr, SCALE: tl.constexpr, K_CHUNK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    q_offsets = q_stride0 * offsets
    k_offsets = k_stride0 * offsets
    g_offsets = g_stride0 * offsets
    A_offsets = A_stride0 * offsets
    A_intra_offsets = A_intra_stride0 * offsets

    q = tl.load(q_ptr + q_offsets)
    k = tl.load(k_ptr + k_offsets)
    g = tl.load(g_ptr + g_offsets)

    for k_chunk_start in range(0, K_CHUNK_SIZE, BLOCK_SIZE):
        k_chunk_offsets = k_chunk_start + tl.arange(0, BLOCK_SIZE)
        k_chunk = tl.load(k_ptr + k_chunk_offsets)

        dot_product = tl.dot(q, k_chunk, allow_tf32=True)
        scaled_dot_product = dot_product * SCALE
        adjusted_dot_product = scaled_dot_product + g

        tl.store(A_intra_ptr + A_intra_offsets + k_chunk_offsets, adjusted_dot_product)

@triton.jit
def chunk_gla_fwd_A_kernel_intra_sub_intra_merge(
    A_ptr, A_intra_ptr,
    A_stride0, A_stride1, A_intra_stride0, A_intra_stride1,
    BLOCK_SIZE: tl.constexpr, K_CHUNK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    A_offsets = A_stride0 * offsets
    A_intra_offsets = A_intra_stride0 * offsets

    for k_chunk_start in range(0, K_CHUNK_SIZE, BLOCK_SIZE):
        k_chunk_offsets = k_chunk_start + tl.arange(0, BLOCK_SIZE)
        A_intra_chunk = tl.load(A_intra_ptr + A_intra_offsets + k_chunk_offsets)

        tl.atomic_add(A_ptr + A_offsets + k_chunk_offsets, A_intra_chunk)

@triton.jit
def chunk_gla_fwd_kernel_o(
    q_ptr, v_ptr, g_cumsum_ptr, A_ptr, h_ptr, o_ptr,
    q_stride0, q_stride1, v_stride0, v_stride1, g_cumsum_stride0, g_cumsum_stride1, A_stride0, A_stride1, h_stride0, h_stride1, o_stride0, o_stride1,
    BLOCK_SIZE: tl.constexpr, SCALE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    q_offsets = q_stride0 * offsets
    v_offsets = v_stride0 * offsets
    g_cumsum_offsets = g_cumsum_stride0 * offsets
    A_offsets = A_stride0 * offsets
    h_offsets = h_stride0 * offsets
    o_offsets = o_stride0 * offsets

    q = tl.load(q_ptr + q_offsets)
    v = tl.load(v_ptr + v_offsets)
    g_cumsum = tl.load(g_cumsum_ptr + g_cumsum_offsets)
    A = tl.load(A_ptr + A_offsets)
    h = tl.load(h_ptr + h_offsets)

    dot_product = tl.dot(q, v, allow_tf32=True)
    scaled_dot_product = dot_product * SCALE
    adjusted_dot_product = scaled_dot_product + g_cumsum
    final_result = adjusted_dot_product * A * h

    tl.store(o_ptr + o_offsets, final_result)

def chunk_fwd_intra_gated_gk_fn(q, k, g, A, BLOCK_SIZE, SCALE):
    grid = (A.shape[0] // BLOCK_SIZE, 1)
    chunk_gla_fwd_A_kernel_intra_sub_inter[grid](
        q, k, g, A,
        q.stride(0), q.stride(1), k.stride(0), k.stride(1), g.stride(0), g.stride(1), A.stride(0), A.stride(1),
        BLOCK_SIZE, SCALE
    )

def chunk_fwd_o_gated_gk_fn(q, v, g_cumsum, A, h, o, BLOCK_SIZE, SCALE):
    grid = (o.shape[0] // BLOCK_SIZE, 1)
    chunk_gla_fwd_kernel_o[grid](
        q, v, g_cumsum, A, h, o,
        q.stride(0), q.stride(1), v.stride(0), v.stride(1), g_cumsum.stride(0), g_cumsum.stride(1), A.stride(0), A.stride(1), h.stride(0), h.stride(1), o.stride(0), o.stride(1),
        BLOCK_SIZE, SCALE
    )

import torch

# Example tensors
q = torch.randn(1024, 64, device='cuda')
k = torch.randn(1024, 64, device='cuda')
g = torch.randn(1024, 64, device='cuda')
A = torch.zeros(1024, 64, device='cuda')
v = torch.randn(1024, 64, device='cuda')
g_cumsum = torch.cumsum(g, dim=0)
h = torch.randn(1024, 64, device='cuda')
o = torch.zeros(1024, 64, device='cuda')

# Parameters
BLOCK_SIZE = 128
SCALE = 0.1

# Compute A
chunk_fwd_intra_gated_gk_fn(q, k, g, A, BLOCK_SIZE, SCALE)

# Compute o
chunk_fwd_o_gated_gk_fn(q, v, g_cumsum, A, h, o, BLOCK_SIZE, SCALE)

print(o)
