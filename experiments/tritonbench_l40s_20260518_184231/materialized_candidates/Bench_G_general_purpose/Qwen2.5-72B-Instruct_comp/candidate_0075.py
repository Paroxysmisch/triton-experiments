import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_stages=3, num_warps=8),
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr, o_ptr,
    s_q_h, s_q_t, s_q_d,
    s_k_h, s_k_t, s_k_d,
    s_v_h, s_v_t, s_v_d,
    s_h_h, s_h_t, s_h_d,
    s_g_h, s_g_t, s_g_d,
    s_o_h, s_o_t, s_o_d,
    BT, BK, BV, scale,
    m_s: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_heads = h_ptr.shape[0]
    head_id = pid % num_heads
    block_id = pid // num_heads

    # Compute the block indices
    block_m = block_id * BT
    block_n = block_id * BV

    # Compute the block pointers
    p_q = q_ptr + head_id * s_q_h + block_m * s_q_t
    p_k = k_ptr + head_id * s_k_h + block_n * s_k_t
    p_v = v_ptr + head_id * s_v_h + block_n * s_v_t
    p_h = h_ptr + head_id * s_h_h + block_m * s_h_t
    p_g = g_ptr + head_id * s_g_h + block_m * s_g_t
    p_o = o_ptr + head_id * s_o_h + block_m * s_o_t

    # Load the sub-blocks into registers
    q = tl.load(p_q, mask=block_m + tl.arange(0, BT) < q_ptr.shape[1])
    k = tl.load(p_k, mask=block_n + tl.arange(0, BV) < k_ptr.shape[1])
    v = tl.load(p_v, mask=block_n + tl.arange(0, BV) < v_ptr.shape[1])
    h = tl.load(p_h, mask=block_m + tl.arange(0, BT) < h_ptr.shape[1])
    g = tl.load(p_g, mask=block_m + tl.arange(0, BT) < g_ptr.shape[1])

    # Compute the partial outputs
    b_o = tl.zeros((BT, BV), dtype=tl.float32)
    b_s = tl.zeros((BT, BV), dtype=tl.float32)

    for i in range(0, BK, 32):
        q_chunk = q[:, i:i+32]
        k_chunk = k[:, i:i+32]
        v_chunk = v[:, i:i+32]

        # Compute the dot product
        s = tl.dot(q_chunk, k_chunk, allow_tf32=True) * scale

        # Apply the mask
        s = tl.where(s > m_s, s, m_s)

        # Compute the exponential and normalize
        s_exp = tl.exp(s)
        s_sum = tl.sum(s_exp, axis=1)

        # Compute the weighted sum
        b_o += tl.dot(s_exp, v_chunk, allow_tf32=True)
        b_s += s_sum

    # Normalize the output
    b_o /= b_s[:, None]

    # Store the result
    tl.store(p_o, b_o, mask=block_m + tl.arange(0, BT) < o_ptr.shape[1])

import torch

def chunk_fwd_o_fn(q, k, v, h, g, scale, BT, BK, BV, m_s):
    # Prepare the grid dimensions
    num_heads = q.shape[0]
    num_blocks = (q.shape[1] + BT - 1) // BT
    grid = (num_heads * num_blocks,)

    # Prepare the strides
    s_q_h, s_q_t, s_q_d = q.stride(0), q.stride(1), q.stride(2)
    s_k_h, s_k_t, s_k_d = k.stride(0), k.stride(1), k.stride(2)
    s_v_h, s_v_t, s_v_d = v.stride(0), v.stride(1), v.stride(2)
    s_h_h, s_h_t, s_h_d = h.stride(0), h.stride(1), h.stride(2)
    s_g_h, s_g_t, s_g_d = g.stride(0), g.stride(1), g.stride(2)
    s_o_h, s_o_t, s_o_d = q.stride(0), q.stride(1), q.stride(2)

    # Allocate the output tensor
    o = torch.empty_like(q)

    # Call the kernel
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o,
        s_q_h, s_q_t, s_q_d,
        s_k_h, s_k_t, s_k_d,
        s_v_h, s_v_t, s_v_d,
        s_h_h, s_h_t, s_h_d,
        s_g_h, s_g_t, s_g_d,
        s_o_h, s_o_t, s_o_d,
        BT, BK, BV, scale,
        m_s
    )

    return o
