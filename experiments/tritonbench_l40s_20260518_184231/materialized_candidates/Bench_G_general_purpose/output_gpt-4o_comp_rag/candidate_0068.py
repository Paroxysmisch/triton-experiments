import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_K': 32}, num_warps=8),
        triton.Config({'BLOCK_SIZE_M': 64, 'BLOCK_SIZE_N': 64, 'BLOCK_SIZE_K': 16}, num_warps=4),
    ],
    key=['BT', 'BK', 'BV']
)
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr, o_ptr,
    s_q_h, s_q_t, s_k_h, s_k_t, s_v_h, s_v_t,
    s_h_h, s_h_t, s_g_h, s_g_t, s_o_h, s_o_t,
    scale, BT, BK, BV,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block pointers
    offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)

    # Load sub-blocks into registers
    q = tl.load(q_ptr + offs_m[:, None] * s_q_h + offs_k[None, :] * s_q_t, mask=offs_m[:, None] < BT)
    k = tl.load(k_ptr + offs_k[:, None] * s_k_h + offs_n[None, :] * s_k_t, mask=offs_n[None, :] < BK)
    v = tl.load(v_ptr + offs_k[:, None] * s_v_h + offs_n[None, :] * s_v_t, mask=offs_n[None, :] < BV)
    h = tl.load(h_ptr + offs_m[:, None] * s_h_h + offs_k[None, :] * s_h_t, mask=offs_m[:, None] < BT)
    g = tl.load(g_ptr + offs_m[:, None] * s_g_h + offs_n[None, :] * s_g_t, mask=offs_m[:, None] < BT)

    # Compute partial outputs
    b_o = tl.dot(q, k)
    b_s = tl.exp(b_o * scale)

    # Masking and final computation
    m_s = tl.load(h_ptr + offs_m[:, None] * s_h_h + offs_k[None, :] * s_h_t)
    b_s = tl.where(m_s, b_s, 0.0)
    o = tl.dot(b_s, v)

    # Store result
    tl.store(o_ptr + offs_m[:, None] * s_o_h + offs_n[None, :] * s_o_t, o, mask=offs_m[:, None] < BT)

### Wrapper Function

The wrapper function `chunk_fwd_o_fn` prepares the grid dimensions, calculates chunk sizes, and calls the kernel.
