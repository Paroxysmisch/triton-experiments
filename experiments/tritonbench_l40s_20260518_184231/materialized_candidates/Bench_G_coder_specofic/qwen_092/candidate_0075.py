import triton
import triton.language as tl

@triton.autotune
@triton.jit
def chunk_simple_gla_fwd_kernel_o(
    q, k, v, h, g, o, m_s, s_k_h, s_k_t, s_k_r, s_q_h, s_q_t, s_q_r, s_h_h, s_h_t, s_h_r, s_o_h, s_o_t, s_o_r, s_g_h, s_g_t, s_g_r, scale, BT, BK, BV
):
    # Get the program ID and indices
    pid = tl.program_id(axis=0)
    i = pid // BK
    j = pid % BK

    # Compute block pointers
    p_q = q + i * s_q_h + j * s_q_t
    p_k = k + i * s_k_h + j * s_k_t
    p_h = h + i * s_h_h + j * s_h_t
    p_o = o + i * s_o_h + j * s_o_t

    # Load sub-blocks of q, k, h into registers
    q_block = tl.load(p_q, (BT, BK, BV), mask=(i < BT, j < BK, True))
    k_block = tl.load(p_k, (BT, BK, BV), mask=(i < BT, j < BK, True))
    h_block = tl.load(p_h, (BT, BK, BV), mask=(i < BT, j < BK, True))

    # Initialize partial outputs
    b_o = tl.zeros((BT, BK, BV), dtype=tl.float32)
    b_s = tl.zeros((BT, BK, BV), dtype=tl.float32)

    # Compute partial outputs using dot products
    for r in range(BV):
        b_o += q_block * k_block[:, :, r]
        b_s += h_block * g[:, :, r]

    # Adjust with exponentials and conditions based on the mask
    b_o = b_o * scale
    b_s = b_s * scale
    b_o = tl.where(m_s, b_o, 0.0)
    b_s = tl.where(m_s, b_s, 0.0)

    # Store the result in the output tensor
    tl.store(p_o, b_o, mask=(i < BT, j < BK, True))
    tl.store(p_o + s_o_r, b_s, mask=(i < BT, j < BK, True))

def chunk_fwd_o_fn(q, k, v, h, g, o, m_s, s_k_h, s_k_t, s_k_r, s_q_h, s_q_t, s_q_r, s_h_h, s_h_t, s_h_r, s_o_h, s_o_t, s_o_r, s_g_h, s_g_t, s_g_r, scale, BT, BK, BV):
    # Calculate chunk sizes
    N = q.shape[0]
    M = k.shape[1]
    P = v.shape[2]

    # Calculate the number of blocks
    num_blocks = (N * M * P + BT * BK * BV - 1) // (BT * BK * BV)

    # Create the grid
    grid = (num_blocks,)

    # Call the kernel with the pre-computed grid and problem parameters
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g, o, m_s, s_k_h, s_k_t, s_k_r, s_q_h, s_q_t, s_q_r, s_h_h, s_h_t, s_h_r, s_o_h, s_o_t, s_o_r, s_g_h, s_g_t, s_g_r, scale, BT, BK, BV
    )

# Example usage
# Assuming you have tensors q, k, v, h, g, o, m_s and their strides
# chunk_fwd_o_fn(q, k, v, h, g, o, m_s, s_k_h, s_k_t, s_k_r, s_q_h, s_q_t, s_q_r, s_h_h, s_h_t, s_h_r, s_o_h, s_o_t, s_o_r, s_g_h, s_g_t, s_g_r, scale, BT, BK, BV)
