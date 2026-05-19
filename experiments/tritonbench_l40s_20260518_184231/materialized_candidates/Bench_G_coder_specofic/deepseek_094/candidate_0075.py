import triton
import triton.language as tl

@triton.autotune
def chunk_simple_gla_fwd_kernel_o(
    q_ptr, k_ptr, v_ptr, h_ptr, g_ptr,
    s_k_h, s_k_t, s_v_h, s_v_t,
    scale,
    BT, BK, BV,
    o_ptr,
):
    # Define the indices and pointers
    pid = tl.program_id(axis=0)
    b_id = pid // (BT * BK * BV)
    t_id = (pid % (BT * BK * BV)) // (BK * BV)
    k_id = (pid % (BT * BK * BV)) % (BK * BV) // BV
    v_id = (pid % (BT * BK * BV)) % (BK * BV) % BV

    # Load the sub-blocks of q, k, h into registers
    p_q = q_ptr + b_id * s_k_h + t_id * s_k_t
    p_k = k_ptr + k_id * s_k_h + t_id * s_k_t
    p_h = h_ptr + k_id * s_k_h + t_id * s_k_t

    # Compute partial outputs b_o and b_s
    b_o = tl.dot(p_q, p_k)
    b_s = tl.dot(p_h, p_k)

    # Adjust b_o and b_s with exponentials and condition based on the mask m_s
    m_s = tl.exp(b_s)
    b_o = b_o * m_s

    # Store the result in the output tensor o
    tl.store(o_ptr + pid * scale, b_o)

def chunk_fwd_o_fn(q, k, v, h, g, scale, BT, BK, BV):
    # Calculate chunk sizes
    grid = lambda meta: (meta.n_threads, 1, 1)
    chunk_sizes = (BT, BK, BV)

    # Prepare grid dimensions
    n_threads = BT * BK * BV
    n_stages = tl.shape(q)[0] // n_threads

    # Call the kernel with the pre-computed grid and problem parameters
    chunk_simple_gla_fwd_kernel_o[grid](
        q, k, v, h, g,
        tl.shape(k)[-1], tl.shape(k)[0], tl.shape(v)[-1], tl.shape(v)[0],
        scale,
        BT, BK, BV,
        o,
    )
