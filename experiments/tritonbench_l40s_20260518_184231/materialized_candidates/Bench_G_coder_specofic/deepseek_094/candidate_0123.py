import triton
import triton.language as tl

@triton.jit
def chunk_global_reversed_cumsum_vector_kernel(
    s_ptr,
    z_ptr,
    B,
    H,
    T,
    S,
    dtype,
    BS=32,
    BT=32,
    **meta
):
    # Define the block pointer
    b_s = tl.load(s_ptr + tl.program_id(0) * BS * S)
    b_z = tl.load(z_ptr + tl.program_id(0) * BS * S)

    # Compute the masked dot product
    m_s = tl.dot(b_s, tl.tril(tl.ones((BS, S))))

    # Store the result
    tl.store(z_ptr + tl.program_id(0) * BS * S, m_s)

    # Update the cumulative sum
    b_z += b_s
    tl.store(z_ptr + tl.program_id(0) * BS * S, b_z)
