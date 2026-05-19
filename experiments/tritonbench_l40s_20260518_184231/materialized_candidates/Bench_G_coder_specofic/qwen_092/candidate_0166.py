import triton
import triton.language as tl

@triton.jit
def attention_fwd_kernel(
    q_ptr, k_ptr, v_ptr, o_ptr, b_h_ptr,
    b_s_ptr, b_o_ptr,
    n, h, T, BT, head_dim, scale,
    BLOCK_SIZE: tl.constexpr, HEAD_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(n * h, BLOCK_SIZE)

    if pid >= grid_size:
        return

    # Compute the indices for the current batch and head
    b, h = pid // (n // BT), pid % (n // BT)
    t = tl.arange(0, BLOCK_SIZE)

    # Compute the indices for the current block of the sequence
    b_start = b * BT
    b_end = b_start + BT

    # Load the query and key blocks
    q = tl.load(q_ptr + (b_start + t) * h * T * head_dim + h * T * head_dim + t * head_dim, mask=t < BT, other=0.0)
    k = tl.load(k_ptr + (b_start + t) * h * T * head_dim + h * T * head_dim + t * head_dim, mask=t < BT, other=0.0)

    # Compute the scaled dot-product attention scores
    b_s = tl.dot(q, k.T, allow_tf32=True) * scale

    # Load the value block
    v = tl.load(v_ptr + (b_start + t) * h * T * head_dim + h * T * head_dim + t * head_dim, mask=t < BT, other=0.0)

    # Compute the output block
    b_o = tl.dot(b_s, v, allow_tf32=True)

    # Accumulate the intermediate result
    tl.atomic_add(b_h_ptr + (b_start + t) * h * T * head_dim + h * T * head_dim + t * head_dim, b_o, mask=t < BT)

    # Optionally store the intermediate result
    if STORE:
        tl.store(o_ptr + (b_start + t) * h * T * head_dim + h * T * head_dim + t * head_dim, b_o, mask=t < BT)
