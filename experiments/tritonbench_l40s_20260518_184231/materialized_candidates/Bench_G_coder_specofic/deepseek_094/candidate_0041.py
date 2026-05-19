import triton
import triton.lang as tl

@triton.jit
def _triton_rope(
    q_ptr,
    k_ptr,
    cos_ptr,
    sin_ptr,
    batch_stride,
    seq_len,
    head_dim,
    num_rope_partitions,
    BACKWARD_PASS: tl.constexpr,
    **meta
):
    pid = tl.program_id()
    batch_idx = pid // seq_len
    seq_idx = pid % seq_len

    # Calculate the starting index for this thread
    start_idx = batch_idx * batch_stride + seq_idx * head_dim

    # Load slices from the q and k matrices
    q = tl.load(q_ptr + start_idx)
    k = tl.load(k_ptr + start_idx)

    # Apply rotary transformations
    for i in range(head_dim // num_rope_partitions):
        offset = i * num_rope_partitions
        for j in range(num_rope_partitions):
            cos_val = tl.load(cos_ptr + offset + j)
            sin_val = tl.load(sin_ptr + offset + j)
            q_slice = q[j::num_rope_partitions]
            k_slice = k[j::num_rope_partitions]
            q[j::num_rope_partitions] = q_slice * cos_val - k_slice * sin_val
            k[j::num_rope_partitions] = q_slice * sin_val + k_slice * cos_val

    # Store the transformed slices back
    tl.store(q_ptr + start_idx, q)
    tl.store(k_ptr + start_idx, k)
