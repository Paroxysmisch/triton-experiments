triton
@triton.jit
def _fwd_kernel(
    Q, K, V, Out, mask, sm_scale, stride_q, stride_k, stride_v, stride_out, stride_mask, 
    kv_group_num, BLOCK_M: int, BLOCK_N: int, num_heads: int, max_seq_len: int
):
    pid = triton.program_id(0)
    grid_m = (num_heads * max_seq_len - 1) // BLOCK_M + 1
    m = pid % grid_m
    n = pid // grid_m

    # Calculate the indices
    head = m // max_seq_len
    seq_len = m % max_seq_len
    kv_group = n // (BLOCK_N // kv_group_num)
    kv_head = n % (BLOCK_N // kv_group_num)

    # Get the indices for Q, K, V, Out, and mask
    q_idx = head * stride_q + seq_len * stride_q // num_heads + kv_group * stride_q // kv_group_num
    k_idx = head * stride_k + seq_len * stride_k // num_heads + kv_group * stride_k // kv_group_num
    v_idx = head * stride_v + seq_len * stride_v // num_heads + kv_group * stride_v // kv_group_num
    out_idx = head * stride_out + seq_len * stride_out // num_heads + kv_group * stride_out // kv_group_num
    mask_idx = head * stride_mask + seq_len * stride_mask // num_heads + kv_group * stride_mask // kv_group_num

    # Load Q, K, V, and mask
    q = Q[q_idx]
    k = K[k_idx]
    v = V[v_idx]
    msk = mask[mask_idx]

    # Initialize the output
    out = 0.0

    # Compute the dot product and softmax
    for i in range(BLOCK_N):
        dot_product = q @ k[i]
        scaled_dot_product = dot_product * sm_scale
        exp_scaled_dot_product = triton.math.exp(scaled_dot_product)
        if msk[i] == 0:
            exp_scaled_dot_product = 0.0
        out += exp_scaled_dot_product * v[i]

    # Store the result
    Out[out_idx] = out
