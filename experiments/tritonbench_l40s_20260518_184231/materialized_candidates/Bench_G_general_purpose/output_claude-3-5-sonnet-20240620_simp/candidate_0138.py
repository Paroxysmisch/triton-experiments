import triton
import triton.language as tl

# Define block sizes
BLOCK_M = 128
BLOCK_N = 128
BLOCK_D = 64

@triton.jit
def block_sparse_attention_kernel(
    q_ptr, k_ptr, v_ptr, out_ptr,
    row_ptr, col_idx,
    seq_len, num_heads, head_dim,
    stride_qm, stride_qh, stride_qd,
    stride_km, stride_kh, stride_kd,
    stride_vm, stride_vh, stride_vd,
    stride_om, stride_oh, stride_od,
    NUM_D_BLOCKS: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_D: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Compute the block indices
    block_m = pid // (num_heads * (seq_len // BLOCK_M))
    block_h = (pid % (num_heads * (seq_len // BLOCK_M))) // (seq_len // BLOCK_M)
    block_n = (pid % (num_heads * (seq_len // BLOCK_M))) % (seq_len // BLOCK_M)

    # Compute the starting indices for this block
    start_m = block_m * BLOCK_M
    start_n = block_n * BLOCK_N

    # Load the row pointers for this block
    row_start = tl.load(row_ptr + start_m)
    row_end = tl.load(row_ptr + start_m + BLOCK_M)

    # Initialize the output accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_D), dtype=tl.float32)

    # Iterate over the non-zero blocks in this row
    for block_idx in range(row_start, row_end):
        col_block = tl.load(col_idx + block_idx)
        
        # Load Q, K, and V blocks
        q = tl.load(q_ptr + start_m * stride_qm + block_h * stride_qh + tl.arange(0, BLOCK_M)[:, None] * stride_qd + tl.arange(0, BLOCK_D)[None, :])
        k = tl.load(k_ptr + col_block * BLOCK_N * stride_km + block_h * stride_kh + tl.arange(0, BLOCK_N)[:, None] * stride_kd + tl.arange(0, BLOCK_D)[None, :])
        v = tl.load(v_ptr + col_block * BLOCK_N * stride_vm + block_h * stride_vh + tl.arange(0, BLOCK_N)[:, None] * stride_vd + tl.arange(0, BLOCK_D)[None, :])

        # Compute attention scores
        scores = tl.dot(q, k.transpose())
        scores = scores * (1.0 / tl.sqrt(float(head_dim)))

        # Apply softmax
        scores = tl.softmax(scores)

        # Compute weighted sum of values
        acc += tl.dot(scores, v)

    # Store the output
    tl.store(out_ptr + start_m * stride_om + block_h * stride_oh + tl.arange(0, BLOCK_M)[:, None] * stride_od + tl.arange(0, BLOCK_D)[None, :], acc)

# Wrapper function
def block_sparse_attention(q, k, v, row_ptr, col_idx, seq_len, num_heads):
    # Get input shapes and sizes
    batch_size, _, head_dim = q.shape
    
    # Compute grid size
    grid = (batch_size * num_heads * (seq_len // BLOCK_M),)
    
    # Prepare output tensor
    out = torch.empty_like(q)
    
    # Launch kernel
    block_sparse_attention_kernel[grid](
        q, k, v, out,
        row_ptr, col_idx,
        seq_len, num_heads, head_dim,
        q.stride(0), q.stride(1), q.stride(2),
        k.stride(0), k.stride(1), k.stride(2),
        v.stride(0), v.stride(1), v.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        NUM_D_BLOCKS=head_dim // BLOCK_D,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_D=BLOCK_D,
    )
    
    return out
