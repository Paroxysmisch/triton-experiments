import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_fwd_kernel(
    a_ptr, b_ptr, out_ptr,
    seq_idx_ptr,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    M, N, K,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = min(num_pid_m, 65536 - first_pid_m)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    block_start_m = pid_m * BLOCK_SIZE_M
    block_start_n = pid_n * BLOCK_SIZE_N

    # Initialize pointers to A and B
    a_block_ptr = a_ptr + block_start_m * stride_am
    b_block_ptr = b_ptr + block_start_n * stride_bn

    # Initialize pointer to m and n indices
    rm = block_start_m + tl.arange(0, BLOCK_SIZE_M)
    rn = block_start_n + tl.arange(0, BLOCK_SIZE_N)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Load sequence indices if needed
    if HAS_SEQ_IDX:
        seq_idx_m = tl.load(seq_idx_ptr + rm)
        seq_idx_n = tl.load(seq_idx_ptr + rn)

    # Main loop
    for k in range(0, K, BLOCK_SIZE_K):
        k_block = tl.arange(0, BLOCK_SIZE_K) + k
        
        # Fetch fragments from A and B
        a = tl.load(a_block_ptr + k_block[None, :] * stride_ak, mask=k_block[None, :] < K, other=0.0)
        b = tl.load(b_block_ptr + k_block[:, None] * stride_bk, mask=k_block[:, None] < K, other=0.0)
        
        # Compute matrix multiplication
        acc += tl.dot(a, b)
        
        # Apply causal mask if needed
        if IS_CAUSAL:
            causal_mask = k_block[:, None] <= rn[None, :]
            acc = tl.where(causal_mask, acc, float("-inf"))

    # Apply sequence index masking if needed
    if HAS_SEQ_IDX:
        seq_mask = seq_idx_m[:, None] == seq_idx_n[None, :]
        acc = tl.where(seq_mask, acc, 0.0)

    # Write output
    out_block_ptr = out_ptr + block_start_m * stride_om + block_start_n * stride_on
    tl.store(out_block_ptr, acc, mask=(rm[:, None] < M) & (rn[None, :] < N))

def _bmm_chunk_fwd(a, b, out=None, seq_idx=None, causal=False):
    # Extract tensor dimensions
    batch_size, num_heads, seq_len, head_dim = a.shape
    _, _, _, out_dim = b.shape

    # Ensure inputs are contiguous
    a = a.contiguous()
    b = b.contiguous()

    # Compute output shape and allocate if not provided
    output_shape = (batch_size, num_heads, seq_len, out_dim)
    if out is None:
        out = torch.empty(output_shape, dtype=a.dtype, device=a.device)
    else:
        assert out.shape == output_shape, f"Expected output shape {output_shape}, got {out.shape}"
        out = out.contiguous()

    # Compute strides
    stride_am = a.stride(2)
    stride_ak = a.stride(3)
    stride_bk = b.stride(2)
    stride_bn = b.stride(3)
    stride_om = out.stride(2)
    stride_on = out.stride(3)

    # Define block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16

    # Compute grid size
    grid = (triton.cdiv(seq_len, BLOCK_SIZE_M) * triton.cdiv(out_dim, BLOCK_SIZE_N),)

    # Launch kernel
    _bmm_chunk_fwd_kernel[grid](
        a, b, out,
        seq_idx if seq_idx is not None else a.new_empty(0),  # Empty tensor if seq_idx is None
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_om, stride_on,
        seq_len, out_dim, head_dim,
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N, BLOCK_SIZE_K=BLOCK_SIZE_K,
        IS_CAUSAL=causal,
        HAS_SEQ_IDX=seq_idx is not None,
    )

    return out
