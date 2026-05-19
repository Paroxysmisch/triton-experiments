import triton
import triton.language as tl
import torch

@triton.jit
def _bmm_chunk_fwd_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, out_ptr,
    # Matrix dimensions
    M, N, K,
    # Batch/chunk/group dimensions
    batch_size, num_chunks, num_groups, num_heads,
    # Sequence index pointer (optional)
    seq_idx_ptr,
    # Matrix strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_om, stride_on,
    # Batch/chunk/group strides
    stride_batch, stride_chunk, stride_group, stride_head,
    # Constants
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    IS_CAUSAL: tl.constexpr,
    HAS_SEQ_IDX: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    
    # Compute indices for batch, chunk, group, head
    group_id = pid // num_pid_in_group
    pid_in_group = pid % num_pid_in_group
    batch_id = group_id // (num_chunks * num_groups * num_heads)
    chunk_id = (group_id // (num_groups * num_heads)) % num_chunks
    group_id = (group_id // num_heads) % num_groups
    head_id = group_id % num_heads
    
    # Block indices for the current program
    block_m = (pid_in_group // num_pid_n) * BLOCK_SIZE_M
    block_n = (pid_in_group % num_pid_n) * BLOCK_SIZE_N

    # Initialize pointers to current batch/chunk/group/head
    a_ptr = a_ptr + batch_id * stride_batch + chunk_id * stride_chunk + \
            group_id * stride_group + head_id * stride_head
    b_ptr = b_ptr + batch_id * stride_batch + chunk_id * stride_chunk + \
            group_id * stride_group + head_id * stride_head
    
    # Initialize the accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Pointers to rows and columns
    offs_am = (block_m + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (block_n + tl.arange(0, BLOCK_SIZE_N)) % N
    
    # Load sequence indices if needed
    if HAS_SEQ_IDX:
        seq_idx_m = tl.load(seq_idx_ptr + offs_am)
        seq_idx_n = tl.load(seq_idx_ptr + offs_bn)
    
    # Main loop
    for k in range(0, K, BLOCK_SIZE_K):
        offs_k = tl.arange(0, BLOCK_SIZE_K) + k
        
        # Fetch fragments from A and B
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak,
                   mask=(offs_k[None, :] < K), other=0.0)
        b = tl.load(b_ptr + offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn,
                   mask=(offs_k[:, None] < K), other=0.0)
        
        # Compute matrix multiplication
        acc += tl.dot(a, b)
    
    # Apply causal mask if needed
    if IS_CAUSAL:
        causal_mask = offs_am[:, None] >= offs_bn[None, :]
        acc = tl.where(causal_mask, acc, 0.0)
    
    # Apply sequence index mask if needed
    if HAS_SEQ_IDX:
        seq_mask = seq_idx_m[:, None] == seq_idx_n[None, :]
        acc = tl.where(seq_mask, acc, 0.0)
    
    # Store the result
    out_ptr = out_ptr + batch_id * stride_batch + chunk_id * stride_chunk + \
              group_id * stride_group + head_id * stride_head
    tl.store(out_ptr + offs_am[:, None] * stride_om + offs_bn[None, :] * stride_on,
             acc, mask=(offs_am[:, None] < M) & (offs_bn[None, :] < N))

def _bmm_chunk_fwd(a: torch.Tensor, b: torch.Tensor, seq_idx: torch.Tensor = None,
                   is_causal: bool = False):
    # Extract dimensions
    batch_size, num_chunks, num_groups, num_heads, M, K = a.shape
    _, _, _, _, K, N = b.shape
    
    # Handle non-contiguous inputs
    a = a.contiguous()
    b = b.contiguous()
    
    # Allocate output
    out = torch.empty((batch_size, num_chunks, num_groups, num_heads, M, N),
                     device=a.device, dtype=a.dtype)
    
    # Configure meta-parameters
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    
    # Calculate grid size
    grid = (triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N) *
            batch_size * num_chunks * num_groups * num_heads,)
    
    # Launch kernel
    _bmm_chunk_fwd_kernel[grid](
        a, b, out,
        M, N, K,
        batch_size, num_chunks, num_groups, num_heads,
        seq_idx if seq_idx is not None else torch.empty(0, device=a.device),
        a.stride(-2), a.stride(-1),
        b.stride(-2), b.stride(-1),
        out.stride(-2), out.stride(-1),
        a.stride(0), a.stride(1), a.stride(2), a.stride(3),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        IS_CAUSAL=is_causal,
        HAS_SEQ_IDX=seq_idx is not None,
    )
    
    return out
