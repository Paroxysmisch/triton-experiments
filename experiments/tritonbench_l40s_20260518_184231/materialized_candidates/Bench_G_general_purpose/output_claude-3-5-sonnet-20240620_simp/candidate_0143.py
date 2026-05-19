import torch
import triton
import triton.language as tl

@triton.jit
def _bmm_chunk_fwd_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, out_ptr,
    # Matrix dimensions
    batch_size, seq_len_q, seq_len_k, head_dim,
    # Strides for accessing tensors
    stride_a_batch, stride_a_seq, stride_a_head,
    stride_b_batch, stride_b_seq, stride_b_head,
    stride_out_batch, stride_out_seq_q, stride_out_seq_k,
    # Optional parameters
    chunk_size: tl.constexpr,
    causal: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    # Program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_batch = tl.program_id(2)

    # Block start indices
    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Iterate over k dimension in blocks
    for k in range(0, head_dim, BLOCK_K):
        # Load a block from matrix A
        a_block_ptr = a_ptr + (
            pid_batch * stride_a_batch +
            start_m * stride_a_seq +
            k * stride_a_head
        )
        a_block = tl.load(
            a_block_ptr,
            mask=(
                start_m + tl.arange(0, BLOCK_M) < seq_len_q,
                k + tl.arange(0, BLOCK_K) < head_dim
            ),
            other=0.0
        )

        # Load a block from matrix B
        b_block_ptr = b_ptr + (
            pid_batch * stride_b_batch +
            start_n * stride_b_seq +
            k * stride_b_head
        )
        b_block = tl.load(
            b_block_ptr,
            mask=(
                start_n + tl.arange(0, BLOCK_N) < seq_len_k,
                k + tl.arange(0, BLOCK_K) < head_dim
            ),
            other=0.0
        )

        # Compute matrix multiplication for this block
        acc += tl.dot(a_block, b_block)

    # Apply causal mask if needed
    if causal:
        causal_mask = tl.arange(0, BLOCK_M)[:, None] >= tl.arange(0, BLOCK_N)[None, :]
        acc = tl.where(causal_mask, acc, float("-inf"))

    # Write output
    out_block_ptr = out_ptr + (
        pid_batch * stride_out_batch +
        start_m * stride_out_seq_q +
        start_n * stride_out_seq_k
    )
    tl.store(
        out_block_ptr,
        acc,
        mask=(
            start_m + tl.arange(0, BLOCK_M)[:, None] < seq_len_q,
            start_n + tl.arange(0, BLOCK_N)[None, :] < seq_len_k
        )
    )

def _bmm_chunk_fwd(a: torch.Tensor, b: torch.Tensor, *, chunk_size: int = 512, causal: bool = False):
    """
    Wrapper function for the batched matrix multiplication kernel
    
    Args:
        a: Input tensor of shape (batch_size, seq_len_q, head_dim)
        b: Input tensor of shape (batch_size, seq_len_k, head_dim)
        chunk_size: Size of chunks for processing
        causal: Whether to apply causal masking
    
    Returns:
        Output tensor of shape (batch_size, seq_len_q, seq_len_k)
    """
    batch_size, seq_len_q, head_dim = a.shape
    _, seq_len_k, _ = b.shape

    # Compute optimal block sizes
    BLOCK_M = min(128, seq_len_q)
    BLOCK_N = min(128, seq_len_k)
    BLOCK_K = min(32, head_dim)

    # Create output tensor
    out = torch.empty(
        (batch_size, seq_len_q, seq_len_k),
        device=a.device,
        dtype=torch.float32
    )

    # Launch kernel
    grid = (
        triton.cdiv(seq_len_q, BLOCK_M),
        triton.cdiv(seq_len_k, BLOCK_N),
        batch_size
    )

    _bmm_chunk_fwd_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        out_ptr=out,
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        seq_len_k=seq_len_k,
        head_dim=head_dim,
        stride_a_batch=a.stride(0),
        stride_a_seq=a.stride(1),
        stride_a_head=a.stride(2),
        stride_b_batch=b.stride(0),
        stride_b_seq=b.stride(1),
        stride_b_head=b.stride(2),
        stride_out_batch=out.stride(0),
        stride_out_seq_q=out.stride(1),
        stride_out_seq_k=out.stride(2),
        chunk_size=chunk_size,
        causal=causal,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
    )

    return out
