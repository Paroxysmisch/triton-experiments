import torch
import triton
import triton.language as tl

@triton.jit
def mean_dim_kernel(
    X,  # Input tensor pointer
    Mean,  # Output tensor pointer
    M,  # First dimension size
    N,  # Second dimension size
    stride_xm,  # Stride for X along M dimension
    stride_xn,  # Stride for X along N dimension
    stride_mean,  # Stride for output tensor
    BLOCK_M: tl.constexpr,  # Block size for M dimension
    BLOCK_N: tl.constexpr,  # Block size for N dimension
):
    # Program ID and mask computation
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    
    # Create mask for valid rows
    row_mask = row_start + tl.arange(0, BLOCK_M) < M
    
    # Initialize accumulator
    _mean = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Iterate over N dimension in blocks
    for n in range(0, N, BLOCK_N):
        # Create mask for current block
        col_mask = n + tl.arange(0, BLOCK_N) < N
        mask = row_mask[:, None] & col_mask[None, :]
        
        # Compute memory offsets
        offs_x = (row_start + tl.arange(0, BLOCK_M)[:, None]) * stride_xm + \
                 (n + tl.arange(0, BLOCK_N)[None, :]) * stride_xn
        
        # Load and accumulate
        x = tl.load(X + offs_x, mask=mask, other=0.0)
        _mean += tl.sum(x, axis=1)
    
    # Compute mean by dividing by N
    _mean = _mean / N
    
    # Store result
    offs_mean = row_start + tl.arange(0, BLOCK_M)
    tl.store(Mean + offs_mean * stride_mean, _mean, mask=row_mask)

def dim_compress(inp, dims):
    """Rearrange tensor dimensions for efficient reduction."""
    dims = sorted(dims)
    ndim = inp.ndim
    non_reduce_dims = [i for i in range(ndim) if i not in dims]
    permutation = non_reduce_dims + dims
    return inp.permute(permutation)

def mean_dim(x, dim, keepdim=False):
    """Compute mean across specified dimensions using Triton kernel."""
    # Handle negative dimensions
    dim = [d if d >= 0 else x.ndim + d for d in dim]
    
    # Compress dimensions for kernel processing
    x_compressed = dim_compress(x, dim)
    
    # Calculate M and N
    M = 1
    for i in range(x.ndim - len(dim)):
        M *= x_compressed.shape[i]
    N = 1
    for i in range(len(dim)):
        N *= x_compressed.shape[-1-i]
    
    # Prepare output tensor
    out_shape = list(x_compressed.shape[:-len(dim)])
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 32
    
    # Launch kernel
    grid = (triton.cdiv(M, BLOCK_M),)
    mean_dim_kernel[grid](
        x_compressed.contiguous(),
        out,
        M,
        N,
        x_compressed.stride(-2),
        x_compressed.stride(-1),
        out.stride(0),
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
    )
    
    # Reshape output if keepdim is True
    if keepdim:
        out_shape = list(x.shape)
        for d in dim:
            out_shape[d] = 1
        out = out.reshape(out_shape)
    
    return out
