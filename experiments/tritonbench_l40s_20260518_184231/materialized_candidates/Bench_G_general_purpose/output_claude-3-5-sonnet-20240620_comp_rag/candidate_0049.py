import torch
import triton
import triton.language as tl

@triton.jit
def mean_dim_kernel(
    X, # Pointer to input tensor
    Mean, # Pointer to output tensor 
    M, # Number of rows
    N, # Number of columns
    BLOCK_M: tl.constexpr, # Block size for rows
    BLOCK_N: tl.constexpr, # Block size for columns
):
    # Get program ID and compute row indices
    pid = tl.program_id(0)
    row_start = pid * BLOCK_M
    rows = row_start + tl.arange(0, BLOCK_M)
    row_mask = rows[:, None] < M

    # Initialize accumulator for mean calculation
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)
    
    # Compute offsets for input and output
    offs_x = rows[:, None] * N + tl.arange(0, BLOCK_N)[None, :]
    
    # Iterate over columns in blocks
    for n in range(0, N, BLOCK_N):
        cols = n + tl.arange(0, BLOCK_N)
        col_mask = cols[None, :] < N
        mask = row_mask & col_mask
        
        # Load and accumulate values
        x = tl.load(X + offs_x + n, mask=mask, other=0.0)
        acc += x.to(tl.float32)
    
    # Compute mean by dividing by N
    mean = tl.sum(acc, axis=1) / N
    
    # Store result
    tl.store(Mean + rows, mean, mask=rows < M)

def dim_compress(inp: torch.Tensor, dims):
    """Rearrange tensor dimensions for efficient memory access"""
    if isinstance(dims, int):
        dims = [dims]
    ndim = inp.ndim
    # Get non-reduction dimensions
    batch_dims = [i for i in range(ndim) if i not in dims]
    # Sort reduction dimensions by stride
    red_dims = sorted(dims, key=lambda x: inp.stride()[x], reverse=True)
    # Combine and permute
    order = batch_dims + red_dims
    return inp.permute(order).contiguous()

def mean_dim(x: torch.Tensor, dim, keepdim: bool = False, *, dtype=None):
    """Wrapper function to compute mean along specified dimensions"""
    if dtype is None:
        dtype = x.dtype
        
    # Handle None case (global mean)
    if dim is None:
        result = x.mean(dtype=dtype)
        return result.reshape([1] * x.ndim) if keepdim else result

    # Normalize dimensions
    if isinstance(dim, int):
        dim = [dim]
    dim = [d if d >= 0 else d + x.ndim for d in dim]
    
    # Prepare shape and compute M, N
    shape = list(x.shape)
    N = 1
    for d in dim:
        N *= shape[d]
        shape[d] = 1 if keepdim else 0
    shape = [s for s in shape if s != 0]
    
    # Compress dimensions and prepare for kernel
    x = dim_compress(x, dim)
    M = x.numel() // N
    
    # Allocate output tensor
    out = torch.empty(shape, dtype=dtype, device=x.device)
    
    # Configure grid and launch kernel
    BLOCK_M = 32
    BLOCK_N = 128
    grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]),)
    
    mean_dim_kernel[grid](
        x, out,
        M, N,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N
    )
    
    return out

# Example usage
if __name__ == "__main__":
    # Create test tensor
    x = torch.randn(2, 3, 4, 5, device="cuda")
    
    # Compute mean along dimensions 1 and 2
    result = mean_dim(x, [1, 2])
    print(result.shape)  # Should be (2, 5)
    
    # Verify against PyTorch implementation
    expected = x.mean(dim=[1, 2])
    assert torch.allclose(result, expected, rtol=1e-3, atol=1e-3)
