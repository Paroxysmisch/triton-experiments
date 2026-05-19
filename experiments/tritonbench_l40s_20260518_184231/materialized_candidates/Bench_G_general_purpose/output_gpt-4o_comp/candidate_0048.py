import triton
import triton.language as tl
import torch

@triton.jit
def mean_dim_kernel(X, Mean, M, N, BLOCK_M, BLOCK_N, **meta):
    pid = tl.program_id(0)
    
    # Compute the start of the block
    start_m = pid * BLOCK_M
    
    # Create a mask for the row
    row_mask = start_m + tl.arange(0, BLOCK_M) < M
    
    # Initialize mean accumulator
    _mean = tl.zeros([BLOCK_M], dtype=tl.float32)
    
    # Iterate over the columns in BLOCK_N chunks
    for start_n in range(0, N, BLOCK_N):
        # Load a block of data from X
        block = tl.load(X + start_m * N + start_n, mask=row_mask[:, None] & (tl.arange(0, BLOCK_N) < N))
        
        # Accumulate the sum
        _mean += tl.sum(block, axis=1)
    
    # Compute the mean
    _mean = _mean / N
    
    # Store the result in Mean
    tl.store(Mean + start_m, _mean, mask=row_mask)

def dim_compress(inp, dims):
    # Permute dimensions to bring 'dims' to the end
    all_dims = list(range(inp.ndim))
    for dim in sorted(dims, reverse=True):
        all_dims.remove(dim)
    perm = all_dims + dims
    return inp.permute(perm)

def mean_dim(x, dim, keepdim=False):
    # Compress dimensions
    x = dim_compress(x, dim)
    
    # Calculate M and N
    dims_to_reduce = [x.size(d) for d in dim]
    N = 1
    for d in dims_to_reduce:
        N *= d
    M = x.numel() // N
    
    # Define block sizes
    BLOCK_M = 128
    BLOCK_N = 128
    
    # Prepare output tensor
    out_shape = list(x.shape)
    for d in sorted(dim, reverse=True):
        del out_shape[d]
    out_shape.append(1) if keepdim else None
    out = torch.empty(out_shape, dtype=x.dtype, device=x.device)
    
    # Launch kernel
    grid = lambda META: (triton.cdiv(M, BLOCK_M),)
    mean_dim_kernel[grid](x, out, M, N, BLOCK_M, BLOCK_N)
    
    # Squeeze out reduced dimensions if keepdim is False
    if not keepdim:
        out = out.squeeze(dim)
    
    return out

# Example usage
x = torch.randn(4, 4, device='cuda')
result = mean_dim(x, dim=[1], keepdim=False)
print(result)
