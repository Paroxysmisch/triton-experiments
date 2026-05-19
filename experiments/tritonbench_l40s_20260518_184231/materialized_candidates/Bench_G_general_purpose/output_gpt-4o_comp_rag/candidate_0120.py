import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'num_warps': 1}),
        triton.Config({'num_warps': 2}),
        triton.Config({'num_warps': 4}),
        triton.Config({'num_warps': 8}),
        triton.Config({'num_warps': 16}),
        triton.Config({'num_warps': 32}),
    ],
    key=['N', 'D']
)
@triton.jit
def logsumexp_fwd_kernel(x_ptr, z_ptr, N, D, HAS_SCALE: tl.constexpr, **meta):
    pid_n = tl.program_id(0)
    pid_d = tl.program_id(1)
    
    # Create block indices
    block_n = pid_n * meta['BLOCK_SIZE_N']
    block_d = pid_d * meta['BLOCK_SIZE_D']
    
    # Compute indices within the block
    idx_n = block_n + tl.arange(0, meta['BLOCK_SIZE_N'])
    idx_d = block_d + tl.arange(0, meta['BLOCK_SIZE_D'])
    
    # Create masks to handle out-of-bounds accesses
    mask_n = idx_n < N
    mask_d = idx_d < D
    
    # Load the input block
    x = tl.load(x_ptr + idx_n[:, None] * D + idx_d[None, :], mask=mask_n[:, None] & mask_d[None, :], other=-float('inf'))
    
    # Optionally scale the input
    if HAS_SCALE:
        x = x * tl.load(scale_ptr)
    
    # Compute the maximum value along the last dimension
    max_val = tl.max(x, axis=1)
    
    # Compute the log-sum-exp
    x_exp = tl.exp(x - max_val[:, None])
    sum_exp = tl.sum(x_exp, axis=1)
    log_sum_exp = tl.log(sum_exp) + max_val
    
    # Store the result
    tl.store(z_ptr + idx_n, log_sum_exp, mask=mask_n)

def logsumexp_fwd(x, has_scale=False):
    N, D = x.shape
    B = triton.next_power_of_2(D)
    ND = D // B
    z = torch.empty((N,), dtype=x.dtype, device=x.device)
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE_N']), triton.cdiv(D, meta['BLOCK_SIZE_D']))
    logsumexp_fwd_kernel[grid](x, z, N, D, has_scale, BLOCK_SIZE_N=128, BLOCK_SIZE_D=128)
    
    return z

# Example usage
x = torch.randn(1024, 512, device='cuda')
result = logsumexp_fwd(x)
print(result)
