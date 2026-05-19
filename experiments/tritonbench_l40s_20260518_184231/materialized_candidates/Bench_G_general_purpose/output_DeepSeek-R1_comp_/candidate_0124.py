import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128, 'HAS_SCALE': False}, num_warps=1),
        triton.Config({'BLOCK_SIZE': 256, 'HAS_SCALE': True}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 512, 'HAS_SCALE': False}, num_warps=8),
        triton.Config({'BLOCK_SIZE': 1024, 'HAS_SCALE': True}, num_warps=16),
    ],
    key=['N', 'D'],
)
@triton.jit
def logsumexp_fwd_kernel(
    x_ptr, z_ptr,
    N, D,
    x_stride_n, x_stride_d,
    z_stride_n, z_stride_d,
    BLOCK_SIZE: tl.constexpr,
    HAS_SCALE: tl.constexpr,
    SCALE: tl.constexpr,
):
    # Indices for the current block
    i_n = tl.program_id(0)
    i_d = tl.program_id(1)
    
    # Offsets for the current block in the last dimension
    offs_d = i_d * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offs_d < D
    
    # Pointers to the input block
    x_ptrs = x_ptr + i_n * x_stride_n + offs_d * x_stride_d
    x = tl.load(x_ptrs, mask=mask, other=-tl.infinity(x_ptr.dtype.element_ty))
    
    # Apply scaling if necessary to prevent overflow
    if HAS_SCALE:
        x_scaled = x / SCALE
    else:
        x_scaled = x
    
    # Compute the maximum value in the current block
    b_m = tl.max(x_scaled, axis=0)
    
    # Subtract max and compute exponentials
    x_minus_max = x_scaled - b_m
    exp_x = tl.exp(x_minus_max)
    
    # Sum the exponentials
    sum_exp = tl.sum(exp_x, axis=0)
    
    # Compute log sum and add back the max
    log_sum = tl.log(sum_exp) + b_m
    
    # Apply inverse scaling if needed
    if HAS_SCALE:
        log_sum = log_sum * SCALE
    
    # Store the result in the output tensor
    z_ptrs = z_ptr + i_n * z_stride_n + i_d * z_stride_d
    tl.store(z_ptrs, log_sum)

def logsumexp_fwd(x, dtype=None):
    # Reshape input to 2D (N, D)
    x_ = x.reshape(-1, x.shape[-1])
    N, D = x_.shape
    device = x_.device
    
    # Heuristic for scaling based on input dtype
    HAS_SCALE = x_.dtype in [torch.float16, torch.bfloat16]
    SCALE = 1.0
    if HAS_SCALE:
        max_value = torch.max(torch.abs(x_)).item()
        if max_value > 50.0:  # Threshold to prevent overflow in exp
            SCALE = max_value
    
    # Get the optimal BLOCK_SIZE from the tuned configurations
    with torch.cuda.device(device):
        # Define a dummy function to access the autotuner's best config
        def grid(meta):
            return (N, triton.cdiv(D, meta['BLOCK_SIZE']))
        _, best_config = triton.autotune(logsumexp_fwd_kernel.configs, key=['N', 'D'])(x_, x_, N, D, 0, 0, 0, 0, 0, False, 0.0)
        BLOCK_SIZE = best_config['BLOCK_SIZE']
        ND = triton.cdiv(D, BLOCK_SIZE)
    
    # Allocate output tensor
    z = torch.empty((N, ND), device=device, dtype=x_.dtype)
    
    # Launch the kernel
    grid = (N, ND)
    logsumexp_fwd_kernel[grid](
        x_, z, N, D,
        x_.stride(0), x_.stride(1),
        z.stride(0), z.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        HAS_SCALE=HAS_SCALE,
        SCALE=SCALE,
    )
    
    # Final reduction across the block results
    result = torch.logsumexp(z, dim=1)
    
    # Cast to desired dtype if specified
    if dtype is not None:
        result = result.to(dtype)
    
    # Reshape to original shape excluding the last dimension
    return result.reshape(*x.shape[:-1])
