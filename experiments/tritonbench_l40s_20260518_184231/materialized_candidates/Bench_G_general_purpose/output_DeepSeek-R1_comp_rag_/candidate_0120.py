import triton
import triton.language as tl
import torch

def next_power_of_2(n):
    n -= 1
    n |= n >> 1
    n |= n >> 2
    n |= n >> 4
    n |= n >> 8
    n |= n >> 16
    n += 1
    return n

@triton.autotune(
    configs=[
        triton.Config({'num_warps': num_warps}, num_warps=num_warps)
        for num_warps in [1, 2, 4, 8, 16, 32]
    ],
    key=['N', 'D'],
)
@triton.jit
def logsumexp_fwd_kernel(
    x_ptr,
    z_ptr,
    N,
    D,
    stride_x_n,
    stride_x_d,
    stride_z_n,
    stride_z_d,
    BLOCK: tl.constexpr,
    HAS_SCALE: tl.constexpr,
):
    i_n = tl.program_id(0)
    i_d = tl.program_id(1)
    
    offs_d = i_d * BLOCK + tl.arange(0, BLOCK)
    mask = offs_d < D
    
    x_ptrs = x_ptr + i_n * stride_x_n + offs_d * stride_x_d
    x = tl.load(x_ptrs, mask=mask, other=-float('inf'))
    x = x.to(tl.float32)
    
    max_x = tl.max(x, axis=0)
    
    if HAS_SCALE:
        x_scaled = x - max_x
    else:
        x_scaled = x
    
    exp_x = tl.exp(x_scaled)
    sum_exp = tl.sum(exp_x, axis=0)
    log_sum_exp = tl.log(sum_exp) + max_x
    
    z_ptrs = z_ptr + i_n * stride_z_n + i_d * stride_z_d
    tl.store(z_ptrs, log_sum_exp)

def logsumexp_fwd(x, dtype=None):
    original_shape = x.shape
    x_flat = x.reshape(-1, original_shape[-1])
    N, D = x_flat.shape
    
    BLOCK = 1024  # Example block size; can be tuned based on D
    ND = (D + BLOCK - 1) // BLOCK
    z = torch.empty((N, ND), dtype=torch.float32, device=x.device)
    
    # Heuristic: Enable scaling if not in float32 to prevent overflow
    HAS_SCALE = x_flat.dtype != torch.float32
    
    grid = (N, ND)
    logsumexp_fwd_kernel[grid](
        x_flat, z, N, D,
        x_flat.stride(0), x_flat.stride(1),
        z.stride(0), z.stride(1),
        BLOCK=BLOCK,
        HAS_SCALE=HAS_SCALE,
    )
    
    result = torch.logsumexp(z, dim=1)
    
    if dtype is not None:
        result = result.to(dtype)
    
    return result.reshape(*original_shape[:-1])
