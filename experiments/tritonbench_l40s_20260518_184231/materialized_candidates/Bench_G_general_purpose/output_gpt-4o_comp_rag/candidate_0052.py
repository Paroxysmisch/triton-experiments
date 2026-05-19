import torch
import triton
import triton.language as tl

# Kernel 1: max_kernel_1
@triton.jit
def max_kernel_1(
    inp,
    mid,
    M,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < M
    inp_val = tl.load(inp + offset, mask=mask, other=-float("inf"))
    amax_val = tl.max(inp_val, axis=0)
    tl.store(mid + pid, amax_val)


# Kernel 2: max_kernel_2
@triton.jit
def max_kernel_2(mid, out, mid_size):
    offset = tl.arange(0, mid_size)
    mask = offset < mid_size
    mid_val = tl.load(mid + offset, mask=mask, other=-float("inf"))
    amax_val = tl.max(mid_val, axis=0)
    tl.store(out, amax_val)


# Kernel 3: max_kernel
@triton.jit
def max_kernel(
    inp,
    out,
    M,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offsets_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offsets_k = pid_k * BLOCK_K + tl.arange(0, BLOCK_K)
    mask_m = offsets_m < M
    mask_k = offsets_k < K
    mask = mask_m[:, None] & mask_k[None, :]
    
    inp_ptrs = inp + offsets_m[:, None] * K + offsets_k[None, :]
    inp_vals = tl.load(inp_ptrs, mask=mask, other=-float("inf"))
    amax_vals = tl.max(inp_vals, axis=1)
    out_ptrs = out + offsets_m
    tl.store(out_ptrs, amax_vals, mask=mask_m)


# Wrapper function for sequential execution of max_kernel_1 and max_kernel_2
def max(inp):
    M = inp.numel()
    BLOCK_SIZE = 1024  # Example block size, can be tuned
    mid_size = (M + BLOCK_SIZE - 1) // BLOCK_SIZE
    mid = torch.empty(mid_size, dtype=inp.dtype, device=inp.device)
    out = torch.empty(1, dtype=inp.dtype, device=inp.device)

    max_kernel_1[(mid_size,)](inp, mid, M, BLOCK_SIZE)
    max_kernel_2[(1,)](mid, out, mid_size)
    
    return out


# Function to compute max along a specified dimension
def max_dim(inp, dim):
    assert 0 <= dim < inp.ndim, "Invalid dimension"
    
    M = inp.shape[dim]
    K = inp.numel() // M
    BLOCK_M = 128  # Example block size, can be tuned
    BLOCK_K = 128  # Example block size, can be tuned

    out_shape = list(inp.shape)
    out_shape[dim] = 1
    out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    
    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(K, BLOCK_K))
    max_kernel[grid](inp, out, M, K, BLOCK_M, BLOCK_K)
    
    return out.squeeze(dim)

# Example usage
inp = torch.randn(1024, device='cuda')
result = max(inp)
print(result)

inp_2d = torch.randn(32, 32, device='cuda')
result_dim = max_dim(inp_2d, dim=1)
print(result_dim)
