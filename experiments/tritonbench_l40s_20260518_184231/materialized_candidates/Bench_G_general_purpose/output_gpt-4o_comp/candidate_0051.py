import triton
import triton.language as tl

# Define BLOCK_SIZE for efficient memory access and computation
BLOCK_SIZE = 1024

@triton.jit
def max_kernel_1(x_ptr, mid_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    block_max = tl.max(x, axis=0)
    tl.store(mid_ptr + pid, block_max)

@triton.jit
def max_kernel_2(mid_ptr, out_ptr, n_blocks):
    offsets = tl.arange(0, n_blocks)
    mid = tl.load(mid_ptr + offsets)
    overall_max = tl.max(mid, axis=0)
    tl.store(out_ptr, overall_max)

@triton.jit
def max_kernel(x_ptr, out_ptr, shape, stride, dim, M, K):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offsets_m = pid_m * stride[dim] + tl.arange(0, shape[dim])
    offsets_k = pid_k * stride[dim + 1] + tl.arange(0, K)
    mask = offsets_m < shape[dim]
    x = tl.load(x_ptr + offsets_m + offsets_k, mask=mask, other=-float('inf'))
    max_val = tl.max(x, axis=dim)
    tl.store(out_ptr + pid_m * K + pid_k, max_val)

def max(x):
    n_elements = x.numel()
    n_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    mid = torch.empty(n_blocks, device=x.device, dtype=x.dtype)
    out = torch.empty(1, device=x.device, dtype=x.dtype)
    
    grid = (n_blocks,)
    max_kernel_1[grid](x, mid, n_elements, BLOCK_SIZE)
    
    grid = (1,)
    max_kernel_2[grid](mid, out, n_blocks)
    
    return out.item()

def max_dim(x, dim):
    assert 0 <= dim < x.ndim, "Invalid dimension"
    shape = x.shape
    M = 1
    for i in range(dim):
        M *= shape[i]
    K = 1
    for i in range(dim + 1, x.ndim):
        K *= shape[i]
    
    out = torch.empty((M, K), device=x.device, dtype=x.dtype)
    grid = (M, K)
    max_kernel[grid](x, out, shape, x.stride(), dim, M, K)
    
    return out

# Example usage:
# tensor = torch.randn(10000, device='cuda')
# max_value = max(tensor)
# print("Max value:", max_value)

# tensor = torch.randn(10, 20, 30, device='cuda')
# max_values = max_dim(tensor, dim=1)
# print("Max values along dimension 1:", max_values)
