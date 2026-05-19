import torch
import triton
import triton.language as tl

@triton.jit
def max_kernel_1(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    max_value = tl.max(x, axis=0)
    tl.store(output_ptr + pid, max_value)

@triton.jit
def max_kernel_2(
    x_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    max_value = tl.max(x, axis=0)
    tl.store(output_ptr, max_value)

@triton.jit
def max_kernel(
    x_ptr,
    output_ptr,
    index_ptr,
    stride,
    n,
    BLOCK_SIZE: tl.constexpr,
):
    pid_x, pid_y = tl.program_id(0), tl.program_id(1)
    block_start = pid_x * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x_ptr = x_ptr + pid_y * stride
    x = tl.load(x_ptr + offsets, mask=mask, other=-float('inf'))
    max_value = tl.max(x, axis=0)
    max_index = tl.argmax(x, axis=0)
    tl.store(output_ptr + pid_y, max_value)
    tl.store(index_ptr + pid_y, max_index + block_start)

def max(x):
    n_elements = x.numel()
    BLOCK_SIZE = triton.next_power_of_2(min(n_elements, 1024))
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    output = torch.empty(grid[0], dtype=x.dtype, device=x.device)
    max_kernel_1[grid](x, output, n_elements, BLOCK_SIZE)
    
    if grid[0] > 1:
        max_kernel_2[(1,)](output, output, grid[0], BLOCK_SIZE)
    
    return output[0]

def max_dim(x, dim):
    n = x.shape[dim]
    other_dims = [d for d in range(x.ndim) if d != dim]
    x = x.permute(other_dims + [dim]).contiguous()
    x = x.view(-1, n)
    BLOCK_SIZE = triton.next_power_of_2(min(n, 1024))
    grid = (triton.cdiv(n, BLOCK_SIZE), x.shape[0])
    output = torch.empty(x.shape[0], dtype=x.dtype, device=x.device)
    indices = torch.empty(x.shape[0], dtype=torch.int64, device=x.device)
    max_kernel[grid](x, output, indices, x.stride(0), n, BLOCK_SIZE)
    return output.view(x.shape[:-1]), indices.view(x.shape[:-1])
