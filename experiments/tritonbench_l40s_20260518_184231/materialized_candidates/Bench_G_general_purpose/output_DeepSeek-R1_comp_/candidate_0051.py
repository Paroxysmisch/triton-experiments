import triton
import triton.language as tl
import torch

@triton.jit
def max_kernel_1(
    input_ptr,
    mid_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < n_elements
    block = tl.load(input_ptr + offset, mask=mask, other=-float('inf'))
    block_max = tl.max(block, axis=0)
    tl.store(mid_ptr + pid, block_max)

@triton.jit
def max_kernel_2(
    mid_ptr,
    out_ptr,
    n_mid_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset = pid * BLOCK_SIZE
    mask = offset + tl.arange(0, BLOCK_SIZE) < n_mid_elements
    block = tl.load(mid_ptr + offset, mask=mask, other=-float('inf'))
    block_max = tl.max(block, axis=0)
    tl.atomic_max(out_ptr, block_max)

@triton.jit
def max_kernel(
    input_ptr,
    output_ptr,
    D,
    M,
    K,
    stride_m,
    stride_d,
    stride_k,
    BLOCK_SIZE: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    m = pid_k // K
    k = pid_k % K
    
    offset_d = pid_m * BLOCK_SIZE
    indices = offset_d + tl.arange(0, BLOCK_SIZE)
    mask = indices < D
    
    input_offset = m * stride_m + indices * stride_d + k * stride_k
    block = tl.load(input_ptr + input_offset, mask=mask, other=-float('inf'))
    block_max = tl.max(block, axis=0)
    
    output_offset = m * K + k
    tl.atomic_max(output_ptr + output_offset, block_max)

def max(input_tensor):
    n_elements = input_tensor.numel()
    BLOCK_SIZE = 1024  # Optimal block size based on heuristics
    mid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    mid = torch.full((mid_size,), -float('inf'), dtype=input_tensor.dtype, device=input_tensor.device)
    out = torch.tensor(-float('inf'), dtype=input_tensor.dtype, device=input_tensor.device)
    
    grid_1 = (mid_size,)
    max_kernel_1[grid_1](input_tensor, mid, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    
    grid_2 = ((mid_size + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    max_kernel_2[grid_2](mid, out, mid_size, BLOCK_SIZE=BLOCK_SIZE)
    
    return out

def max_dim(input_tensor, dim):
    assert 0 <= dim < input_tensor.ndim, "Invalid dimension"
    
    shape = input_tensor.shape
    M = int(torch.prod(torch.tensor(shape[:dim])).item())
    D = shape[dim]
    K = int(torch.prod(torch.tensor(shape[dim+1:])).item())
    
    input_reshaped = input_tensor.view(M, D, K) if input_tensor.ndim > 1 else input_tensor.view(1, D, 1)
    output = torch.full((M, K), -float('inf'), dtype=input_tensor.dtype, device=input_tensor.device)
    
    BLOCK_SIZE = 1024  # Optimal block size based on heuristics
    grid_m = (D + BLOCK_SIZE - 1) // BLOCK_SIZE
    grid_k = M * K
    
    max_kernel[(grid_m, grid_k)](
        input_reshaped, output, D, M, K,
        input_reshaped.stride(0), input_reshaped.stride(1), input_reshaped.stride(2),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    output_shape = list(shape)
    del output_shape[dim]
    return output.view(output_shape) if len(output_shape) > 0 else output.view(())
