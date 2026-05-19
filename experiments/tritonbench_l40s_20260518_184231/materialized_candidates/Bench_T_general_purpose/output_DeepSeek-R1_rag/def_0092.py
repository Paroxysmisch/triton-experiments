import triton
import triton.language as tl
import torch
import math

# Triton kernel for element-wise reciprocal square root
@triton.jit
def rsqrt_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    # Load data
    x = tl.load(input_ptr + offsets, mask=mask)
    # Convert to float32 for computation
    x_float = x.to(tl.float32)
    # Compute reciprocal square root
    rsqrt_x = tl.rsqrt(x_float)
    # Convert back to original data type
    output = rsqrt_x.to(x.dtype)
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function to apply Triton rsqrt kernel
def triton_rsqrt(x):
    output = torch.empty_like(x)
    n_elements = x.numel()
    if n_elements == 0:
        return output  # Handle empty tensor case
    # Ensure tensors are contiguous and flattened
    x_flat = x.contiguous().view(-1)
    output_flat = output.contiguous().view(-1)
    # Compute block and grid sizes
    max_block_size = 1024  # Reasonable maximum block size for most GPUs
    block_size = min(triton.next_power_of_2(math.ceil(math.sqrt(n_elements))), max_block_size)
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch kernel
    rsqrt_kernel[(grid_size, 1, 1)](x_flat, output_flat, n_elements, BLOCK_SIZE=block_size)
    return output

# Main wrapper function combining tensordot and rsqrt
def tensordot_rsqrt(a: torch.Tensor, b: torch.Tensor, dims) -> torch.Tensor:
    # Compute the tensor contraction
    contracted = torch.tensordot(a, b, dims=dims)
    # Apply element-wise reciprocal square root using Triton
    return triton_rsqrt(contracted)
