import triton
import triton.language as tl
import torch
import math

@triton.jit
def sqrt_exp_kernel(
    a_ptr,
    b_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input elements
    a = tl.load(a_ptr + offsets, mask=mask)
    # Compute square root and cast to float32 for computation
    sqrt_val = tl.sqrt(a.to(tl.float32))
    # Apply exponential function
    exp_val = tl.exp(sqrt_val)
    # Cast result to the output tensor's data type
    exp_val = exp_val.to(b_ptr.dtype.element_ty)
    # Store the result
    tl.store(b_ptr + offsets, exp_val, mask=mask)

def sqrt_exp(input, out=None) -> torch.Tensor:
    # Ensure input is on CUDA
    assert input.is_cuda, "Input tensor must be on CUDA device"

    # Create or validate the output tensor
    if out is None:
        out = torch.empty_like(input, dtype=torch.float32)
    else:
        assert out.shape == input.shape, "Output tensor shape must match input"
        assert out.is_cuda, "Output tensor must be on CUDA device"
        assert out.dtype.is_floating_point, "Output tensor must be floating-point"

    n_elements = input.numel()
    # Calculate block size as next power of two of sqrt(n_elements)
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Compute grid size to cover all elements
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch kernel
    sqrt_exp_kernel[grid_size](input, out, n_elements, BLOCK_SIZE=block_size)
    
    return out
