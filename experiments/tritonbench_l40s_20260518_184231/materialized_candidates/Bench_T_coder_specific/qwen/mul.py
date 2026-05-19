import triton
import triton.language as tl
import torch

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_X': 1024}, num_stages=1, num_warps=8),
        triton.Config({'BLOCK_SIZE_X': 512}, num_stages=1, num_warps=4),
        triton.Config({'BLOCK_SIZE_X': 256}, num_stages=1, num_warps=2),
        triton.Config({'BLOCK_SIZE_X': 128}, num_stages=1, num_warps=1),
    ],
    key=['n_elements']
)
def mul(input, other, out=None):
    # Determine the output tensor
    if out is None:
        out = torch.empty_like(input)

    # Ensure input tensors are on the same device
    assert input.device == other.device, "Input tensors must be on the same device"

    # Perform type promotion
    dtype = torch.promote_types(input.dtype, other.dtype)
    input = input.to(dtype)
    other = other.to(dtype)

    # Get the number of elements
    n_elements = input.numel()

    # Launch the kernel
    grid = (triton.cdiv(n_elements, 1024),)
    mul_kernel[grid](input.data_ptr(), other.data_ptr(), out.data_ptr(), n_elements)

    return out
