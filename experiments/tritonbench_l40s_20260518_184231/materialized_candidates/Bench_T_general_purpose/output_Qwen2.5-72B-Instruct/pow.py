import triton
import triton.language as tl

@triton.jit
def pow_kernel(
    x_ptr,  # Pointer to the input tensor
    exponent_ptr,  # Pointer to the exponent tensor (or scalar)
    out_ptr,  # Pointer to the output tensor
    n_elements,  # Number of elements in the input tensor
    exponent_is_scalar,  # Flag indicating if the exponent is a scalar
    BLOCK_SIZE: tl.constexpr,  # Block size for parallelization
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    if exponent_is_scalar:
        exponent = tl.load(exponent_ptr)
        out = x ** exponent
    else:
        exponent = tl.load(exponent_ptr + offsets, mask=mask)
        out = x ** exponent

    tl.store(out_ptr + offsets, out, mask=mask)

import torch
import triton
import triton.language as tl

def pow(input, exponent, *, out=None):
    # Check if exponent is a scalar or a tensor
    if isinstance(exponent, (int, float)):
        exponent = torch.tensor([exponent], dtype=input.dtype, device=input.device)
        exponent_is_scalar = True
    else:
        exponent_is_scalar = False

    # Ensure the shapes are broadcastable
    if not exponent_is_scalar:
        if not torch.broadcast_shapes(input.shape, exponent.shape) == input.shape:
            raise ValueError("Exponent tensor shape must be broadcastable with input tensor shape.")

    # Determine the number of elements
    n_elements = input.numel()

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Define grid and block sizes
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Launch the kernel
    pow_kernel[grid](
        input.contiguous().view(-1).ptr,  # Flatten the input tensor
        exponent.contiguous().view(-1).ptr,  # Flatten the exponent tensor
        out.contiguous().view(-1).ptr,  # Flatten the output tensor
        n_elements,
        exponent_is_scalar,
        BLOCK_SIZE
    )

    return out

# Example usage
input_tensor = torch.tensor([2.0, 3.0, 4.0], device='cuda')
exponent_scalar = 2.0
exponent_tensor = torch.tensor([2.0, 3.0, 4.0], device='cuda')

# Test with scalar exponent
output_scalar = pow(input_tensor, exponent_scalar)
print("Output with scalar exponent:", output_scalar)  # Expected: [4.0, 9.0, 16.0]

# Test with tensor exponent
output_tensor = pow(input_tensor, exponent_tensor)
print("Output with tensor exponent:", output_tensor)  # Expected: [4.0, 27.0, 256.0]
