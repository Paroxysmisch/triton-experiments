import triton
import triton.language as tl

@triton.jit
def pow_kernel(input_ptr, exponent_ptr, out_ptr, n_elements, is_scalar_exponent, BLOCK_SIZE: tl.constexpr):
    # Compute the index of the element this program is responsible for
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements

    # Load input elements
    x = tl.load(input_ptr + idx, mask=mask)

    # Load exponent elements (handle scalar or tensor case)
    if is_scalar_exponent:
        exponent = tl.load(exponent_ptr, mask=mask)
    else:
        exponent = tl.load(exponent_ptr + idx, mask=mask)

    # Compute the power
    out = x ** exponent

    # Store the result
    tl.store(out_ptr + idx, out, mask=mask)

import torch

def pow(input, exponent, *, out=None):
    # Determine the size of the input tensor
    n_elements = input.numel()
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(input)

    # Determine if the exponent is a scalar
    is_scalar_exponent = isinstance(exponent, (int, float))

    # Prepare the exponent tensor
    if is_scalar_exponent:
        # If the exponent is a scalar, we create a tensor with the same value
        exponent_tensor = torch.full((1,), exponent, dtype=input.dtype, device=input.device)
    else:
        # Ensure the shapes are broadcastable
        exponent_tensor = torch.broadcast_to(exponent, input.shape)

    # Launch the Triton kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    pow_kernel[grid](input, exponent_tensor, out, n_elements, is_scalar_exponent, BLOCK_SIZE=BLOCK_SIZE)

    return out
