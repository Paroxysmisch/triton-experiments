import triton
import triton.language as tl
import torch
import math

@triton.jit
def relu_sqrt_kernel(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    a_val = tl.load(a_ptr + offset, mask=mask)
    # Apply ReLU
    relu_val = tl.where(a_val > 0, a_val, 0.0)
    # Compute the square root
    sqrt_val = tl.sqrt(relu_val.to(tl.float32))
    tl.store(b_ptr + offset, sqrt_val, mask=mask)

def relu_sqrt(input, inplace=False, out=None) -> torch.Tensor:
    if inplace:
        output = input
    elif out is not None:
        output = out
    else:
        output = torch.empty_like(input)

    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid = triton.cdiv(n_elements, block_size)

    relu_sqrt_kernel[(grid,)](
        input, 
        output, 
        n_elements, 
        block_size
    )
    return output
