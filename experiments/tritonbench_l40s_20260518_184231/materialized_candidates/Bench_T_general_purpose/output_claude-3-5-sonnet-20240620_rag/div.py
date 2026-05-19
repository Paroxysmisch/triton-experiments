import triton
import triton.language as tl
import torch

# Kernel function for division
@triton.jit
def div_kernel(input_ptr, other_ptr, out_ptr, n_elements, rounding_mode, BLOCK_SIZE: tl.constexpr):
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offset < n_elements
    input_data = tl.load(input_ptr + offset, mask=mask)
    other_data = tl.load(other_ptr + offset, mask=mask)

    # Perform division
    out = input_data / other_data

    # Apply rounding if specified
    if rounding_mode == 'floor':
        out = tl.floor(out)
    elif rounding_mode == 'ceil':
        out = tl.ceil(out)
    elif rounding_mode == 'round':
        out = tl.round(out)

    tl.store(out_ptr + offset, out, mask=mask)

# Wrapper function for the division operation
def div(input, other, *, rounding_mode=None, out=None):
    if out is None:
        out = torch.empty_like(input)

    n_elements = input.numel()
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    grid_size = triton.cdiv(n_elements, block_size)

    # Launch the kernel
    div_kernel[(grid_size, 1, 1)](input, other, out, n_elements, rounding_mode, block_size)
    return out
