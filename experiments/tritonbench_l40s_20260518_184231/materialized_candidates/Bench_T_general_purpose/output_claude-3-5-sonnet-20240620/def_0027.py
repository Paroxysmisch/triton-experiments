import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def sqrt_tanh_kernel(input_ptr, output_ptr, n_elements):
    # Compute the square root and then apply tanh
    pid = tl.program_id(0)
    start = pid * tl.num_warps() * tl.warp_size()
    end = start + n_elements

    for i in range(start, end):
        if i < n_elements:
            value = tl.load(input_ptr + i)
            # Handle negative values
            sqrt_value = tl.sqrt(value) if value >= 0 else 0
            tl.store(output_ptr + i, tl.tanh(sqrt_value))

def sqrt_tanh(input: Tensor, out: Tensor = None) -> Tensor:
    # Ensure output tensor is created if not provided
    if out is None:
        out = input.new_zeros(input.shape)

    # Launch the kernel
    n_elements = input.numel()
    grid = (triton.cdiv(n_elements, triton.num_warps() * triton.warp_size()),)
    sqrt_tanh_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements)

    return out
