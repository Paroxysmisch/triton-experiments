import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def mul_relu_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load elements from input and other
    x = tl.load(input_ptr + offsets, mask=mask)
    y = tl.load(other_ptr + offsets, mask=mask)

    # Perform element-wise multiplication
    z = x * y

    # Apply ReLU activation
    z = tl.maximum(z, 0.0)

    # Store the result
    tl.store(output_ptr + offsets, z, mask=mask)
