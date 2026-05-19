import torch
import triton
import triton.language as tl
import math

# Kernel function: Computes the cosine of each element in the input tensor and determines the sign bit.
@triton.jit
def cos_signbit_kernel(a, b, c, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the offset for each block and thread
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Determine which elements are valid within the block
    mask = offset < n_elements
    # Load elements from tensor `a` into `a_value`
    a_value = tl.load(a + offset, mask=mask)
    # Compute the cosine of each element in `a_value`
    b_value = tl.cos(a_value.to(tl.float32))
    # Compute the sign bit of each cosine result
    c_value = b_value < 0
    # Store the cosine result back to tensor `b`
    tl.store(b + offset, b_value, mask=mask)
    # Store the sign bit result back to tensor `c`
    tl.store(c + offset, c_value, mask=mask)

# Wrapper function to invoke the Triton kernel and perform the computation
def cos_signbit(input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # Prepare output tensors `B` and `C` with the same shape and type as `input`
    B = torch.empty_like(input)
    C = torch.empty_like(input, dtype=torch.bool)
    # Determine the total number of elements in the input tensor `input`
    n_elements = input.numel()
    # Calculate the optimal block size
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Determine the grid size needed to cover all elements
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel
    cos_signbit_kernel[(grid_size, 1, 1)](input, B, C, n_elements, block_size)
    return B, C
