import triton
import triton.language as tl

@triton.jit
def cos_func(a_ptr, b_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Compute the starting index of the block
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Compute the offsets for the current block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load the input elements
    a = tl.load(a_ptr + offsets, mask=mask)

    # Compute the cosine of the input elements
    b = tl.cos(a)

    # Store the results back to the output tensor
    tl.store(b_ptr + offsets, b, mask=mask)

import torch

def cos(A: torch.Tensor) -> torch.Tensor:
    # Ensure the input tensor is on the GPU
    assert A.is_cuda, "Input tensor must be on the GPU"

    # Create the output tensor with the same shape and type as the input tensor
    B = torch.empty_like(A)

    # Get the number of elements in the tensor
    n_elements = A.numel()

    # Define the block size
    BLOCK_SIZE = 256

    # Calculate the number of blocks needed
    grid_size = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE

    # Launch the kernel
    cos_func[grid_size, BLOCK_SIZE](A, B, n_elements, BLOCK_SIZE)

    return B
