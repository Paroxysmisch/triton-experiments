import triton
import triton.language as tl
import torch
import math

@triton.jit
def exp_func(a, b, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate offset using the program ID and block size
    offset = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Ensure the offset does not exceed the number of elements
    mask = offset < n_elements
    # Load values from array 'a' using the computed offset
    a_value = tl.load(a + offset, mask=mask)
    # Compute the exponential of the loaded values
    b_value = tl.exp(a_value.to(tl.float32))
    # Store the computed values back to array 'b'
    tl.store(b + offset, b_value, mask=mask)

def exp(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Create an empty tensor 'B' with the same size and type as 'input' if 'out' is not provided
    if out is None:
        out = torch.empty_like(input)
    # Determine the total number of elements in 'input'
    n_elements = input.numel()
    # Calculate the block size based on the number of elements
    block_size = triton.next_power_of_2(math.ceil(math.sqrt(n_elements)))
    # Determine the grid size required to cover all elements
    grid_size = triton.cdiv(n_elements, block_size)
    # Launch the Triton kernel 'exp_func' with the computed grid and block size
    exp_func[(grid_size, 1, 1)](input, out, n_elements, block_size)
    return out
