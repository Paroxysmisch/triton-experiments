import triton
import triton.language as tl

@triton.jit
def signbit_kernel(input_ptr, output_ptr, n_elements, **meta):
    # Get the program id and the number of programs
    pid = tl.program_id(axis=0)
    num_programs = meta['num_programs']

    # Calculate the start and end indices for this program
    block_start = pid * meta['block_size']
    block_end = tl.min(block_start + meta['block_size'], n_elements)

    # Loop over the elements of the input tensor in this block
    for i in range(block_start, block_end):
        # Load the input element
        input_val = tl.load(input_ptr + i)

        # Check if the sign bit is set
        sign_bit_set = input_val < 0 or (input_val == 0 and tl.is_negative_zero(input_val))

        # Store the result in the output tensor
        tl.store(output_ptr + i, sign_bit_set)


import torch

def signbit(input, *, out=None):
    # Ensure the input is a tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    # Determine the size of the input tensor
    n_elements = input.numel()

    # If the output tensor is not provided, allocate a new one
    if out is None:
        out = torch.empty_like(input, dtype=torch.bool)

    # Launch the Triton kernel
    block_size = 1024  # Choose an appropriate block size
    grid = (triton.cdiv(n_elements, block_size),)
    signbit_kernel[grid](input, out, n_elements, num_programs=grid[0], block_size=block_size)

    return out
