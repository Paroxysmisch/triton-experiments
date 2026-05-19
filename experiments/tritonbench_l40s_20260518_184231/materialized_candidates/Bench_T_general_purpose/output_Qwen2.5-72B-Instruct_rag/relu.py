import torch
import triton
import triton.language as tl

# Triton kernel for ReLU
@triton.jit
def relu_kernel(output_ptr, input_ptr, size, BLOCK_SIZE: tl.constexpr):
    # Compute the program ID in a 1D grid
    pid = tl.program_id(0)
    # Compute the range of elements this program should handle
    block_start = pid * BLOCK_SIZE
    # Create a range of offsets for the elements this program will process
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to guard memory operations against out-of-bounds accesses
    mask = offsets < size
    # Load the input elements into SRAM
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    # Apply the ReLU function
    y = tl.where(x > 0, x, 0.0)
    # Store the output elements back to DRAM
    tl.store(output_ptr + offsets, y, mask=mask)

# Python wrapper function for ReLU
def relu(input, inplace=False):
    # Ensure the input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("Input must be a torch.Tensor")
    
    # Determine the size of the input tensor
    size = input.numel()
    
    # Determine the block size (a power of two)
    BLOCK_SIZE = triton.next_power_of_2(min(1024, size))
    
    # Determine the number of programs (blocks) needed
    num_programs = (size + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Prepare the output tensor
    if inplace:
        output = input
    else:
        output = torch.empty_like(input)
    
    # Call the Triton kernel
    relu_kernel[(num_programs,)](output, input, size, BLOCK_SIZE=BLOCK_SIZE)
    
    return output
