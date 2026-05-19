import logging
import triton
import triton.language as tl

# Triton kernel for multiplying elements by 2
@triton.jit
def mul2_kernel(
    output_ptr,  # Pointer to the output tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr  # Size of each block
):
    pid = tl.program_id(0)  # Get the program ID (block index)
    block_start = pid * BLOCK_SIZE  # Calculate the starting index for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Calculate offsets within the block
    mask = offsets < n_elements  # Ensure we stay within n_elements
    x = tl.load(offsets, mask=mask) * 2  # Load, multiply, and store
    tl.store(output_ptr + offsets, x, mask=mask)

# Function to call the Triton kernel
def triton_mul2(x):
    output = torch.empty_like(x)  # Initialize an output tensor
    n_elements = output.numel()  # Get the total number of elements
    BLOCK_SIZE = 1024  # Set the block size
    grid_size = triton.cdiv(  # Calculate the grid size
        n_elements, BLOCK_SIZE)
    logging.debug("triton mul2 grid %d, %d", grid_size, BLOCK_SIZE)  # Log grid dimensions
    mul2_kernel[(grid_size,)](  # Call the Triton kernel
        output,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output  # Return the result

# Triton kernel for in-place multiplication by 2
@triton.jit
def mul2_inplace_kernel(
    input_ptr,   # Pointer to the input tensor
    n_elements,  # Total number of elements
    BLOCK_SIZE: tl.constexpr  # Size of each block
):
    pid = tl.program_id(0)  # Get the program ID (block index)
    block_start = pid * BLOCK_SIZE  # Calculate the starting index for the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Calculate offsets within the block
    mask = offsets < n_elements  # Ensure we stay within n_elements
    x = tl.load(input_ptr + offsets, mask=mask) * 2  # Load, multiply, and store
    tl.store(input_ptr + offsets, x, mask=mask)

# Function to call the Triton kernel
def triton_mul2_inplace(x):
    n_elements = x.numel()  # Get the total number of elements
    BLOCK_SIZE = 1024  # Set the block size
    grid_size = triton.cdiv(  # Calculate the grid size
        n_elements, BLOCK_SIZE)
    logging.debug("triton mul2 inplace grid %d, %d", grid_size, BLOCK_SIZE)  # Log grid dimensions
    mul2_inplace_kernel[(grid_size,)](  # Call the Triton kernel
        x,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return x  # Return the result
