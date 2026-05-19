import torch
import triton
import triton.language as tl

@triton.jit
def sin_kernel(
    in_ptr0,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    # Compute the program ID for the current block
    pid = tl.program_id(axis=0)
    # Calculate the start index for the current block
    start = pid * BLOCK_SIZE
    # Create a mask for valid elements within the current block
    mask = start + tl.arange(0, BLOCK_SIZE) < n_elements
    # Load input data for the current block
    x = tl.load(in_ptr0 + start + tl.arange(0, BLOCK_SIZE), mask=mask)
    # Apply the sine function to the loaded data
    x = tl.sin(x)
    # Store the result back to the output pointer
    tl.store(out_ptr + start + tl.arange(0, BLOCK_SIZE), x, mask=mask)

def sin_triton(x: torch.Tensor):
    # Get the number of elements in the input tensor
    n_elements = x.numel()
    # Call the Triton kernel with the appropriate grid size and block size
    sin_kernel[(n_elements,)](x, x, n_elements, BLOCK_SIZE=4)
    # Return the modified input tensor
    return x
