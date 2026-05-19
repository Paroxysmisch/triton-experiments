import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to first input vector.
    y_ptr,  # *Pointer* to second input vector.
    output_ptr,  # *Pointer* to output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each program should process.
):
    # Determine the block index using a 1D launch grid
    pid = tl.program_id(axis=0)
    # Calculate the starting index of the block
    block_start = pid * BLOCK_SIZE
    # Compute the offsets for elements within this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = offsets < n_elements
    # Load elements from the input vectors with the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    # Compute the element-wise sum
    output = x + y
    # Store the result in the output vector with the mask
    tl.store(output_ptr + offsets, output, mask=mask)

def add(x: torch.Tensor, y: torch.Tensor):
    # Prepare an output tensor with the same shape and type as the input tensors
    output = torch.empty_like(x)
    # Ensure all tensors are on the CUDA device
    assert x.is_cuda and y.is_cuda and output.is_cuda
    # Calculate the total number of elements in the output tensor
    n_elements = output.numel()
    # Define the grid size for launching the kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    # Launch the kernel with the computed grid size
    add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=1024)
    # Return the output tensor containing the summed elements
    return output
