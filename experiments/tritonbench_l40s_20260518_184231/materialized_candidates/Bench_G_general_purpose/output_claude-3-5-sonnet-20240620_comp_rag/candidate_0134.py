import torch
import triton
import triton.language as tl

@triton.jit
def add_kernel(
    in_ptr0,  # pointer to first input tensor
    in_ptr1,  # pointer to second input tensor
    out_ptr,  # pointer to output tensor
    n_elements,  # number of elements in the tensor
    BLOCK_SIZE: tl.constexpr,  # size of the block for processing
):
    # Program ID in the block dimension
    pid = tl.program_id(0)
    
    # Calculate starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offset range for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask to handle boundary conditions
    mask = offsets < n_elements
    
    # Load input values using the mask
    x = tl.load(in_ptr0 + offsets, mask=mask)
    y = tl.load(in_ptr1 + offsets, mask=mask)
    
    # Perform element-wise addition
    output = x + y
    
    # Store the result
    tl.store(out_ptr + offsets, output, mask=mask)

def add_wrapper(x: torch.Tensor, y: torch.Tensor):
    # Ensure inputs are on GPU
    assert x.is_cuda and y.is_cuda
    
    # Ensure inputs have same shape
    assert x.shape == y.shape
    
    # Create output tensor
    out = torch.zeros_like(x)
    
    # Calculate number of elements
    n_elements = x.numel()
    
    # Define block size
    BLOCK_SIZE = 1024
    
    # Calculate grid size
    num_blocks = (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE
    
    # Launch kernel
    add_kernel[(num_blocks,)](
        x.data_ptr(),
        y.data_ptr(),
        out.data_ptr(),
        n_elements,
        BLOCK_SIZE,
    )
    
    return out
