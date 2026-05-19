import triton
import triton.language as tl

# Define the Triton kernel
@triton.jit
def puzzle1_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr, value: tl.constexpr):
    # Program ID for the current block
    pid = tl.program_id(axis=0)
    
    # Calculate the start index for this block
    block_start = pid * BLOCK_SIZE
    
    # Create offsets within this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to ensure we only process valid indices
    mask = offsets < N
    
    # Load data from the input pointer with masking
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Perform the element-wise addition
    result = x + value
    
    # Store the result back to the output pointer with masking
    tl.store(output_ptr + offsets, result, mask=mask)

# Define the wrapper function
def puzzle1(x, BLOCK_SIZE=1024, value=10):
    # Ensure input tensor is compatible with CUDA
    assert x.is_cuda, "Input tensor must be on CUDA device"
    
    # Calculate the total number of elements
    N = x.numel()
    
    # Create an output tensor with the same shape and type as input
    output = torch.empty_like(x)
    
    # Define the grid size
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    # Launch the Triton kernel
    puzzle1_kernel[grid](x, output, N, BLOCK_SIZE, value)
    
    # Return the output tensor
    return output
