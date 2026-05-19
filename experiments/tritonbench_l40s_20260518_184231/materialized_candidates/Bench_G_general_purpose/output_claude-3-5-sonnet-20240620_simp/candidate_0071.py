import torch
import triton
import triton.language as tl

# Triton kernel for computing element-wise square
@triton.jit
def square_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    n_cols,     # Number of columns in the tensor
    stride,     # Stride for the tensor
    BLOCK_SIZE: tl.constexpr,  # Static block size for optimization
):
    # Get the program ID for parallel execution
    pid = tl.program_id(axis=0)
    
    # Calculate the starting offset for this program instance
    offset = pid * stride
    
    # Create a range of indices for this block
    col_offsets = tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid memory accesses
    mask = col_offsets < n_cols
    
    # Load input data
    x = tl.load(input_ptr + offset + col_offsets, mask=mask)
    
    # Compute square
    output = x * x
    
    # Store the result
    tl.store(output_ptr + offset + col_offsets, output, mask=mask)

# Wrapper function for the kernel
def square(input_tensor):
    # Input validation
    assert input_tensor.dim() == 2, "Input must be a 2D tensor"
    assert input_tensor.is_cuda, "Input must be a CUDA tensor"
    
    # Get tensor dimensions
    n_rows, n_cols = input_tensor.shape
    
    # Create output tensor
    output = torch.empty_like(input_tensor)
    
    # Calculate stride
    stride = input_tensor.stride(0)
    
    # Determine block size (round up to nearest multiple of 32 for efficiency)
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)  # Cap at 1024 for hardware limits
    
    # Determine number of warps based on block size
    num_warps = 4
    if BLOCK_SIZE >= 512:
        num_warps = 8
    elif BLOCK_SIZE >= 256:
        num_warps = 4
    else:
        num_warps = 2
        
    # Launch kernel
    grid = (n_rows,)  # One program per row
    square_kernel[grid](
        input_ptr=input_tensor,
        output_ptr=output,
        n_cols=n_cols,
        stride=stride,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    
    return output
