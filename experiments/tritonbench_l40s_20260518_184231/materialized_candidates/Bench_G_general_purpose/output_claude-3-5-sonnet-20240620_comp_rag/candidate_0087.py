import triton
import triton.language as tl
import torch

@triton.jit
def puzzle1_kernel(
    x_ptr,          # Pointer to input tensor
    output_ptr,     # Pointer to output tensor
    N,              # Total number of elements
    BLOCK_SIZE: tl.constexpr,  # Block size as compile-time constant
    value,          # Constant value to add
):
    # Get the program ID for the current thread block
    pid = tl.program_id(axis=0)
    
    # Calculate starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for each thread within the block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create mask for valid indices (within bounds of N)
    mask = offsets < N
    
    # Load input values using the calculated offsets
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Add the constant value to create output
    output = x + value
    
    # Store results back to memory
    tl.store(output_ptr + offsets, output, mask=mask)

def puzzle1(x: torch.Tensor):
    # Input validation
    assert x.is_cuda, "Input tensor must be on GPU"
    
    # Create output tensor with same properties as input
    output = torch.empty_like(x)
    
    # Get total number of elements
    N = output.numel()
    
    # Define grid size based on block size and total elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    # Launch kernel with specified parameters
    puzzle1_kernel[grid](
        x_ptr=x,
        output_ptr=output,
        N=N,
        BLOCK_SIZE=1024,  # Standard block size for good occupancy
        value=10,         # Constant value to add
    )
    
    return output

def main():
    # Create test tensor and move to GPU
    a = torch.tensor([4, 5, 3, 2], dtype=torch.float32, device='cuda')
    
    # Run kernel and print results
    result = puzzle1(a)
    print(f"Input: {a}")
    print(f"Output: {result}")

if __name__ == '__main__':
    main()
