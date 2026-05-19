import triton
import triton.language as tl
import torch

# Triton kernel
@triton.jit
def puzzle1_kernel(
    x_ptr,
    output_ptr,
    N,
    BLOCK_SIZE: tl.constexpr,
    value
):
    # Get the program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    
    # Generate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask to handle the case where N is not divisible by BLOCK_SIZE
    mask = offsets < N
    
    # Load input data
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Perform the addition
    output = x + value
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function to call the kernel
def puzzle1(x: torch.Tensor):
    # Create an output tensor with the same shape and type as the input
    output = torch.empty_like(x)
    
    # Ensure both input and output tensors are on CUDA
    assert x.is_cuda and output.is_cuda
    
    # Get the total number of elements
    N = output.numel()
    
    # Define the grid
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    
    # Launch the kernel
    puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=1024, value=10)
    
    return output

def main():
    # Create a sample input tensor
    a = torch.tensor([4, 5, 3, 2], dtype=torch.float32)
    
    # Move the tensor to CUDA
    a = a.cuda()
    
    # Call the puzzle1 function and print the result
    result = puzzle1(a)
    print(result)

if __name__ == '__main__':
    main()
