import triton
import triton.language as tl
import torch

# Triton kernel
@triton.jit
def puzzle1_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr, value):
    # Get the program ID which identifies the block of work
    pid = tl.program_id(axis=0)
    # Calculate the starting index for this block
    block_start = pid * BLOCK_SIZE
    # Generate offsets for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to ensure we don't read/write out of bounds
    mask = offsets < N
    # Load input data with the mask applied
    x = tl.load(x_ptr + offsets, mask=mask)
    # Perform the addition with the constant value
    output = x + value
    # Store the result back to the output tensor
    tl.store(output_ptr + offsets, output, mask=mask)

# Wrapper function to call the kernel
def puzzle1(x: torch.Tensor):
    # Create an output tensor with the same shape and type as the input
    output = torch.empty_like(x)
    # Ensure the input and output tensors are on the GPU
    assert x.is_cuda and output.is_cuda
    # Get the total number of elements in the tensor
    N = output.numel()
    # Define the grid size for the kernel launch
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)
    # Launch the Triton kernel with specified grid and block size
    puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=1024, value=10)
    # Return the output tensor
    return output

def main():
    # Create a tensor and move it to the GPU
    a = torch.Tensor([4, 5, 3, 2]).to(device=torch.device('cuda'))
    # Call the wrapper function and print the result
    print(puzzle1(a))

if __name__ == '__main__':
    main()
