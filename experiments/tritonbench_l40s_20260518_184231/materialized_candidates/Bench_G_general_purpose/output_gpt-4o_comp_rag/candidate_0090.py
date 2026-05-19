import triton
import triton.language as tl

@triton.jit
def puzzle1_kernel(x_ptr, output_ptr, N, BLOCK_SIZE: tl.constexpr, value):
    pid = tl.program_id(axis=0)  # Get the unique program ID along axis 0
    block_start = pid * BLOCK_SIZE  # Calculate the starting index for this block
    offsets = block_start + tl.arange(0, BLOCK_SIZE)  # Compute the offsets within the block
    mask = offsets < N  # Ensure offsets are within valid range
    x = tl.load(x_ptr + offsets, mask=mask)  # Load data from input tensor
    output = x + value  # Add the constant value to each element
    tl.store(output_ptr + offsets, output, mask=mask)  # Store the result in the output tensor

import torch

def puzzle1(x: torch.Tensor):
    output = torch.empty_like(x)  # Create an output tensor with the same shape and type as x
    assert x.is_cuda and output.is_cuda  # Ensure tensors are on the GPU
    N = output.numel()  # Get the total number of elements
    grid = lambda meta: (triton.cdiv(N, meta['BLOCK_SIZE']),)  # Define grid size
    puzzle1_kernel[grid](x, output, N, BLOCK_SIZE=1024, value=10)  # Launch the kernel
    return output  # Return the output tensor

def main():
    a = torch.Tensor([4, 5, 3, 2])
    a = a.to(device=torch.device('cuda'))  # Move the tensor to the GPU
    print(puzzle1(a))  # Call the wrapper function and print the result

if __name__ == '__main__':
    main()
