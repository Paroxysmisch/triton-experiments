import torch
import triton
import triton.language as tl

# Triton kernel for vector addition
@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program ID (block index)
    blockIdx = tl.program_id(axis=0)

    # Calculate the offsets for the block
    offsets = blockIdx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)

    # Create a mask to handle boundary conditions
    mask = offsets < n_elements

    # Load the values from the input vectors
    x_values = tl.load(x_ptr + offsets, mask=mask)
    y_values = tl.load(y_ptr + offsets, mask=mask)

    # Compute the element-wise sum
    output_values = x_values + y_values

    # Store the result in the output vector
    tl.store(output_ptr + offsets, output_values, mask=mask)

# Launcher function to set up and launch the kernel
def vector_add_launcher(x: torch.Tensor, y: torch.Tensor, GPU_ID: int, b_size=1024):
    # Create an empty tensor for the output
    output = torch.empty_like(x).to(GPU_ID)

    # Ensure the input tensors have the same shape
    assert x.shape == y.shape, "Shape incorrect"

    # Get the number of elements in the vectors
    elements = x.numel()

    # Ensure the tensors are on the GPU
    assert x.is_cuda and y.is_cuda and output.is_cuda, "Tensors must be on GPU."

    # Define the grid configuration
    grid = lambda meta: (triton.cdiv(elements, meta['BLOCK_SIZE']), )

    # Launch the kernel
    compiled_func = vector_add_kernel[grid](x, y, output, elements, BLOCK_SIZE=b_size)
    
    return output

# Example usage
BLOCK_SIZE = 1024
size = int(1e5)
x = torch.rand(size, device='cuda')
y = torch.rand(size, device='cuda')
torch.cuda.synchronize()

# Perform the vector addition
out = vector_add_launcher(x, y, 0, BLOCK_SIZE)

# Synchronize to ensure the kernel has finished
torch.cuda.synchronize()

# Print the result (optional)
print(out)
