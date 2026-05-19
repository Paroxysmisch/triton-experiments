import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_kernel(x_ptr,  # Pointer to the input vector.
                   output_ptr,  # Pointer to the output vector.
                   n_elements,  # Total number of elements in the vector.
                   BLOCK_SIZE: tl.constexpr):  # Number of elements per block.

    # Calculate the program ID in the 1D grid.
    pid = tl.program_id(axis=0)

    # Calculate the starting index for this block.
    block_start = pid * BLOCK_SIZE

    # Create offsets for each element in the block.
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    # Ensure we do not access out-of-bounds elements.
    mask = offsets < n_elements

    # Load input values.
    x = tl.load(x_ptr + offsets, mask=mask)

    # Compute the sigmoid function.
    output = 1 / (1 + tl.exp(-x))

    # Store the result.
    tl.store(output_ptr + offsets, output, mask=mask)

def sigmoid(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure the input tensor is on CUDA.
    assert input.is_cuda, "Input tensor must be on CUDA."

    # Prepare the output tensor.
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda, "Output tensor must be on CUDA."
        assert out.shape == input.shape, "Output tensor must have the same shape as input."

    # Get the number of elements.
    n_elements = input.numel()

    # Define the grid size for the kernel launch.
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']), )

    # Launch the Triton kernel.
    sigmoid_kernel[grid](input, out, n_elements, BLOCK_SIZE=1024)

    return out

# Example usage
torch.manual_seed(0)
size = 98432
x = torch.rand(size, device='cuda')
output_torch = torch.sigmoid(x)
output_triton = sigmoid(x)
print(output_torch)
print(output_triton)
print(f'The maximum difference between torch and triton is '
      f'{torch.max(torch.abs(output_torch - output_triton))}')
