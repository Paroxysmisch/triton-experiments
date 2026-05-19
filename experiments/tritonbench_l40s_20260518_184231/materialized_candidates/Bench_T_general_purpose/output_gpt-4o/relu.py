import triton
import triton.language as tl
import torch

# Triton kernel for ReLU operation
@triton.jit
def relu_kernel(input_ptr, output_ptr, n_elements, inplace: tl.constexpr):
    # Obtain the current index within the grid
    pid = tl.program_id(axis=0)
    # Compute the range of elements this program will handle
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Ensure we don't go out of bounds
    mask = offsets < n_elements

    # Load the input values
    x = tl.load(input_ptr + offsets, mask=mask)

    # Apply the ReLU operation
    y = tl.where(x > 0, x, 0)

    # Store the result
    if inplace:
        tl.store(input_ptr + offsets, y, mask=mask)
    else:
        tl.store(output_ptr + offsets, y, mask=mask)

# Wrapper function for ReLU operation
def relu(input, inplace=False):
    # Ensure input is a Torch tensor
    assert isinstance(input, torch.Tensor), "Input must be a torch.Tensor"

    # Get the number of elements in the input tensor
    n_elements = input.numel()

    # Create an output tensor if not inplace
    if not inplace:
        output = torch.empty_like(input)
    else:
        output = input

    # Launch the Triton kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    relu_kernel[grid](input, output, n_elements, inplace=inplace, BLOCK_SIZE=1024)

    return output

# Example usage
x = torch.tensor([-1.0, 0.0, 1.0, 2.0, -3.0], device='cuda')
y = relu(x)  # Out-of-place operation
print(y)  # Output: tensor([0., 0., 1., 2., 0.], device='cuda')

z = relu(x, inplace=True)  # In-place operation
print(z)  # Output: tensor([0., 0., 1., 2., 0.], device='cuda')
