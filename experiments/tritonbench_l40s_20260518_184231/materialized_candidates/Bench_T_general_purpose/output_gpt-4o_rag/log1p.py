import torch
import triton
import triton.language as tl

@triton.jit
def log1p_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Calculate the index of the current element
    idx = tl.program_id(0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to ensure we don't access out of bounds
    mask = idx < n_elements
    # Load input values
    x = tl.load(input_ptr + idx, mask=mask, other=0.0)
    # Compute log1p
    y = tl.log(1.0 + x)
    # Store the result
    tl.store(output_ptr + idx, y, mask=mask)

def log1p(input, *, out=None):
    # Ensure input is a tensor
    assert isinstance(input, torch.Tensor), "Input must be a tensor"
    # Get the number of elements
    n_elements = input.numel()
    # Determine block size
    BLOCK_SIZE = 1024  # You can adjust this value based on your needs
    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)
    # Launch the kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    log1p_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out

# Example usage:
x = torch.tensor([0.1, 0.01, 0.001], dtype=torch.float32)
y = log1p(x)
print(y)  # Should print tensor close to [0.0953, 0.00995, 0.0009995]
