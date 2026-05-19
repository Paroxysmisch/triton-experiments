import torch
import triton
import triton.language as tl
from scipy.special import gammaln as scipy_gammaln

@triton.jit
def gammaln_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the program ID for parallelization
    pid = tl.program_id(0)
    # Compute the range of elements this program instance will process
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Load the input data into SRAM, using a mask to avoid out-of-bounds accesses
    mask = offsets < n_elements
    input_data = tl.load(input_ptr + offsets, mask=mask)
    # Compute the natural logarithm of the absolute value of the gamma function
    result = tl.libdevice.lgamma(tl.abs(input_data))
    # Store the result back to the output pointer
    tl.store(output_ptr + offsets, result, mask=mask)

def gammaln(input, *, out=None):
    # Ensure the input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise ValueError("Input must be a torch.Tensor")

    # Flatten the input tensor to process it as a 1D array
    input_flat = input.flatten()
    n_elements = input_flat.numel()

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input_flat)
    else:
        out = out.flatten()

    # Define the block size for Triton
    BLOCK_SIZE = 1024  # This can be tuned for performance

    # Launch the Triton kernel
    gammaln_kernel[(n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE](
        input_flat,
        out,
        n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape the output tensor to match the input tensor's shape
    return out.view_as(input)

# Example usage
x = torch.tensor([0.5, 1.0, 2.5, 3.0], dtype=torch.float32)
result = gammaln(x)
print(result)
