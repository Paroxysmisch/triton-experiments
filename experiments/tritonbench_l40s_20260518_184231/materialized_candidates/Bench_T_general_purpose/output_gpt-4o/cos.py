import triton
import triton.language as tl
import torch

# Triton kernel for computing the cosine of each element in the input tensor
@triton.jit
def cos_kernel(
    input_ptr, output_ptr, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    # Compute the index of the current program
    pid = tl.program_id(0)
    # Create a range of offsets for the block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to handle out-of-bounds
    mask = offsets < n_elements
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask)
    # Compute the cosine
    cos_data = tl.math.cos(input_data)
    # Store the result
    tl.store(output_ptr + offsets, cos_data, mask=mask)

def cos(input, *, out=None):
    # Ensure the input is a torch tensor
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a torch.Tensor")

    # If out is provided, ensure it is a torch tensor
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a torch.Tensor if provided")

    # Determine the size of the input
    n_elements = input.numel()

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input)

    # Launch the Triton kernel
    BLOCK_SIZE = 1024  # You can tune this value based on your hardware
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    cos_kernel[grid](
        input_ptr=input.data_ptr(),
        output_ptr=out.data_ptr(),
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out

# Example usage
input_tensor = torch.tensor([0.0, 1.0, 2.0, 3.0], dtype=torch.float32)
output_tensor = cos(input_tensor)
print(output_tensor)
