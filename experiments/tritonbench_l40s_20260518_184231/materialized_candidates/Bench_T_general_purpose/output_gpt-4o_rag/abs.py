import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def abs_kernel(
        input_ptr,
        output_ptr,
        N,
        BLOCK_SIZE: tl.constexpr
):
    # Get the index for the current program
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask to handle out-of-bounds accesses
    mask = idx < N

    # Load elements from input_ptr with masking
    input_val = tl.load(input_ptr + idx, mask=mask)
    # Compute the absolute value
    abs_val = tl.abs(input_val)
    # Store the result in output_ptr with masking
    tl.store(output_ptr + idx, abs_val, mask=mask)


def abs(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Flatten the input tensor to a 1D array for simplicity
    input_flat = input.flatten()
    N = input_flat.numel()

    # Prepare an output tensor if not provided
    if out is None:
        out = torch.empty_like(input_flat)
    else:
        assert out.numel() == N, "Output tensor must have the same number of elements as the input"

    # Ensure both input and output tensors are on the GPU
    assert input.is_cuda and out.is_cuda, 'Both input and output tensors must be on GPU'

    # Define block size and grid size
    BLOCK_SIZE = 1024  # You can choose an appropriate block size
    grid = (triton.cdiv(N, BLOCK_SIZE),)

    # Launch the Triton kernel
    abs_kernel[grid](
        input_ptr=input_flat,
        output_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape the output to match the input shape
    return out.view_as(input)

# Example usage:
# input_tensor = torch.tensor([-1.0, 2.0, -3.0], device=device)
# output_tensor = abs(input_tensor)
# print(output_tensor)  # Should print tensor([1.0, 2.0, 3.0], device='cuda:0')
