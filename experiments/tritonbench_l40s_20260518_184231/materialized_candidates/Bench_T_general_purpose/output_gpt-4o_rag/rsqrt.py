import torch
import triton
import triton.language as tl

@triton.jit
def rsqrt_kernel(
        input_ptr,
        output_ptr,
        N,
        BLOCK_SIZE: tl.constexpr
):
    # Get the index for the current program
    idx = tl.program_id(axis=0)
    # Compute the offset for each element in the input tensor
    offsets = idx * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Create a mask for elements within bounds
    mask = offsets < N

    # Load the input values, and handle out-of-bounds with a default value of 0
    input_vals = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Compute reciprocal of square root
    rsqrt_vals = tl.where(input_vals > 0, 1.0 / tl.sqrt(input_vals), float('nan'))

    # Store the result in the output tensor with masking
    tl.store(output_ptr + offsets, rsqrt_vals, mask=mask)

def rsqrt(input: torch.Tensor, out: torch.Tensor = None) -> torch.Tensor:
    # Ensure the input tensor is on the correct device
    device = input.device
    N = input.numel()  # Total number of elements

    # If no output tensor is provided, create a new one
    if out is None:
        out = torch.empty_like(input, device=device)

    # Block size (should be a power of 2 for optimal performance)
    BLOCK_SIZE = 1024  # Or use triton.next_power_of_2(N) if dynamic block size is needed

    # Launch the Triton kernel with the required grid size
    grid = (N + BLOCK_SIZE - 1) // BLOCK_SIZE  # Compute grid size based on total elements
    rsqrt_kernel[(grid,)](
        input_ptr=input,
        output_ptr=out,
        N=N,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out
