import torch
import triton
import triton.language as tl


@triton.jit
def bessel_j1_kernel(
    input_ptr,
    output_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Load input values
    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)

    # Polynomial approximation of J1(x) for demonstration
    # J1(x) ≈ x/2 - x^3/16 + x^5/384 - x^7/18432
    x2 = x * x
    x3 = x2 * x
    x5 = x3 * x2
    x7 = x5 * x2
    y = 0.5 * x - (x3 / 16.0) + (x5 / 384.0) - (x7 / 18432.0)

    # Store result
    tl.store(output_ptr + offsets, y, mask=mask)


def bessel_j1(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    # Allocate output if none is provided
    if out is None:
        out = torch.empty_like(input)

    # Ensure input and output are on CUDA
    assert input.is_cuda, "Input must be on CUDA"
    assert out.is_cuda, "Output must be on CUDA"

    # Flatten to 1D for kernel processing
    n_elements = input.numel()
    input_flat = input.view(-1)
    out_flat = out.view(-1)

    # Define block size and grid
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch the Triton kernel
    bessel_j1_kernel[grid](
        input_ptr=input_flat,
        output_ptr=out_flat,
        n_elements=n_elements,
        BLOCK_SIZE=BLOCK_SIZE
    )

    # Reshape to original shape if needed
    return out
