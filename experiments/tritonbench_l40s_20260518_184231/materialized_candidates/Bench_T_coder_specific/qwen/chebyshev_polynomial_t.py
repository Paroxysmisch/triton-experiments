import torch
import triton
import triton.language as tl

# Define the Triton kernel function
chebyshev_polynomial_kernel = triton.compile(
    chebyshev_polynomial_kernel,
    constants={'N': None},
    num_warps=4,
    num_stages=3,
)

def chebyshev_polynomial_t(input, n, out=None):
    if out is None:
        out = torch.empty_like(input)

    # Ensure input and n are tensors
    input = input.contiguous()
    n = n.contiguous()

    # Check if input is on GPU
    if input.device.type == 'cuda':
        # Launch the Triton kernel
        grid_size = (input.numel() + 1023) // 1024
        block_size = 1024
        chebyshev_polynomial_kernel[grid_size, block_size](input.data_ptr(), n.data_ptr(), out.data_ptr(), input.numel())
    else:
        raise ValueError("Input must be on GPU")

    return out
