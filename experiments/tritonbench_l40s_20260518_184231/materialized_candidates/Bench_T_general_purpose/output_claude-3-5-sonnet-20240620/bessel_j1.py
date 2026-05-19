import triton
import triton.language as tl
import torch

@triton.jit
def bessel_j1_kernel(input_ptr, output_ptr, n_elements):
    # Compute the Bessel function of the first kind of order 1
    idx = tl.program_id(0)
    if idx < n_elements:
        x = tl.load(input_ptr + idx)
        # Bessel function approximation (simplified for demonstration)
        result = (tl.sin(x) / x) - (tl.sin(2 * x) / (2 * x))  # Simplified Bessel J1 approximation
        tl.store(output_ptr + idx, result)

def bessel_j1(input: torch.Tensor, *, out: torch.Tensor = None) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    
    # Launch the Triton kernel
    n_elements = input.numel()
    grid = (n_elements,)
    bessel_j1_kernel[grid](input.data_ptr(), out.data_ptr(), n_elements)
    
    return out
