import torch
import triton
import triton.language as tl

@triton.jit
def _chebyshev_polynomial_t_triton(input, n, out):
    # Convert input to float32
    input_fp32 = input.to(tl.float32)
    # Compute the Chebyshev polynomial using the trigonometric formula
    out = tl.cos(n * tl.acos(input_fp32), out=out)
    return out

def chebyshev_polynomial_t_triton(input, n, *, out=None):
    # Ensure input is a tensor
    if not isinstance(input, torch.Tensor):
        input = torch.as_tensor(input)
    # Ensure n is a tensor
    if not isinstance(n, torch.Tensor):
        n = torch.as_tensor(n)
    # Reshape n to 1D if it's not already
    if n.dim() != 1:
        n = n.reshape(-1)
    # Apply the Triton kernel
    return _chebyshev_polynomial_t_triton(input, n, out=out)
