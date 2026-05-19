import triton
import torch

@triton.jit
def chebyshev_polynomial_t_kernel(input, n, out):
    if n == 0:
        return torch.full_like(out, 1)
    if n == 1:
        return torch.full_like(out, input)
    if n < 6 or torch.abs(input) > 1:
        T_n_minus_1 = torch.full_like(out, 1)
        T_n = input
        for _ in range(2, n + 1):
            T_n_plus_1 = 2 * input * T_n - T_n_minus_1
            T_n_minus_1 = T_n
            T_n = T_n_plus_1
        return T_n
    else:
        x_acos = torch.acos(input)
        return torch.cos(n * x_acos)

def chebyshev_polynomial_t(input, n, *, out=None):
    if not isinstance(input, torch.Tensor):
        raise TypeError("input must be a Tensor")
    if not isinstance(n, torch.Tensor):
        raise TypeError("n must be a Tensor")
    if out is not None and not isinstance(out, torch.Tensor):
        raise TypeError("out must be a Tensor")
    if out is None:
        out = torch.empty_like(input)
    if input.ndim == 0:
        input = input.unsqueeze(0)
    if n.ndim == 0:
        n = n.unsqueeze(0)
    return chebyshev_polynomial_t_kernel(input, n, out)
