import triton
import torch

@triton.jit
def chebyshev_polynomial_t_kernel(input, n, out):
    if n == 0:
        out[0] = 1
    elif n == 1:
        out[0] = input[0]
    elif n < 6 or abs(input[0]) > 1:
        T_n_minus_1 = 1
        T_n = 2 * input[0]
        for _ in range(n - 1):
            T_n_plus_1 = 2 * input[0] * T_n - T_n_minus_1
            T_n_minus_1, T_n = T_n, T_n_plus_1
        out[0] = T_n
    else:
        out[0] = torch.cos(n * torch.acos(input[0]))

def chebyshev_polynomial_t(input, n, *, out=None):
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.is_cuda, "Only CUDA tensors are supported"
        assert out.shape == input.shape, "Output shape must match input shape"
    chebyshev_polynomial_t_kernel[1, 1](input, n, out)
    return out
