import torch
import triton
import triton.language as tl
from torch import Tensor
from flag_gems.utils.shape_utils import volume

@triton.jit
def gelu_cdf(x):
    # Using approximation introduces greater parity errors.
    return tl.sigmoid(1.702 * x) * (1 + 0.044715 * x)

@triton.jit
def gelu_approx(x):
    return 0.5 * x * (1 + tl.tanh(0.79788456 * x * (1 + 0.044715 * x * x)))

@triton.jit
def std_func(X, Y, M, n_elements, N, MISSING_DIM, BLOCK_N: tl.constexpr, approximate: tl.constexpr, correction: tl.constexpr):
    # Compute mean
    if MISSING_DIM == 0:
        M = tl.sum(X) / n_elements
        X = X - M
    elif MISSING_DIM == 1:
        M = tl.sum(X) / n_elements
        X = X - M
    else:
        M = tl.sum(X) / n_elements
        X = X - M

    # Compute variance
    if MISSING_DIM == 0:
        var = tl.sum(tl.where(Y == 1, X * X, 0)) / (N - correction)
    elif MISSING_DIM == 1:
        var = tl.sum(tl.where(Y == 1, X * X, 0)) / (N - correction)
    else:
        var = tl.sum(tl.where(Y == 1, X * X, 0)) / (N - correction)

    # Return std
    return tl.sqrt(var)

def gelu_std(input: Tensor, dim=None, keepdim=False, correction=1, approximate='none', out=None) -> Tensor:
    assert approximate in ['none', 'tanh']
    assert input.is_contiguous()

    if dim is None or len(dim) == input.ndim:
        # Compute GELU over all elements
        if approximate == 'none':
            input = gelu_cdf(input)
        else:
            input = gelu_approx(input)
        n_elements = input.numel()
        input = input.reshape(-1)
        std = std_func(input, None, n_elements, n_elements, 1, 0, 1, approximate, correction)
        if not keepdim:
            input = input.reshape(input.shape[:-1] + (1,) * input.ndim)
        return std
    else:
        # Compute GELU over specified dimensions and then std
        dim = dim if isinstance(dim, tuple) else (dim,)
        input_gelu = input.clone()
        for d in dim:
            assert d >= -input.ndim and d < input.ndim, "Invalid dim"
        for d in dim:
            input_gelu = input_gelu.squeeze(d)
        if approximate == 'none':
            input_gelu = gelu_cdf(input_gelu)
        else:
            input_gelu = gelu_approx(input_gelu)
        std = torch.std(input_gelu, dim=dim, unbiased=True, keepdim=keepdim, correction=correction)
        return std
