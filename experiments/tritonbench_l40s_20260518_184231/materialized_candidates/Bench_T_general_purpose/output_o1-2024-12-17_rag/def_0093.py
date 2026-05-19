import triton
import triton.language as tl
import torch

# ----------------------
# Triton kernel for streaming logsumexp along the last dimension
# ----------------------
@triton.jit
def _logsumexp_kernel(
    X, OUT, xm_stride, xn_stride, out_stride, N, BLOCK_N: tl.constexpr
):
    rm = tl.program_id(0)
    alpha = tl.zeros((1,), tl.float32) + -float("inf")
    res = tl.zeros((1,), tl.float32)

    for bn in range(0, N, BLOCK_N):
        rn = bn + tl.arange(0, BLOCK_N)
        Xmn = X + rm * xm_stride + rn * xn_stride
        x = tl.load(Xmn, mask=rn < N, other=-float("inf"))
        c = tl.max(x, axis=0)
        # adjust the partial sums for numerical stability
        res = tl.where(c > alpha, res * tl.exp(alpha - c), res)
        alpha = tl.where(c > alpha, c, alpha)
        res += tl.sum(tl.exp(x - alpha), axis=0)

    out_val = tl.log(res) + alpha
    OUT_ptr = OUT + rm * out_stride
    tl.store(OUT_ptr, out_val)


def _logsumexp(x: torch.Tensor) -> torch.Tensor:
    """
    Computes logsumexp along the last dimension of x using the above Triton kernel.
    """
    assert x.is_cuda, "Input must be a CUDA tensor."

    # reshape so that the reduced dimension is the last
    *leading_dims, N = x.shape
    x_2d = x.view(-1, N)
    out = x.new_empty(*leading_dims).view(-1)

    M = x_2d.shape[0]
    # launch kernel
    _logsumexp_kernel[(M,)](
        x_2d,
        out,
        x_2d.stride(0),
        x_2d.stride(1),
        out.stride(0),
        N,
        BLOCK_N=4096,
        num_warps=4,
    )
    return out.view(*leading_dims)


def _softmax_lastdim(x: torch.Tensor) -> torch.Tensor:
    """
    Applies softmax along the last dimension using logsumexp for numerical stability.
    Expects x to be a CUDA tensor.
    """
    # first compute logsumexp and subtract
    c = _logsumexp(x)
    # subtract broadcasted logsumexp from x along last dim, then exponentiate
    return x.sub_(c[..., None]).exp_()


def _softmax_triton(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Applies Triton-based softmax along the specified dimension. Falls back to
    PyTorch softmax if x is not on CUDA or if dim is not the last dimension,
    by rearranging dims to make 'dim' the last one during the kernel call.
    """
    # If not on CUDA, revert to PyTorch implementation
    if not x.is_cuda:
        return x.softmax(dim=dim)

    # normalize dim
    ndims = x.dim()
    if dim < 0:
        dim += ndims

    # if 'dim' is already the last dimension, we can apply directly
    if dim == ndims - 1:
        return _softmax_lastdim(x)

    # otherwise, transpose 'dim' to be the last dimension, apply softmax, and transpose back
    permute_dims = list(range(ndims))
    permute_dims[-1], permute_dims[dim] = permute_dims[dim], permute_dims[-1]
    x_perm = x.permute(*permute_dims)
    x_out = _softmax_lastdim(x_perm)
    # invert the permutation
    inv_permute = list(range(ndims))
    inv_permute[dim], inv_permute[-1] = inv_permute[-1], inv_permute[dim]
    return x_out.permute(*inv_permute)


def softmax_log(input: torch.Tensor, dim: int = -1, dtype=None) -> torch.Tensor:
    """
    Applies the natural logarithm element-wise on the input tensor, followed by the
    softmax function along the specified dimension. This operation results in
    values scaled between 0 and 1, summing to 1 after the logarithmic transformation.

    Args:
        input (Tensor): The input tensor on which logarithm and softmax are applied.
        dim (int): The dimension along which softmax will be computed. Default: -1.
        dtype (torch.dtype, optional): The desired data type of the returned tensor.
                                       If specified, the input tensor is cast to this
                                       dtype before the operation is performed. Useful
                                       for preventing data type overflows. Default: None.

    Returns:
        Tensor: A tensor containing the result of softmax(log(input)) along the specified dimension.
    """
    # cast to specified dtype if provided
    if dtype is not None and input.dtype != dtype:
        input = input.to(dtype)

    # take logarithm of input
    logged_input = input.log()

    # apply softmax along the specified dimension
    result = _softmax_triton(logged_input, dim=dim)

    return result
