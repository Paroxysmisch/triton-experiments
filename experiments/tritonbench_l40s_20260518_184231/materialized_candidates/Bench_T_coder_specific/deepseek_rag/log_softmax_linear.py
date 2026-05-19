import torch
import triton
import triton.language as tl

@triton.jit
def log_softmax_linear_kernel(
    x,
    weight,
    bias,
    out,
    D: tl.constexpr,
    B: tl.constexpr,
    HAS_BIAS: tl.constexpr
):
    i_n, i_d = tl.program_id(0), tl.program_id(1)
    o_d = i_d * B + tl.arange(0, B)
    m_d = o_d < D

    b_x = tl.load(x + i_n * D + o_d, mask=m_d, other=-float('inf'))
    b_w = tl.load(weight + o_d, mask=m_d)
    b_out = tl.dot(b_x, b_w)
    if HAS_BIAS:
        b_out = b_out + tl.load(bias + o_d, mask=m_d)
    b_max = tl.max(b_out, 0)
    b_out = tl.log(tl.sum(tl.exp(b_out - b_max), 0)) + b_max
    tl.store(out + i_n * tl.cdiv(D, B) + i_d, b_out)


def log_softmax_linear(
    input,
    weight,
    bias=None,
    dim=-1,
    dtype=None
):
    r"""
    Applies a linear transformation to the input tensor followed by the log_softmax activation function.

    Args:
        input (Tensor): The input tensor of shape `(*, in_features)`, where `*` represents any number of additional dimensions.
        weight (Tensor): The weight matrix of shape `(out_features, in_features)`.
        bias (Tensor, optional): The optional bias tensor of shape `(out_features)`. Default: None.
        dim (int): The dimension along which log_softmax will be computed. Default: -1.
        dtype (:class:`torch.dtype`, optional): The desired data type of the returned tensor. If specified, the input tensor is cast to :attr:`dtype` before the operation. Default: None.
    Returns:
        Tensor: The result of applying a linear transformation followed by a log_softmax activation function to the input tensor.
    """

    shape = input.shape
    input = input.view(-1, shape[-1])
    N, D = input.shape
    B = min(triton.next_power_of_2(D), 64 * 1024)
    ND = triton.cdiv(D, B)

    out = input.new_empty(N, ND, dtype=torch.float)
    log_softmax_linear_kernel[(N, ND)](
        x=input,
        weight=weight,
        bias=bias,
        out=out,
        D=D,
        B=B,
        HAS_BIAS=bias is not None
    )
    out = out.logsumexp(-1).view(*shape[:-1])
    if dtype is not None and dtype != torch.float:
        out = out.to(dtype)
    return out
