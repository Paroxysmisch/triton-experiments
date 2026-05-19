import triton
import triton.language as tl
import torch

# Triton kernel for softmax
@triton.jit
def _softmax(X, Y, xm_stride, xn_stride, out_stride, N, BLOCK_N: tl.constexpr):
    rm = tl.program_id(0)
    for bn in range(0, N, BLOCK_N):
        rn = bn + tl.arange(0, BLOCK_N)
        Xmn = X + rm * xm_stride + rn * xn_stride
        x = tl.load(Xmn, mask=rn < N, other=-float('inf'))
        c = tl.max(x, axis=0)
        exp_x = tl.exp(x - c)
        exp_sum = tl.sum(exp_x, axis=0)
        y = exp_x / exp_sum
        Ymn = Y + rm * out_stride
        tl.store(Ymn, y, mask=rn < N)

# Wrapper function for softmax
def softmax(input, dim, dtype=None, out=None):
    assert input.is_cuda
    *shape, N = input.shape
    input = input.view(-1, N)
    out = input.new_empty(*shape).view(-1)
    M = input.shape[0]
    _softmax[(M,)](input, out, input.stride(0), input.stride(1), out.stride(0), N, BLOCK_N=4096, num_warps=4)
    return out.view(*shape)

# Triton kernel for softmax multiplication
@triton.jit
def _softmax_mul(X, Y, Z, xm_stride, xn_stride, zm_stride, out_stride, N, BLOCK_N: tl.constexpr):
    rm = tl.program_id(0)
    for bn in range(0, N, BLOCK_N):
        rn = bn + tl.arange(0, BLOCK_N)
        Xmn = X + rm * xm_stride + rn * xn_stride
        Ymn = Y + rm * zm_stride
        x = tl.load(Xmn, mask=rn < N, other=0.0)
        y = tl.load(Ymn, mask=rn < N, other=0.0)
        z = x * y
        Zmn = Z + rm * out_stride
        tl.store(Zmn, z, mask=rn < N)

# Wrapper function for softmax multiplication
def softmax_mul(input, other, dim, dtype=None, out=None):
    assert input.is_cuda and (isinstance(other, torch.Tensor) or isinstance(other, (int, float)))
    *shape, N = input.shape
    input = input.view(-1, N)
    if isinstance(other, torch.Tensor):
        other = other.view(-1, N)
    else:
        other = torch.full_like(input, other)
    out = input.new_empty(*shape).view(-1)
    M = input.shape[0]
    _softmax_mul[(M,)](input, other, out, input.stride(0), input.stride(1), other.stride(0), out.stride(0), N, BLOCK_N=4096, num_warps=4)
    return out.view(*shape)
