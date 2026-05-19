import triton
import triton.language as tl
import torch

# Triton kernel for streaming logsumexp
@triton.jit
def _logsumexp(X, OUT, xm_stride, xn_stride, out_stride, N, BLOCK_N: tl.constexpr):
    rm = tl.program_id(0)
    alpha = tl.zeros((1,), tl.float32) + -float('inf')
    res = tl.zeros((1,), tl.float32)
    for bn in range(0, N, BLOCK_N):
        rn = bn + tl.arange(0, BLOCK_N)
        Xmn = X + rm * xm_stride + rn * xn_stride
        x = tl.load(Xmn, mask=rn < N, other=-float('inf'))
        c = tl.max(x, axis=0)
        # correct the current sum and update the max
        res = tl.where(c > alpha, res * tl.exp(alpha - c), res)
        alpha = tl.where(c > alpha, c, alpha)
        res += tl.sum(tl.exp(x - alpha), axis=0)
    out = tl.log(res) + alpha
    rm = tl.program_id(0) + tl.arange(0, 1)
    OUT = OUT + rm * out_stride
    tl.store(OUT, out)

# Function to call the Triton kernel
def logsumexp(input):
    assert input.is_cuda
    *dims, N = input.shape
    input = input.view(-1, N)
    out = input.new_empty(*dims).view(-1)
    M = input.shape[0]
    _logsumexp[(M,)](input, out, input.stride(0), input.stride(1), out.stride(0), N,
                     BLOCK_N=4096, num_warps=4)
    return out.view(*dims)

# Softmax function using the logsumexp Triton kernel
def softmax_(x):
    if not x.is_cuda:
        return torch.softmax(x, dim=-1, out=x)
    c = logsumexp(x)
    return x.sub_(c[..., None]).exp_()
