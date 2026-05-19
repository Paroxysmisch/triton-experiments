import triton
import triton.language as tl
import torch
from torch.autograd import Function

@triton.jit
def _logsumexp_kernel(X, OUT, xm_stride, xn_stride, out_stride, N, BLOCK_N: tl.constexpr):
    rm = tl.program_id(0)
    alpha = tl.zeros((1,), tl.float32) + -float('inf')
    res = tl.zeros((1,), tl.float32)
    for bn in range(0, N, BLOCK_N):
        rn = bn + tl.arange(0, BLOCK_N)
        X_ptr = X + rm * xm_stride + rn * xn_stride
        x = tl.load(X_ptr, mask=rn < N, other=-float('inf'))
        c = tl.max(x, axis=0)
        res = tl.where(c > alpha, res * tl.exp(alpha - c), res)
        alpha = tl.where(c > alpha, c, alpha)
        res += tl.sum(tl.exp(x - alpha), axis=0)
    out = tl.log(res) + alpha
    OUT_ptr = OUT + rm * out_stride
    tl.store(OUT_ptr, out)

def logsumexp(input, dim):
    assert input.is_cuda, "Input tensor must be on CUDA"
    if dim is None:
        input = input.flatten()
        dim = 0
    ndim = input.dim()
    if dim < 0:
        dim += ndim
    perm = list(range(ndim))
    perm.remove(dim)
    perm.append(dim)
    input_perm = input.permute(perm)
    *dims, N = input_perm.shape
    input_2d = input_perm.reshape(-1, N)
    out = input_2d.new_empty(*dims).view(-1)
    M = input_2d.shape[0]
    _logsumexp_kernel[(M,)](input_2d, out, input_2d.stride(0), input_2d.stride(1), out.stride(0), N, BLOCK_N=4096, num_warps=4)
    out = out.view(*dims)
    return out

class LogSumExp(Function):
    @staticmethod
    def forward(ctx, input, dim):
        ctx.dim = dim
        output = logsumexp(input, dim)
        ctx.save_for_backward(input, output)
        return output

    @staticmethod
    def backward(ctx, grad_output):
        input, output = ctx.saved_tensors
        dim = ctx.dim
        softmax = (input - output.unsqueeze(dim)).exp_()
        grad_input = grad_output.unsqueeze(dim) * softmax
        return grad_input, None

def fused_repeat_interleave_log_softmax(input, repeats, dim=None, *, output_size=None, dtype=None, out=None):
    y = torch.repeat_interleave(input, repeats, dim=dim)
    if output_size is not None and y.size() != output_size:
        raise ValueError(f"output_size {output_size} does not match computed output size {y.size()}")
    if dim is None:
        y = y.flatten()
        dim = 0
    else:
        if dim < 0:
            dim += y.dim()
    logsumexp_y = LogSumExp.apply(y, dim)
    log_softmax_y = y - logsumexp_y.unsqueeze(dim)
    if dtype is not None:
        log_softmax_y = log_softmax_y.to(dtype=dtype)
    if out is not None:
        if not out.is_cuda:
            raise RuntimeError("out tensor must be on CUDA")
        out.resize_(log_softmax_y.size())
        out.copy_(log_softmax_y)
        return out
    return log_softmax_y
