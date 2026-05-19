import triton
import triton.language as tl
import torch

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
        res = tl.where(c > alpha, res * tl.exp(alpha - c), res)
        alpha = tl.where(c > alpha, c, alpha)
        res += tl.sum(tl.exp(x - alpha), axis=0)
    out = tl.log(res) + alpha
    rm = tl.program_id(0) + tl.arange(0, 1)
    OUT_ptr = OUT + rm * out_stride
    tl.store(OUT_ptr, out)

def logsumexp(input):
    assert input.is_cuda
    *dims, N = input.shape
    input = input.contiguous().view(-1, N)
    out = input.new_empty(*dims).view(-1)
    M = input.shape[0]
    _logsumexp[(M,)](input, out, input.stride(0), input.stride(1), out.stride(0), N,
                     BLOCK_N=4096, num_warps=4)
    return out.view(*dims)

def softmax_log(input, dim=-1, dtype=None):
    if dtype is not None:
        input = input.to(dtype)
    
    if not input.is_cuda:
        log_input = input.log()
        lse = log_input.logsumexp(dim=dim, keepdim=True)
        return (log_input - lse).exp()
    
    log_input = input.log()
    dim = dim if dim >= 0 else input.ndim + dim
    if dim < 0 or dim >= input.ndim:
        raise ValueError("dim out of range")
    
    if dim != log_input.ndim - 1:
        perm = list(range(log_input.ndim))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        log_input_perm = log_input.permute(perm)
    else:
        log_input_perm = log_input
    
    lse = logsumexp(log_input_perm)
    
    output_perm = (log_input_perm - lse.unsqueeze(-1)).exp_()
    
    if dim != log_input.ndim - 1:
        output = output_perm.permute(perm)
    else:
        output = output_perm
    
    return output
