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
    OUT = OUT + rm * out_stride
    tl.store(OUT, out)

def logsumexp(input, dim, keepdim=False, *, out=None):
    assert input.is_cuda, "Input must be a CUDA tensor."
    nd = input.dim()
    if dim < 0:
        dim += nd
    
    # Permute so that the reduction dimension is last
    if dim != nd - 1:
        perm = list(range(nd))
        perm[dim], perm[-1] = perm[-1], perm[dim]
        input = input.permute(perm)
        inv_perm = list(range(nd))
        inv_perm[dim], inv_perm[-1] = inv_perm[-1], inv_perm[dim]
    else:
        perm = None
        inv_perm = None
    
    shape_perm = input.shape
    M = 1
    for s in shape_perm[:-1]:
        M *= s
    N = shape_perm[-1]
    
    input_2d = input.reshape(M, N)
    result_2d = input_2d.new_empty(M)
    
    _logsumexp[(M,)](
        input_2d,
        result_2d,
        input_2d.stride(0),
        input_2d.stride(1),
        result_2d.stride(0),
        N,
        BLOCK_N=4096,
        num_warps=4
    )
    
    result_nd = result_2d.reshape(shape_perm[:-1])
    if inv_perm is not None:
        result_nd = result_nd.permute(inv_perm)
    
    if keepdim:
        result_nd = result_nd.unsqueeze(dim)
    
    if out is not None:
        out.copy_(result_nd)
        return out
    return result_nd
