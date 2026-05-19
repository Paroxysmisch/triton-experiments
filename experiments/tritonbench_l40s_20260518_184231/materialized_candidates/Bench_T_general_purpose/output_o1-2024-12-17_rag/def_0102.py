import triton
import triton.language as tl
import torch

@triton.jit
def _logsumexp_kernel(X, OUT, xm_stride, xn_stride, out_stride, N, BLOCK_N: tl.constexpr):
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
    out_val = tl.log(res) + alpha
    rm_out = tl.program_id(0)
    OUT = OUT + rm_out * out_stride
    tl.store(OUT, out_val)

def _logsumexp(input_2d):
    assert input_2d.is_cuda
    M, N = input_2d.shape
    out = input_2d.new_empty(M)
    _logsumexp_kernel[(M,)](
        input_2d, out,
        input_2d.stride(0), input_2d.stride(1), out.stride(0),
        N,
        BLOCK_N=4096, num_warps=4
    )
    return out

def softmax_mul(input, other, dim, dtype=None, out=None):
    """
    def softmax_mul(input, other, dim, dtype=None, out=None) -> Tensor:
        Applies softmax along 'dim' of 'input', then multiplies by 'other'.
        input (Tensor): Input for softmax.
        other (Tensor or Number): Multiplier after softmax.
        dim (int): Dimension along which softmax is calculated.
        dtype (torch.dtype, optional): Cast input to this dtype if specified.
        out (Tensor, optional): Output tensor.
    """
    if dtype is not None:
        input = input.to(dtype)
    # If 'other' is a Tensor and needs casting:
    if isinstance(other, torch.Tensor) and dtype is not None:
        other = other.to(dtype)

    # Permute 'dim' to last dimension for our logsumexp kernel
    if dim < 0:
        dim += input.dim()
    perm = list(range(input.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    x = input.permute(perm)
    shape_x = x.shape
    M, N = shape_x[:-1], shape_x[-1]
    x_2d = x.view(-1, N)
    
    # Compute logsumexp along last axis
    lse = _logsumexp(x_2d)
    lse = lse.view(*M, 1)
    
    # Compute softmax in-place: x = exp(x - lse)
    x_2d.sub_(lse.view(-1))
    x_2d.exp_()
    
    # Multiply by 'other'
    if isinstance(other, torch.Tensor):
        # Permute 'other' the same way if needed for broadcast
        if other.dim() > dim: 
            oshape = list(other.shape)
            # Adjust if dimension mismatch
            while len(oshape) < input.dim():
                oshape.append(1)
            other = other.view(oshape)
            other = other.permute(perm)
        x_2d *= other.view(-1, N)
    else:
        x_2d *= other
    
    # Reshape back
    x = x.view(shape_x).permute(perm)  # swap back to original
    if out is None:
        out = x
    else:
        out.copy_(x)
    return out
