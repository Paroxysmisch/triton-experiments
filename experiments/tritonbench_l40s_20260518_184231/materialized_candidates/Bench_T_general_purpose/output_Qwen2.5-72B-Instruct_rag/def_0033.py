import triton
import triton.language as tl
import torch

# Triton kernel for streaming logsumexp
@triton.jit
def _logsumexp(X, OUT, stride_x, stride_dim, stride_out, N, BLOCK_N: tl.constexpr):
    pid = tl.program_id(0)
    alpha = tl.zeros((1,), tl.float32) + -float('inf')
    res = tl.zeros((1,), tl.float32)
    for bn in range(0, N, BLOCK_N):
        rn = bn + tl.arange(0, BLOCK_N)
        Xmn = X + pid * stride_x + rn * stride_dim
        x = tl.load(Xmn, mask=rn < N, other=-float('inf'))
        c = tl.max(x, axis=0)
        # correct the current sum and update the max
        res = tl.where(c > alpha, res * tl.exp(alpha - c), res)
        alpha = tl.where(c > alpha, c, alpha)
        res += tl.sum(tl.exp(x - alpha), axis=0)
    out = tl.log(res) + alpha
    tl.store(OUT + pid * stride_out, out)

# Function to call the Triton kernel
def logsumexp(input, dim, keepdim=False, *, out=None) -> torch.Tensor:
    assert input.is_cuda, "Input tensor must be on a CUDA device."
    *dims, N = input.shape
    input = input.view(-1, N)
    if out is None:
        out_shape = (*dims, 1) if keepdim else (*dims,)
        out = input.new_empty(out_shape)
    else:
        assert out.is_cuda, "Output tensor must be on a CUDA device."
        assert out.shape == (*dims, 1) if keepdim else (*dims,), "Output tensor shape mismatch."
    
    M = input.shape[0]
    _logsumexp[(M,)](input, out, input.stride(0), input.stride(1), out.stride(0), N,
                     BLOCK_N=4096, num_warps=4)
    
    if keepdim:
        out = out.view(*dims, 1)
    else:
        out = out.view(*dims)
    
    return out

# Example usage
if __name__ == "__main__":
    x = torch.randn((32, 64), device='cuda')
    result = logsumexp(x, dim=1, keepdim=True)
    print(result)
