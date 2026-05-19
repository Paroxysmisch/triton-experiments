import triton
import triton.language as tl
import torch

# Triton kernel for softmax and multiplication
@triton.jit
def _softmax_mul(X, OTHER, OUT, xm_stride, xn_stride, other_stride, out_stride, N, BLOCK_N: tl.constexpr):
    rm = tl.program_id(0)
    alpha = tl.zeros((1,), tl.float32) + -float('inf')
    res = tl.zeros((BLOCK_N,), tl.float32)
    
    for bn in range(0, N, BLOCK_N):
        rn = bn + tl.arange(0, BLOCK_N)
        Xmn = X + rm * xm_stride + rn * xn_stride
        x = tl.load(Xmn, mask=rn < N, other=-float('inf'))
        
        # Compute the maximum value for numerical stability
        c = tl.max(x, axis=0)
        
        # Compute the exponentials
        exp_x = tl.exp(x - c)
        
        # Compute the sum of exponentials
        sum_exp = tl.sum(exp_x, axis=0)
        
        # Compute the softmax values
        softmax_x = exp_x / sum_exp
        
        # Load the other tensor
        Othermn = OTHER + rm * other_stride + rn * other_stride
        other = tl.load(Othermn, mask=rn < N, other=1.0)
        
        # Compute the final result
        res = softmax_x * other
        
        # Store the result
        OUTmn = OUT + rm * out_stride + rn * out_stride
        tl.store(OUTmn, res, mask=rn < N)

# Python wrapper function
def softmax_mul(input, other, dim, dtype=None, out=None):
    if not input.is_cuda:
        input = input.cuda()
    if isinstance(other, (int, float)):
        other = torch.tensor(other, device=input.device, dtype=input.dtype)
    if not other.is_cuda:
        other = other.cuda()
    
    if dtype is not None:
        input = input.to(dtype)
        other = other.to(dtype)
    
    if out is None:
        out = torch.empty_like(input, device=input.device, dtype=input.dtype)
    
    *dims, N = input.shape
    input = input.view(-1, N)
    other = other.view(-1, N)
    out = out.view(-1, N)
    
    M = input.shape[0]
    _softmax_mul[(M,)](input, other, out, input.stride(0), input.stride(1), other.stride(1), out.stride(1), N, BLOCK_N=4096, num_warps=4)
    
    return out.view(*dims)

# Example usage
input = torch.randn(2, 3, 4, device='cuda')
other = torch.randn(2, 3, 4, device='cuda')
dim = 2
result = softmax_mul(input, other, dim)
print(result)
