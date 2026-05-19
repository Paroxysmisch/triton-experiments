import triton
import triton.language as tl
import torch

# Triton kernel for logsumexp
@triton.jit
def _logsumexp_kernel(X, OUT, stride_xm, stride_xn, stride_out, N, BLOCK_N: tl.constexpr):
    # Get the program ID for the reduction axis
    rm = tl.program_id(0)
    # Initialize variables for max value and result
    alpha = tl.zeros((1,), tl.float32) + -float('inf')
    res = tl.zeros((1,), tl.float32)
    # Iterate over blocks of the reduction axis
    for bn in range(0, N, BLOCK_N):
        rn = bn + tl.arange(0, BLOCK_N)
        # Compute the offset for the current block
        Xmn = X + rm * stride_xm + rn * stride_xn
        # Load the input data with masking
        x = tl.load(Xmn, mask=rn < N, other=-float('inf'))
        # Find the maximum value for numerical stability
        c = tl.max(x, axis=0)
        # Update the result and max value
        res = tl.where(c > alpha, res * tl.exp(alpha - c), res)
        alpha = tl.where(c > alpha, c, alpha)
        res += tl.sum(tl.exp(x - alpha), axis=0)
    # Compute the final logsumexp value
    out = tl.log(res) + alpha
    # Store the result in the output tensor
    rm = tl.program_id(0) + tl.arange(0, 1)
    OUT = OUT + rm * stride_out
    tl.store(OUT, out)

# Wrapper function for logsumexp
def logsumexp(input, dim, keepdim=False, *, out=None):
    assert input.is_cuda, "Input must be a CUDA tensor"
    # Move the specified dimension to the last
    input = input.transpose(dim, -1)
    *dims, N = input.shape
    # Reshape input for processing
    input = input.view(-1, N)
    # Prepare the output tensor
    out_shape = list(dims)
    if keepdim:
        out_shape.insert(dim, 1)
    out = input.new_empty(out_shape).view(-1)
    M = input.shape[0]
    # Launch the Triton kernel
    _logsumexp_kernel[(M,)](input, out, input.stride(0), input.stride(1), out.stride(0), N,
                            BLOCK_N=4096, num_warps=4)
    # Reshape the output tensor
    out = out.view(*dims)
    if keepdim:
        out = out.unsqueeze(dim)
    return out

# Example usage
x = torch.randn(128, 64, device='cuda')
result = logsumexp(x, dim=1, keepdim=True)
print(result)
