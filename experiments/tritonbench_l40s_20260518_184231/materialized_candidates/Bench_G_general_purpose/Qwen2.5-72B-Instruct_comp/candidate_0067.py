import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(
    X, W, Y, N, eps,
    X.stride(0), X.stride(1),
    W.stride(0),
    Y.stride(0), Y.stride(1),
    pid: tl.constexpr,
):
    # Compute the starting index for the current block
    row_start = pid * X.stride(0)
    # Load the row of X
    x = tl.load(X + row_start + tl.arange(0, N), mask=tl.arange(0, N) < N, other=0.0)
    # Compute the variance
    var = tl.sum(x * x, axis=0) / N
    # Compute the reciprocal of the root of variance plus epsilon
    rrms = 1.0 / tl.sqrt(var + eps)
    # Load the weight for the current row
    w = tl.load(W + pid)
    # Compute the normalized and scaled output
    y = (x * rrms).to(Y.dtype.element_ty) * w
    # Store the result in Y
    tl.store(Y + row_start + tl.arange(0, N), y, mask=tl.arange(0, N) < N)

import torch
import triton
import triton.language as tl

class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w, eps=1e-6):
        # Ensure the input and weight tensors are on the same device
        assert x.device == w.device, "Input and weight tensors must be on the same device"
        # Ensure the weight tensor has the correct shape
        assert w.shape[0] == x.shape[0], "Weight tensor must have the same number of rows as the input tensor"
        
        # Allocate output tensor
        y = torch.empty_like(x)
        
        # Launch the kernel
        grid = (x.shape[0],)
        rms_norm_kernel[grid](
            x, w, y, x.shape[1], eps,
            x.stride(0), x.stride(1),
            w.stride(0),
            y.stride(0), y.stride(1),
        )
        
        # Save the input and weight tensors for the backward pass
        ctx.save_for_backward(x, w, y)
        ctx.eps = eps
        
        return y

def rms_norm(x, normalized_shape, weight, eps=1e-6):
    # Ensure the input tensor has the correct shape
    assert x.shape[-1] == normalized_shape, "Input tensor must have the same last dimension as normalized_shape"
    # Ensure the weight tensor has the correct shape
    assert weight.shape[0] == x.shape[0], "Weight tensor must have the same number of rows as the input tensor"
    
    # Apply the RMS normalization
    return RmsNorm.apply(x, weight, eps)
