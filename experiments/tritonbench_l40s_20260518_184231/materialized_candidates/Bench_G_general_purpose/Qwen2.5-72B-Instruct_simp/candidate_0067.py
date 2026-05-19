import triton
import triton.language as tl

@triton.jit
def rms_norm_kernel(X, W, Y, stride_xm, stride_ym, stride_xn, stride_wn, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    row_start = pid * BLOCK_SIZE

    offsets_m = row_start + tl.arange(0, BLOCK_SIZE)
    offsets_n = tl.arange(0, N)

    mask = offsets_m < M

    X_row = tl.load(X + offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn, mask=mask[:, None], other=0.0)
    W_row = tl.load(W + offsets_n * stride_wn)

    # Compute the mean of the squares
    X_squared = X_row * X_row
    mean_squared = tl.sum(X_squared, axis=1) / N

    # Compute the root mean square
    rms = tl.sqrt(mean_squared)

    # Normalize the row
    X_normalized = X_row / (rms[:, None] + 1e-6)

    # Multiply by the weight vector
    Y_row = X_normalized * W_row

    # Store the result
    tl.store(Y + offsets_m[:, None] * stride_ym + offsets_n[None, :] * stride_yn, Y_row, mask=mask[:, None])

import torch
import triton
import triton.language as tl
from triton.runtime import JITFunction

# Import the Triton kernel
from rms_norm_kernel import rms_norm_kernel

class RmsNorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W):
        M, N = X.shape
        Y = torch.empty_like(X)

        # Define the grid and block sizes
        grid = (triton.cdiv(M, 1024),)
        block = (1024,)

        # Launch the kernel
        rms_norm_kernel[grid, block](
            X, W, Y,
            X.stride(0), Y.stride(0), X.stride(1), W.stride(0),
            M, N, block[0]
        )

        ctx.save_for_backward(X, W)
        return Y

    @staticmethod
    def backward(ctx, dY):
        X, W = ctx.saved_tensors
        M, N = X.shape
        dX = torch.empty_like(X)
        dW = torch.empty_like(W)

        # Define the grid and block sizes
        grid = (triton.cdiv(M, 1024),)
        block = (1024,)

        # Launch the backward kernel (not implemented here, but you can add it if needed)
        # For simplicity, we assume the backward pass is not required for this example

        return dX, dW

# Define a PyTorch module to use the RmsNorm function
class RmsNormModule(torch.nn.Module):
    def __init__(self, num_features):
        super(RmsNormModule, self).__init__()
        self.weight = torch.nn.Parameter(torch.ones(num_features))

    def forward(self, X):
        return RmsNorm.apply(X, self.weight)

# Example usage
if __name__ == "__main__":
    # Create a random input tensor
    X = torch.randn(1024, 128, device='cuda')
    # Create the RmsNorm module
    rms_norm = RmsNormModule(X.shape[1]).to('cuda')
    # Perform the RMS normalization
    Y = rms_norm(X)
    print(Y)
