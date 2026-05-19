import triton
import triton.language as tl

@triton.jit
def _layer_norm_fwd_kernel(
    X,  # input tensor
    W,  # weight matrix
    Y,  # output tensor
    stride_d1,  # stride for D1 dimension
    stride_d2,  # stride for D2 dimension
    stride_n,   # stride for N dimension
    stride_w_d2,  # stride for D2 dimension in W
    stride_w_n,   # stride for N dimension in W
    N,  # size of the last dimension
    eps: tl.float32,  # small epsilon value for numerical stability
    BLOCK_SIZE: tl.constexpr  # block size for parallelization
):
    # Get the current block's starting position in the D1 and D2 dimensions
    pid = tl.program_id(axis=0)
    pid_d1 = pid // stride_d2
    pid_d2 = pid % stride_d2

    # Compute the range of elements to process in the N dimension
    rng = tl.arange(0, BLOCK_SIZE)
    mask = rng < N

    # Load the input elements for the current block
    x_ptr = X + pid_d1 * stride_d1 + pid_d2 * stride_d2 + rng * stride_n
    x = tl.load(x_ptr, mask=mask)

    # Compute the mean
    mean = tl.sum(x, axis=0) / N

    # Compute the variance
    x_minus_mean = x - mean
    var = tl.sum(x_minus_mean * x_minus_mean, axis=0) / N

    # Compute the normalized output
    rstd = 1 / tl.sqrt(var + eps)
    y = (x - mean) * rstd

    # Load the weight elements for the current block
    w_ptr = W + pid_d2 * stride_w_d2 + rng * stride_w_n
    w = tl.load(w_ptr, mask=mask)

    # Apply the weight
    y = y * w

    # Store the output
    y_ptr = Y + pid_d1 * stride_d1 + pid_d2 * stride_d2 + rng * stride_n
    tl.store(y_ptr, y, mask=mask)

import torch
import triton
import triton.language as tl

def layernorm_forward(X, W, Y, eps=1e-5, BLOCK_SIZE=128):
    # Get the shape of the input tensor
    D1, D2, N = X.shape

    # Ensure the shapes are compatible
    assert W.shape == (D2, N), "Weight matrix shape must match the last two dimensions of the input tensor"

    # Compute the strides for the input tensor
    stride_d1 = D2 * N
    stride_d2 = N
    stride_n = 1

    # Compute the strides for the weight matrix
    stride_w_d2 = N
    stride_w_n = 1

    # Compute the grid size
    grid = (D1 * D2,)

    # Launch the Triton kernel
    _layer_norm_fwd_kernel[grid](
        X, W, Y,
        stride_d1, stride_d2, stride_n,
        stride_w_d2, stride_w_n,
        N, eps, BLOCK_SIZE
    )

# Example usage
if __name__ == "__main__":
    # Create a random input tensor and weight matrix
    D1, D2, N = 2, 3, 4
    X = torch.randn((D1, D2, N), device='cuda')
    W = torch.randn((D2, N), device='cuda')
    Y = torch.empty_like(X)

    # Perform the layer normalization forward pass
    layernorm_forward(X, W, Y)

    # Print the results
    print("Input tensor X:")
    print(X)
    print("Weight matrix W:")
    print(W)
    print("Output tensor Y:")
    print(Y)
