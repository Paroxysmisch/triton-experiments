import torch
import triton
import triton.language as tl

@triton.jit
def combined_activation_kernel(
    X_ptr, W1_ptr, W2_ptr, B_ptr, Y_ptr,
    batch_size, N, D_in, D_out,
    stride_xn, stride_xdin,
    stride_w1din, stride_w1dout,
    stride_w2dout,
    stride_bdout,
    stride_yn, stride_ydout,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the row and column indices for the block
    batch_id = tl.program_id(0)
    n = tl.program_id(1) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    dout = tl.arange(0, BLOCK_SIZE)

    # Load input, weights, and bias
    X = tl.load(X_ptr + batch_id * stride_xn + n[:, None] * stride_xdin, mask=n[:, None] < N)
    W1 = tl.load(W1_ptr + dout[None, :] * stride_w1dout, mask=dout[None, :] < D_out)
    W2 = tl.load(W2_ptr + dout[None, :], mask=dout[None, :] < D_out)
    B = tl.load(B_ptr + dout[None, :], mask=dout[None, :] < D_out)

    # Perform matrix multiplication
    Z = tl.dot(X, W1)

    # Apply sigmoid and tanh
    Z = tl.sigmoid(Z)
    Z = tl.tanh(Z)

    # Element-wise multiplication and addition
    Y = Z * W2 + B

    # Store the result
    tl.store(Y_ptr + batch_id * stride_yn + n[:, None] * stride_ydout, Y, mask=n[:, None] < N)

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Extract dimensions
    *batch_dims, N, D_in = input.shape
    D_out = weight1.shape[1]

    # Prepare output tensor
    if out is None:
        out = torch.empty(*batch_dims, N, D_out, device=input.device, dtype=input.dtype)

    # Compute grid size
    batch_size = int(torch.prod(torch.tensor(batch_dims)))
    grid = (batch_size, triton.cdiv(N, 128))

    # Launch kernel
    combined_activation_kernel[grid](
        input, weight1, weight2, bias, out,
        batch_size, N, D_in, D_out,
        input.stride(-2), input.stride(-1),
        weight1.stride(0), weight1.stride(1),
        weight2.stride(0),
        bias.stride(0),
        out.stride(-2), out.stride(-1),
        BLOCK_SIZE=128
    )

    return out
