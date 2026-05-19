import triton
import triton.language as tl

@triton.jit
def combined_activation_kernel(
    X_ptr, W1_ptr, W2_ptr, B_ptr, Out_ptr,
    N, D_in, D_out,
    stride_xn, stride_xd,
    stride_w1d, stride_w1o,
    stride_w2o,
    stride_bo,
    stride_outn, stride_outo,
    BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_OUT: tl.constexpr
):
    pid = tl.program_id(0)
    batch_idx = pid // D_out
    out_idx = pid % D_out

    # Load inputs
    x = tl.load(X_ptr + batch_idx * stride_xn + tl.arange(0, BLOCK_SIZE_N) * stride_xd)
    w1 = tl.load(W1_ptr + tl.arange(0, D_in) * stride_w1d + out_idx * stride_w1o)

    # Matrix multiplication
    z = tl.dot(x, w1)

    # Sigmoid activation
    sigmoid_z = 1 / (1 + tl.exp(-z))

    # Tanh activation
    tanh_sigmoid_z = (tl.exp(sigmoid_z) - tl.exp(-sigmoid_z)) / (tl.exp(sigmoid_z) + tl.exp(-sigmoid_z))

    # Element-wise multiplication with W2
    w2 = tl.load(W2_ptr + out_idx * stride_w2o)
    elementwise_mult = tanh_sigmoid_z * w2

    # Add bias
    b = tl.load(B_ptr + out_idx * stride_bo)
    result = elementwise_mult + b

    # Store result
    tl.store(Out_ptr + batch_idx * stride_outn + out_idx * stride_outo, result)

import torch

def combined_activation(input, weight1, weight2, bias, *, out=None):
    # Ensure input dimensions are compatible
    assert input.shape[-1] == weight1.shape[0], "Incompatible dimensions for matrix multiplication"
    
    # Determine output dimensions
    *batch_dims, N, D_in = input.shape
    D_out = weight1.shape[1]

    # Prepare output tensor
    if out is None:
        out = torch.empty((*batch_dims, N, D_out), device=input.device, dtype=input.dtype)

    # Calculate strides
    stride_xn, stride_xd = input.stride()[-2:]
    stride_w1d, stride_w1o = weight1.stride()
    stride_w2o = weight2.stride()[-1]
    stride_bo = bias.stride()[-1]
    stride_outn, stride_outo = out.stride()[-2:]

    # Launch Triton kernel
    grid = (triton.cdiv(N * D_out, BLOCK_SIZE_N),)
    combined_activation_kernel[grid](
        input, weight1, weight2, bias, out,
        N, D_in, D_out,
        stride_xn, stride_xd,
        stride_w1d, stride_w1o,
        stride_w2o,
        stride_bo,
        stride_outn, stride_outo,
        BLOCK_SIZE_N=128, BLOCK_SIZE_OUT=128
    )

    return out
