import triton
import triton.language as tl

@triton.jit
def softplus_linear_kernel(
    x_ptr, weight_ptr, bias_ptr, out_ptr,
    M, N, K, beta, threshold,
    stride_xm, stride_xk, stride_wk, stride_wn, stride_b, stride_om, stride_on,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr
):
    # Compute program ID
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Create block indices
    block_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    block_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)

    # Create pointers for the inputs and output
    x_ptrs = x_ptr + block_m[:, None] * stride_xm + block_n[None, :] * stride_xk
    w_ptrs = weight_ptr + block_m[:, None] * stride_wk + block_n[None, :] * stride_wn
    out_ptrs = out_ptr + block_m[:, None] * stride_om + block_n[None, :] * stride_on

    # Load data
    x = tl.load(x_ptrs, mask=block_m[:, None] < M)
    w = tl.load(w_ptrs, mask=block_n[None, :] < N)

    # Perform linear transformation
    linear_out = tl.dot(x, w)

    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + block_n, mask=block_n < N)
        linear_out += bias

    # Apply Softplus
    softplus_out = (1 / beta) * tl.log(1 + tl.exp(beta * linear_out))

    # Apply threshold for numerical stability
    stable_out = tl.where(linear_out > threshold, linear_out, softplus_out)

    # Store result
    tl.store(out_ptrs, stable_out, mask=block_m[:, None] < M)

import torch

def softplus_linear(input, weight, bias=None, beta=1, threshold=20):
    # Ensure input is a 2D tensor
    assert input.ndim == 2, "Input tensor must be 2D"
    assert weight.ndim == 2, "Weight tensor must be 2D"
    
    # Get dimensions
    M, K = input.shape
    K_w, N = weight.shape
    assert K == K_w, "Input and weight dimensions do not match"

    # Prepare output tensor
    output = torch.empty((M, N), device=input.device, dtype=input.dtype)

    # Define block sizes
    BLOCK_SIZE_M = 128
    BLOCK_SIZE_N = 128

    # Launch the kernel
    grid = (triton.cdiv(M, BLOCK_SIZE_M), triton.cdiv(N, BLOCK_SIZE_N))
    softplus_linear_kernel[grid](
        input, weight, bias, output,
        M, N, K, beta, threshold,
        input.stride(0), input.stride(1),
        weight.stride(0), weight.stride(1),
        bias.stride(0) if bias is not None else 0,
        output.stride(0), output.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M, BLOCK_SIZE_N=BLOCK_SIZE_N
    )

    return output
