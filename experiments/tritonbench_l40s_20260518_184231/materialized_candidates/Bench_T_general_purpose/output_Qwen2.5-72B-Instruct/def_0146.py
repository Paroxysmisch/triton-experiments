import triton
import triton.language as tl

@triton.jit
def elu_kernel(X, W, B, Y, alpha, stride_xm, stride_xn, stride_ym, stride_yn, stride_b, M, N, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets_m = block_start + tl.arange(0, BLOCK_SIZE)
    offsets_n = tl.arange(0, N)

    mask = offsets_m < M
    x_ptrs = X + (offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn)
    w_ptrs = W + (offsets_n[:, None] * stride_ym + offsets_m[None, :] * stride_yn)
    y_ptrs = Y + (offsets_m * stride_ym)

    x = tl.load(x_ptrs, mask=mask[:, None], other=0.0)
    w = tl.load(w_ptrs, mask=mask[None, :], other=0.0)

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for n in range(0, N, BLOCK_SIZE):
        acc += tl.dot(x, w)
        x = tl.load(x_ptrs + n * stride_xn, mask=mask[:, None], other=0.0)
        w = tl.load(w_ptrs + n * stride_yn, mask=mask[None, :], other=0.0)

    if B is not None:
        b_ptrs = B + offsets_m * stride_b
        b = tl.load(b_ptrs, mask=mask, other=0.0)
        acc += b

    y = tl.where(acc > 0, acc, alpha * (tl.exp(acc) - 1))
    tl.store(y_ptrs, y, mask=mask)

import torch
import triton
import triton.language as tl

def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
    # Ensure input and weight are on the same device
    assert input.device == weight.device, "Input and weight must be on the same device"
    
    # Get device and data type
    device = input.device
    dtype = input.dtype
    
    # Get dimensions
    M, K = input.shape
    K, N = weight.shape
    
    # Allocate output tensor
    if inplace:
        output = input
    else:
        output = torch.empty((M, N), device=device, dtype=dtype)
    
    # Allocate bias tensor if provided
    if bias is not None:
        assert bias.shape == (N,), "Bias must have the same number of elements as the output dimension"
        bias = bias.to(device)
    else:
        bias = None
    
    # Define grid and block sizes
    BLOCK_SIZE = 128
    grid = (triton.cdiv(M, BLOCK_SIZE),)
    
    # Launch the Triton kernel
    elu_kernel[grid](
        input, weight, bias, output, alpha,
        input.stride(0), input.stride(1),
        output.stride(0), output.stride(1),
        bias.stride(0) if bias is not None else 0,
        M, N, BLOCK_SIZE
    )
    
    return output
