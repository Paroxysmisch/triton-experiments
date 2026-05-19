import triton
import triton.language as tl

@triton.jit
def fused_transformer_block_kernel(
    input_ptr, weight1_ptr, weight2_ptr, residual_ptr, output_ptr,
    input_stride, weight1_stride, weight2_stride, residual_stride, output_stride,
    N, D_in, D_k, D_out, dropout_p, eps,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    # Load input and weight1
    input_offsets = block_start + tl.arange(0, BLOCK_SIZE)
    input_mask = input_offsets < N
    input = tl.load(input_ptr + input_offsets[:, None] * input_stride + tl.arange(0, D_in)[None, :], mask=input_mask[:, None], other=0.0)
    weight1 = tl.load(weight1_ptr + tl.arange(0, D_in)[:, None] * weight1_stride + tl.arange(0, D_k)[None, :])

    # Compute Z1 = X W1
    Z1 = tl.dot(input, weight1)

    # Compute Z2 = softmax(Z1)
    Z2 = tl.softmax(Z1, axis=-1)

    # Compute Z3 = dropout(Z2, p)
    rng = tl.random philox4x4(seed=tl.program_id(axis=1))
    Z3 = tl.where(tl.rand(rng) < dropout_p, 0.0, Z2 / (1 - dropout_p))

    # Load weight2
    weight2 = tl.load(weight2_ptr + tl.arange(0, D_k)[:, None] * weight2_stride + tl.arange(0, D_out)[None, :])

    # Compute Z4 = Z3 W2
    Z4 = tl.dot(Z3, weight2)

    # Load residual
    residual = tl.load(residual_ptr + input_offsets[:, None] * residual_stride + tl.arange(0, D_out)[None, :], mask=input_mask[:, None], other=0.0)

    # Compute Y = LayerNorm(Z4 + R)
    Z4_plus_R = Z4 + residual
    mean = tl.mean(Z4_plus_R, axis=-1)
    var = tl.var(Z4_plus_R, axis=-1)
    Y = (Z4_plus_R - mean[:, None]) * tl.rsqrt(var[:, None] + eps)

    # Store output
    tl.store(output_ptr + input_offsets[:, None] * output_stride + tl.arange(0, D_out)[None, :], Y, mask=input_mask[:, None])

import torch
import triton
import triton.language as tl

def fused_transformer_block(input, weight1, weight2, residual, dropout_p=0.1, eps=1e-5, *, out=None):
    # Check input dimensions
    N, D_in = input.shape[-2:]
    D_k = weight1.shape[-1]
    D_out = weight2.shape[-1]

    # Ensure the dimensions are compatible
    assert input.shape[-1] == D_in, "Input last dimension must match D_in"
    assert weight1.shape == (D_in, D_k), "Weight1 must have shape (D_in, D_k)"
    assert weight2.shape == (D_k, D_out), "Weight2 must have shape (D_k, D_out)"
    assert residual.shape[-2:] == (N, D_out), "Residual must be broadcastable to (N, D_out)"

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty_like(input, device=input.device, dtype=input.dtype)

    # Define grid and block size
    BLOCK_SIZE = 128
    grid = (N // BLOCK_SIZE + (N % BLOCK_SIZE > 0), 1)

    # Launch the kernel
    fused_transformer_block_kernel[grid](
        input, weight1, weight2, residual, out,
        input.stride(-2), weight1.stride(-2), weight2.stride(-2), residual.stride(-2), out.stride(-2),
        N, D_in, D_k, D_out, dropout_p, eps,
        BLOCK_SIZE
    )

    return out
