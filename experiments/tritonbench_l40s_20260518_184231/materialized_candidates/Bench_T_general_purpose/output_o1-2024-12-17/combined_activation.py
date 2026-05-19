#!/usr/bin/env python3
import torch
import triton
import triton.language as tl

@triton.jit
def _matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Pointer offsets for this program ID
    rm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    rn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    rm_mask = rm < M
    rn_mask = rn < N

    # Create accumulators
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # k_tile loop
    for k_tile in range(0, K, BLOCK_K):
        # Offsets for A and B
        a_k = k_tile + tl.arange(0, BLOCK_K)
        b_k = k_tile + tl.arange(0, BLOCK_K)

        a_ptrs = A_ptr + (rm[:, None] * stride_am + a_k[None, :] * stride_ak)
        b_ptrs = B_ptr + (b_k[:, None] * stride_bk + rn[None, :] * stride_bn)

        a = tl.load(a_ptrs, mask=(rm_mask[:, None] & (a_k[None, :] < K)), other=0.0)
        b = tl.load(b_ptrs, mask=(b_k[:, None] < K) & rn_mask[None, :], other=0.0)
        acc += tl.dot(a, b)

    # Write back
    c_ptrs = C_ptr + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    tl.store(c_ptrs, acc, mask=(rm_mask[:, None] & rn_mask[None, :]))


@triton.jit
def _activation_kernel(
    X_ptr, W2_ptr, B_ptr, OUT_ptr,
    size, stride_x, stride_w2, stride_b, stride_out
):
    pid = tl.program_id(0)
    block_size = 256
    start = pid * block_size
    offsets = start + tl.arange(0, block_size)
    mask = offsets < size

    x = tl.load(X_ptr + offsets * stride_x, mask=mask, other=0.0)
    w2 = tl.load(W2_ptr + (offsets * stride_w2), mask=mask, other=1.0)
    b = tl.load(B_ptr + (offsets * stride_b), mask=mask, other=0.0)

    # sigmoid
    x_sig = 1.0 / (1.0 + tl.exp(-x))
    # tanh
    x_tanh = (tl.exp(x_sig) - tl.exp(-x_sig)) / (tl.exp(x_sig) + tl.exp(-x_sig))
    # multiply
    out_val = x_tanh * w2
    # add bias
    out_val = out_val + b

    tl.store(OUT_ptr + offsets * stride_out, out_val, mask=mask)


def combined_activation(input, weight1, weight2, bias, *, out=None):
    # input shape: (*, N, D_in)
    # weight1 shape: (D_in, D_out)
    # weight2 broadcastable to [* , N, D_out]
    # bias broadcastable to [* , N, D_out]
    # output shape: (* , N, D_out)

    assert input.dim() >= 2, "Input must have at least 2 dimensions"
    D_in = input.shape[-1]
    D_out = weight1.shape[-1]
    assert weight1.shape[0] == D_in, "Incompatible matmul dimensions"

    leading_shape = input.shape[:-2]
    N = input.shape[-2]
    batch_size = 1
    for s in leading_shape:
        batch_size *= s

    input_reshaped = input.reshape(batch_size, N, D_in)
    M = batch_size * N
    K = D_in
    # shape of output after matmul: [M, D_out]
    out_matmul = torch.empty((M, D_out), dtype=input.dtype, device=input.device)

    # MatMul strides
    stride_am = input_reshaped.stride(0)
    stride_ak = input_reshaped.stride(2)
    stride_bk = weight1.stride(0)
    stride_bn = weight1.stride(1)
    stride_cm = out_matmul.stride(0)
    stride_cn = out_matmul.stride(1)

    grid = (
        ( (M + 127) // 128 ),
        ( (D_out + 127) // 128 )
    )

    _matmul_kernel[grid](
        input_reshaped, weight1, out_matmul,
        M, D_out, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        BLOCK_M=128, BLOCK_N=128, BLOCK_K=32
    )

    # Now we apply the activation + mul and bias
    final_shape = list(leading_shape) + [N, D_out]
    if out is None:
        out = torch.empty(final_shape, dtype=input.dtype, device=input.device)

    out_reshaped = out.reshape(M, D_out)

    size = M * D_out

    # Expand weight2 and bias to out_reshaped if needed
    broadcast_shape = [M, D_out]
    w2_expanded = weight2.broadcast_to(final_shape).reshape(broadcast_shape)
    b_expanded = bias.broadcast_to(final_shape).reshape(broadcast_shape)

    stride_x = out_matmul.stride(0)
    stride_w2 = w2_expanded.stride(0)
    stride_b = b_expanded.stride(0)
    stride_out_ = out_reshaped.stride(0)

    # For row-major strides, multiply by the second dimension if necessary
    if out_matmul.dim() == 2:
        if out_matmul.stride(1) == 1:
            stride_x = 1
        else:
            stride_x = out_matmul.stride(0)
    if w2_expanded.dim() == 2:
        if w2_expanded.stride(1) == 1:
            stride_w2 = 1
        else:
            stride_w2 = w2_expanded.stride(0)
    if b_expanded.dim() == 2:
        if b_expanded.stride(1) == 1:
            stride_b = 1
        else:
            stride_b = b_expanded.stride(0)
    if out_reshaped.dim() == 2:
        if out_reshaped.stride(1) == 1:
            stride_out_ = 1
        else:
            stride_out_ = out_reshaped.stride(0)

    grid_size = ( (size + 255) // 256, )
    _activation_kernel[grid_size](
        out_matmul, w2_expanded, b_expanded, out_reshaped,
        size, stride_x, stride_w2, stride_b, stride_out_
    )

    return out.reshape(final_shape)
