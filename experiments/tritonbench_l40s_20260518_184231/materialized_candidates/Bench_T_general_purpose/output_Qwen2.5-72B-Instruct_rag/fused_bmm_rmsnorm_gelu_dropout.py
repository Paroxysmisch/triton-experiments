import triton
import triton.language as tl

@triton.jit
def fused_bmm_rmsnorm_gelu_dropout_kernel(
    input1_ptr, input2_ptr, output_ptr, dropout_mask_ptr, dropout_p, eps, training, approximate,
    B, N, M, P, BLOCK_SIZE_B, BLOCK_SIZE_N, BLOCK_SIZE_M, BLOCK_SIZE_P,
    input1_stride_b, input1_stride_n, input1_stride_m,
    input2_stride_b, input2_stride_m, input2_stride_p,
    output_stride_b, output_stride_n, output_stride_p,
):
    pid = tl.program_id(axis=0)
    bid = pid // (BLOCK_SIZE_N * BLOCK_SIZE_P)
    nid = (pid % (BLOCK_SIZE_N * BLOCK_SIZE_P)) // BLOCK_SIZE_P
    pid_p = pid % BLOCK_SIZE_P

    # Offsets for input1 and input2
    input1_offset = bid * input1_stride_b + nid * input1_stride_n
    input2_offset = bid * input2_stride_b + pid_p * input2_stride_p

    # Offsets for output
    output_offset = bid * output_stride_b + nid * output_stride_n + pid_p * output_stride_p

    # Load input1 and input2
    input1 = tl.load(input1_ptr + input1_offset, mask=pid < B * BLOCK_SIZE_N * BLOCK_SIZE_P)
    input2 = tl.load(input2_ptr + input2_offset, mask=pid < B * BLOCK_SIZE_N * BLOCK_SIZE_P)

    # Perform batch matrix multiplication
    accum = tl.zeros((BLOCK_SIZE_N, BLOCK_SIZE_P), dtype=tl.float32)
    for m in range(0, M, BLOCK_SIZE_M):
        input1_block = tl.load(input1_ptr + input1_offset + m * input1_stride_m, mask=pid < B * BLOCK_SIZE_N * BLOCK_SIZE_P)
        input2_block = tl.load(input2_ptr + input2_offset + m * input2_stride_m, mask=pid < B * BLOCK_SIZE_N * BLOCK_SIZE_P)
        accum += tl.dot(input1_block, input2_block, allow_tf32=True)

    # Perform RMS normalization
    mean = tl.sum(accum * accum, axis=1) / P
    inv_std = tl.rsqrt(mean + eps)
    normed = accum * inv_std[:, None]

    # Apply GELU activation
    if approximate == 'none':
        normed = 0.5 * normed * (1 + tl.tanh(0.7978845608 * (normed + 0.044715 * normed * normed * normed)))
    elif approximate == 'tanh':
        normed = normed * 0.5 * (1 + tl.tanh(0.7978845608 * normed))

    # Apply dropout if training
    if training:
        dropout_mask = tl.rand() > dropout_p
        normed = tl.where(dropout_mask, normed / (1 - dropout_p), 0.0)
        tl.store(dropout_mask_ptr + output_offset, dropout_mask, mask=pid < B * BLOCK_SIZE_N * BLOCK_SIZE_P)

    # Store the result
    tl.store(output_ptr + output_offset, normed, mask=pid < B * BLOCK_SIZE_N * BLOCK_SIZE_P)

import torch
import triton
import triton.language as tl

def fused_bmm_rmsnorm_gelu_dropout(input1, input2, normalized_shape, dropout_p=0.1, eps=1e-5, training=True, approximate='none', *, out=None):
    B, N, M = input1.shape
    _, _, P = input2.shape

    if out is None:
        out = torch.empty((B, N, P), dtype=input1.dtype, device=input1.device)

    if training:
        dropout_mask = torch.empty((B, N, P), dtype=torch.bool, device=input1.device)
    else:
        dropout_mask = None

    grid = (B * N * P // (32 * 32),)

    fused_bmm_rmsnorm_gelu_dropout_kernel[grid](
        input1, input2, out, dropout_mask, dropout_p, eps, training, approximate,
        B, N, M, P, 32, 32, 32, 32,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        out.stride(0), out.stride(1), out.stride(2)
    )

    return out
