import triton
import triton.language as tl

@triton.jit
def fused_mul_add_logsoftmax_dropout_bmm_kernel(
    X1_ptr, X2_ptr, O_ptr, M_ptr, Y_ptr,
    stride_X1_b, stride_X1_n, stride_X1_d,
    stride_X2_b, stride_X2_n, stride_X2_d,
    stride_O_b, stride_O_n, stride_O_d,
    stride_M_b, stride_M_n, stride_M_d,
    stride_Y_b, stride_Y_n, stride_Y_d,
    B, N, D_in, D_out,
    p, training, dim,
    BLOCK_SIZE_B: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_D: tl.constexpr
):
    pid = tl.program_id(axis=0)
    num_blocks_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_blocks_d = tl.cdiv(D_in, BLOCK_SIZE_D)
    block_id_n = pid % num_blocks_n
    block_id_d = pid // num_blocks_n

    # Compute the starting indices for the block
    start_n = block_id_n * BLOCK_SIZE_N
    start_d = block_id_d * BLOCK_SIZE_D

    # Initialize offsets for X1, X2, O, and M
    offsets_X1 = start_n * stride_X1_n + start_d * stride_X1_d
    offsets_X2 = start_n * stride_X2_n + start_d * stride_X2_d
    offsets_O = start_n * stride_O_n + start_d * stride_O_d
    offsets_M = start_n * stride_M_n + start_d * stride_M_d

    # Load the data for the block
    X1 = tl.load(X1_ptr + offsets_X1, mask=start_n + tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)
    X2 = tl.load(X2_ptr + offsets_X2, mask=start_n + tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)
    O = tl.load(O_ptr + offsets_O, mask=start_n + tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)
    M = tl.load(M_ptr + offsets_M, mask=start_n + tl.arange(0, BLOCK_SIZE_N) < N, other=0.0)

    # Element-wise multiplication
    Z = X1 * X2

    # Element-wise addition
    S = Z + O

    # Log-softmax
    max_S = tl.max(S, axis=dim)
    S = S - max_S
    exp_S = tl.exp(S)
    sum_exp_S = tl.sum(exp_S, axis=dim)
    L = S - tl.log(sum_exp_S)

    # Dropout
    if training:
        mask = tl.rand() > p
        L = L * mask / (1 - p)

    # Batch matrix multiplication
    Y = tl.dot(L, M)

    # Store the result
    offsets_Y = start_n * stride_Y_n + start_d * stride_Y_d
    tl.store(Y_ptr + offsets_Y, Y, mask=start_n + tl.arange(0, BLOCK_SIZE_N) < N)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE_B': 128, 'BLOCK_SIZE_N': 128, 'BLOCK_SIZE_D': 128}, num_stages=3, num_warps=8),
    ],
    key=['B', 'N', 'D_in', 'D_out']
)
@triton.jit
def fused_mul_add_logsoftmax_dropout_bmm(
    input1, input2, other, mat2, p=0.5, training=True, inplace=False, dim=-1, out=None
):
    B, N, D_in = input1.shape
    _, _, D_out = mat2.shape

    if out is None:
        out = torch.empty((B, N, D_out), device=input1.device, dtype=input1.dtype)

    fused_mul_add_logsoftmax_dropout_bmm_kernel[
        triton.cdiv(N, 128) * triton.cdiv(D_in, 128)
    ](
        input1, input2, other, mat2, out,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        other.stride(0), other.stride(1), other.stride(2),
        mat2.stride(0), mat2.stride(1), mat2.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        B, N, D_in, D_out,
        p, training, dim,
        BLOCK_SIZE_B=128, BLOCK_SIZE_N=128, BLOCK_SIZE_D=128
    )

    return out
