import torch
import triton
import triton.language as tl
from torch.amp import custom_bwd, custom_fwd
from .log_softmax import log_softmax, log_softmax_backward
from .dropout import dropout, dropout_backward
from .bmm import bmm, bmm_batch

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')

def cfggen():
    warps = [1, 2, 4, 8, 16, 32]
    configs = [
        triton.Config({'BLOCK_SIZE_M': 128, 'BLOCK_SIZE_N': 256, 'BLOCK_SIZE_K': 64}, num_stages=3, num_warps=w)
        for w in warps
    ]
    return configs

if TRITON_22:
    @triton.jit
    def fused_mul_add_logsoftmax_dropout_bmm_fwd_h(
        input1_ptr, input2_ptr, other_ptr, mat2_ptr, output_ptr, dropout_ptr, input1_mask_ptr, input2_mask_ptr,
        M, N, K, c, stride_input1_batch, stride_input1_m, stride_input1_k, stride_input2_batch, stride_input2_n,
        stride_input2_k, stride_other_batch, stride_other_m, stride_other_k, stride_mat2_batch, stride_mat2_m,
        stride_mat2_k, stride_output_batch, stride_output_m, stride_output_n, stride_dropout_batch, stride_dropout_m,
        stride_dropout_k, stride_input1_mask_batch, stride_input1_mask_m, stride_input1_mask_k, stride_input2_mask_batch,
        stride_input2_mask_n, stride_input2_mask_k, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr, META: tl.constexpr, BLOCK_SIZE_H: tl.constexpr, P: tl.constexpr
    ):
        pid = tl.program_id(axis=0)
        num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
        num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
        num_pid_in_group = GROUP_SIZE_M * num_pid_n
        group_id = pid // num_pid_in_group
        first_pid_m = group_id * GROUP_SIZE_M
        group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
        pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
        pid_n = (pid % num_pid_in_group) // group_size_m

        input1_ptr += pid_m * BLOCK_SIZE_M * stride_input1_m + pid_n * BLOCK_SIZE_N * stride_input1_k
        input2_ptr += pid_m * BLOCK_SIZE_M * stride_input2_m + pid_n * BLOCK_SIZE_N * stride_input2_k
        other_ptr += pid_m * BLOCK_SIZE_M * stride_other_m + pid_n * BLOCK_SIZE_N * stride_other_k
        mat2_ptr += pid_m * BLOCK_SIZE_M * stride_mat2_m + pid_n * BLOCK_SIZE_N * stride_mat2_k
        output_ptr += pid_m * BLOCK_SIZE_M * stride_output_m + pid_n * BLOCK_SIZE_N * stride_output_n
        dropout_ptr += pid_m * BLOCK_SIZE_M * stride_dropout_m + pid_n * BLOCK_SIZE_N * stride_dropout_k
        input1_mask_ptr += pid_m * BLOCK_SIZE_M * stride_input1_mask_m + pid_n * BLOCK_SIZE_N * stride_input1_mask_k
        input2_mask_ptr += pid_m * BLOCK_SIZE_M * stride_input2_mask_m + pid_n * BLOCK_SIZE_N * stride_input2_mask_k

        offs_m = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
        offs_n = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
        offs_k = tl.arange(0, BLOCK_SIZE_K)
        input1_ptrs = input1_ptr + (offs_m[:, None] * stride_input1_m + offs_k[None, :] * stride_input1_k)
        input2_ptrs = input2_ptr + (offs_k[:, None] * stride_input2_k + offs_n[None, :] * stride_input2_n)
        other_ptrs = other_ptr + (offs_k[:, None] * stride_other_k + offs_n[None, :] * stride_other_n)
        mat2_ptrs = mat2_ptr + (offs_m[:, None] * stride_mat2_m + offs_k[None, :] * stride_mat2_k)
        output_ptrs = output_ptr + (offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n)
        dropout_ptrs = dropout_ptr + (offs_m[:, None] * stride_dropout_m + offs_n[None, :] * stride_dropout_k)
        input1_mask_ptrs = input1_mask_ptr + (offs_m[:, None] * stride_input1_mask_m + offs_k[None, :] * stride_input1_mask_k)
        input2_mask_ptrs = input2_mask_ptr + (offs_k[:, None] * stride_input2_mask_k + offs_n[None, :] * stride_input2_mask_n)

        input1_mask = tl.load(input1_mask_ptrs, mask=(offs_k[None, :] < K) & (offs_m[:, None] < M), other=0.0)
        input2_mask = tl.load(input2_mask_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        input1 = tl.load(input1_ptrs, mask=(offs_k[None, :] < K) & (offs_m[:, None] < M), other=0.0)
        input1 = input1 * input1_mask
        input2 = tl.load(input2_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        input2 = input2 * input2_mask
        mul = input1 * input2
        add = mul + tl.load(other_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        log_softmax = log_softmax(add, axis=1, meta=META)
        dropout = tl.where(log_softmax < 0, 0, log_softmax)
        dropout = dropout * (1 - P)
        if META == 'float16':
            dropout = dropout.to(tl.float16)
        tl.store(output_ptrs, dropout, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
        tl.store(dropout_ptrs, dropout, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
    else:
        dropout_ptrs = dropout_ptr + (offs_m[:, None] * stride_dropout_m + offs_n[None, :] * stride_dropout_k)
        input1_ptrs = input1_ptr + (offs_m[:, None] * stride_input1_m + offs_k[None, :] * stride_input1_k)
        input2_ptrs = input2_ptr + (offs_k[:, None] * stride_input2_k + offs_n[None, :] * stride_input2_n)
        other_ptrs = other_ptr + (offs_k[:, None] * stride_other_k + offs_n[None, :] * stride_other_n)
        mat2_ptrs = mat2_ptr + (offs_m[:, None] * stride_mat2_m + offs_k[None, :] * stride_mat2_k)
        output_ptrs = output_ptr + (offs_m[:, None] * stride_output_m + offs_n[None, :] * stride_output_n)

        input1_mask_ptrs = input1_mask_ptr + (offs_m[:, None] * stride_input1_mask_m + offs_k[None, :] * stride_input1_mask_k)
        input2_mask_ptrs = input2_mask_ptr + (offs_k[:, None] * stride_input2_mask_k + offs_n[None, :] * stride_input2_mask_n)

        input1_mask = tl.load(input1_mask_ptrs, mask=(offs_k[None, :] < K) & (offs_m[:, None] < M), other=0.0)
        input2_mask = tl.load(input2_mask_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        input1 = tl.load(input1_ptrs, mask=(offs_k[None, :] < K) & (offs_m[:, None] < M), other=0.0)
        input1 = input1 * input1_mask
        input2 = tl.load(input2_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)
        input2 = input2 * input2_mask
        mul = input1 * input2
        add = mul + tl.load(other_ptrs, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N), other=0.0)

        log_softmax = log_softmax(add, axis=1, meta=META)
        dropout = tl.where(log_softmax < 0, 0, log_softmax)
        dropout = dropout * (1 - P)

        tl.store(output_ptrs, dropout, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))
        tl.store(dropout_ptrs, dropout, mask=(offs_k[:, None] < K) & (offs_n[None, :] < N))

    @triton.jit
    def fused_mul_add_logsoftmax_dropout_bmm_bwd_dx(
        input1_ptr, input2_ptr, other_ptr, mat2_ptr, dout
