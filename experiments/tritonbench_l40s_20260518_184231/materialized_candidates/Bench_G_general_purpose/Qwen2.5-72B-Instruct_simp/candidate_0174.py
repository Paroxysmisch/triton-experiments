import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att1(
    q_ptr, k_ptr, att_out_ptr,
    B_Loc, B_Start_Loc, B_Seqlen,
    stride_qbs, stride_qd,
    stride_kbs, stride_kd,
    stride_att_bs, stride_att_n,
    max_input_len: tl.constexpr,
    n_head: tl.constexpr,
    head_dim: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr
):
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(max_input_len, BLOCK_M)
    num_pid_n = tl.cdiv(max_input_len, BLOCK_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_pid = pid // num_pid_in_batch
    pid = pid % num_pid_in_batch
    nm = pid % num_pid_m
    no = pid // num_pid_m

    # Offsets for q and k
    off_m = nm * BLOCK_M + tl.arange(0, BLOCK_M)
    off_n = no * BLOCK_N + tl.arange(0, BLOCK_N)
    off_d = tl.arange(0, BLOCK_DMODEL)

    # Offsets for q and k
    q_offsets = (batch_pid * stride_qbs + off_m[:, None] * stride_qd + off_d[None, :])
    k_offsets = (batch_pid * stride_kbs + off_n[:, None] * stride_kd + off_d[None, :])

    # Load q and k
    q = tl.load(q_ptr + q_offsets)
    k = tl.load(k_ptr + k_offsets)

    # Compute attention scores
    att_scores = tl.sum(q * k, axis=1) / tl.sqrt(head_dim)

    # Offsets for att_out
    att_offsets = (batch_pid * stride_att_bs + off_m * stride_att_n + off_n)
    tl.store(att_out_ptr + att_offsets, att_scores)

import torch

def token_att_fwd(q, k, B_Loc, B_Start_Loc, B_Seqlen, max_input_len, att_out):
    # Get tensor dimensions
    batch_size, n_head, seq_len, head_dim = q.shape

    # Define grid and block sizes
    BLOCK_M = 16
    BLOCK_N = 16
    BLOCK_DMODEL = head_dim
    grid = (batch_size * (seq_len // BLOCK_M) * (seq_len // BLOCK_N),)

    # Launch the kernel
    _fwd_kernel_token_att1[grid](
        q, k, att_out,
        B_Loc, B_Start_Loc, B_Seqlen,
        q.stride(0), q.stride(2),
        k.stride(0), k.stride(2),
        att_out.stride(0), att_out.stride(1),
        max_input_len,
        n_head,
        head_dim,
        BLOCK_M,
        BLOCK_N,
        BLOCK_DMODEL
    )

    return att_out
