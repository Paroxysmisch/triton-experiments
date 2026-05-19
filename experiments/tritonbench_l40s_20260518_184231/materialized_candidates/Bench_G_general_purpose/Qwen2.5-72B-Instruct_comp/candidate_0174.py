import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att1(
    Q, K, B_Loc, B_Start_Loc, B_Seqlen, Att_Out,
    stride_qb, stride_qh, stride_qm, stride_kb, stride_kh, stride_km,
    stride_bo, stride_bsl, stride_bs, stride_ao, stride_ah, stride_am,
    nheads, max_input_len, sm_scale, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(max_input_len, BLOCK_M)
    num_pid_n = tl.cdiv(max_input_len, BLOCK_N)
    num_pid_in_batch = num_pid_m * num_pid_n
    batch_idx = pid // num_pid_in_batch
    pid %= num_pid_in_batch
    pid_m = tl.cdiv(pid, num_pid_n)
    pid_n = pid % num_pid_n

    # Compute the block pointers
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    Q_block_ptr = Q + (batch_idx * stride_qb + tl.arange(0, nheads) * stride_qh)[:, None] + offs_m[None, :]
    K_block_ptr = K + (batch_idx * stride_kb + tl.arange(0, nheads) * stride_kh)[:, None] + offs_n[None, :]

    # Load the query and key blocks
    Q_block = tl.load(Q_block_ptr, mask=offs_m[None, :] < max_input_len, other=0.0)
    K_block = tl.load(K_block_ptr, mask=offs_n[None, :] < max_input_len, other=0.0)

    # Compute the dot product
    att = tl.dot(Q_block, K_block, trans_b=True)
    att *= sm_scale

    # Store the result
    Att_Out_block_ptr = Att_Out + (batch_idx * stride_ao + tl.arange(0, nheads) * stride_ah)[:, None] + offs_m[None, :] * stride_am + offs_n[None, :]
    tl.store(Att_Out_block_ptr, att, mask=(offs_m[None, :] < max_input_len) & (offs_n[None, :] < max_input_len))

import triton
import triton.language as tl
import torch

def token_att_fwd(Q, K, B_Loc, B_Start_Loc, B_Seqlen, max_input_len, nheads, sm_scale, BLOCK_M=16, BLOCK_N=16, num_warps=4):
    # Check input dimensions
    assert Q.shape[0] == K.shape[0], "Batch sizes of Q and K must be the same"
    assert Q.shape[1] == K.shape[1], "Number of heads in Q and K must be the same"
    assert Q.shape[2] == max_input_len, "Query sequence length must match max_input_len"
    assert K.shape[2] == max_input_len, "Key sequence length must match max_input_len"

    # Adjust scaling factor
    sm_scale = sm_scale / Q.shape[-1] ** 0.5

    # Initialize output tensor
    Att_Out = torch.empty((Q.shape[0], Q.shape[1], max_input_len, max_input_len), device=Q.device, dtype=Q.dtype)

    # Define grid configuration
    grid = (Q.shape[0] * Q.shape[1] * (max_input_len // BLOCK_M) * (max_input_len // BLOCK_N),)

    # Launch the kernel
    _fwd_kernel_token_att1[grid](
        Q, K, B_Loc, B_Start_Loc, B_Seqlen, Att_Out,
        Q.stride(0), Q.stride(1), Q.stride(2), K.stride(0), K.stride(1), K.stride(2),
        B_Loc.stride(0), B_Start_Loc.stride(0), B_Seqlen.stride(0),
        Att_Out.stride(0), Att_Out.stride(1), Att_Out.stride(2),
        nheads, max_input_len, sm_scale, BLOCK_M, BLOCK_N
    )

    return Att_Out
