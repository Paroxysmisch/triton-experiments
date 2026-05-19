import triton
import triton.language as tl

@triton.jit
def _fwd_kernel_token_att2(
    Prob,
    V,
    Out,
    Req_to_tokens,
    stride_prob_bs,
    stride_prob_h,
    stride_prob_n,
    stride_v_bs,
    stride_v_h,
    stride_v_n,
    stride_out_bs,
    stride_out_h,
    stride_out_n,
    cur_batch: tl.constexpr,
    cur_head: tl.constexpr,
    cur_kv_head: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    # Initialize offsets
    offs_n = tl.arange(0, BLOCK_N)
    offs_m = tl.arange(0, BLOCK_M)
    req_to_tokens = tl.load(Req_to_tokens + cur_batch * BLOCK_M + offs_m)

    # Initialize accumulator
    acc = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    for start_n in range(0, BLOCK_M, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        # Load probability and value slices
        p_value = tl.load(
            Prob + cur_batch * stride_prob_bs + cur_head * stride_prob_h + (req_to_tokens + start_n) * stride_prob_n,
            mask=(req_to_tokens + start_n + offs_n[None, :]) < BLOCK_M,
            other=0.0
        )
        v_value = tl.load(
            V + cur_batch * stride_v_bs + cur_kv_head * stride_v_h + (req_to_tokens + start_n) * stride_v_n,
            mask=(req_to_tokens + start_n + offs_n[None, :]) < BLOCK_M,
            other=0.0
        )

        # Compute and accumulate
        acc += tl.dot(p_value, v_value)

    # Cast and store the result
    out_ptrs = Out + cur_batch * stride_out_bs + cur_head * stride_out_h + req_to_tokens * stride_out_n
    tl.store(out_ptrs, acc, mask=req_to_tokens < BLOCK_M)

import torch

@torch.no_grad()
def token_att_fwd2(prob, v, out, req_to_tokens, b_seq_len, max_input_len, num_heads, kv_group_num):
    # Determine block size based on GPU capability
    if torch.cuda.get_device_capability()[0] >= 8:
        BLOCK = 128
    else:
        BLOCK = 64

    # Compute grid dimensions
    batch, head = b_seq_len.shape[0], num_heads
    grid = (batch, head, triton.cdiv(max_input_len, BLOCK))
    num_warps = 4 if prob.shape[-1] <= 64 else 8

    # Launch the kernel
    _fwd_kernel_token_att2[grid](
        prob,
        v,
        out,
        req_to_tokens,
        prob.stride(0),
        prob.stride(1),
        prob.stride(2),
        v.stride(0),
        v.stride(1),
        v.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        cur_batch=0,
        cur_head=0,
        cur_kv_head=0,
        BLOCK_N=BLOCK,
        BLOCK_M=BLOCK,
        num_warps=num_warps,
        num_stages=1,
    )
