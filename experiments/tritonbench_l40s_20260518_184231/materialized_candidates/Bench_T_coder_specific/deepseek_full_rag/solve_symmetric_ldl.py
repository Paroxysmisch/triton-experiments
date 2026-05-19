)
        if use_fp8:
            accumulator = tl.dot(a, b, acc=accumulator)
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(
            topk_weights_ptr + offs_token, mask=token_mask, other=0.0
        ).to(compute_type)
        accumulator = accumulator * moe_weight[:, None]

    if use_fp8:
        accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def invoke_fused_moe_kernel(A, B, C, A_scale, B_scale, topk_weights, sorted_token_ids, expert_ids, num_tokens_post_padded, top_k, max_experts, use_fp8=False):
    """
    Calls the fused_moe_kernel with the appropriate parameters and settings.
    """
    if top_k > 1:
        sorted_token_ids = sorted_token_ids.repeat_interleave(top_k, dim=0)

    if A.stride(-1) != 1:
        A = A.contiguous()
    if B.stride(-1) != 1:
        B = B.contiguous()

    if use_fp8:
        assert A.dtype == torch.float8_e4m3fn or A.dtype == torch.bfloat16
        assert B.dtype == torch.float8_e4m3fn
        compute_type = torch.float8_e4m3fn if A.dtype == torch.float8_e4m3fn else torch.bfloat16
    else:
        assert A.dtype == torch.bfloat16
        assert B.dtype == torch.float16 or B.dtype == torch.bfloat16
        compute_type = torch.bfloat16 if A.dtype == torch.bfloat16 else torch.float16

    if C.stride(-1) != 1:
        C = C.contiguous()

    num_valid_tokens = A.shape[0]
    N, K = B.shape[1:]
    EM = expert_ids.numel()

    grid = lambda META: (
        triton.cdiv(num_valid_tokens, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]) * META["GROUP_SIZE_M"],
    )

    fused_moe_kernel[grid](
        A,
        B,
        C,
        A_scale,
        B_scale,
        topk_weights,
        sorted_token_ids,
        expert_ids,
        num_tokens_post_padded,
        N,
        K,
        EM,
        num_valid_tokens,
        A.stride(0),
        A.stride(1),
        B.stride(0),
        B.stride(1),
        B.stride(2),
        C.stride(0),
        C.stride(2),
        MUL_ROUTED_WEIGHT=True,
        top_k=top_k,
        compute_type=compute_type,
        use_fp8=use_fp8,
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
    )
