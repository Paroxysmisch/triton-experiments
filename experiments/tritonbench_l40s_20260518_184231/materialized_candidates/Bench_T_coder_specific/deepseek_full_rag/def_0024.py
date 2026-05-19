ulator)
        else:
            accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk

    if MUL_ROUTED_WEIGHT:
        moe_weight = tl.load(topk_weights_ptr + offs_token, mask=token_mask, other=0)
        accumulator = accumulator * moe_weight[:, None]

    if use_fp8:
        accumulator = (accumulator * a_scale * b_scale).to(compute_type)
    else:
        accumulator = accumulator.to(compute_type)

    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_token[:, None] + stride_cn * offs_cn[None, :]
    c_mask = token_mask[:, None] & (offs_cn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

def invoke_fused_moe_kernel(
    A: torch.Tensor,
    B: torch.Tensor,
    C: torch.Tensor,
    a_scale: Optional[torch.Tensor],
    b_scale: Optional[torch.Tensor],
    topk_weights: torch.Tensor,
    sorted_token_ids: torch.Tensor,
    expert_ids: torch.Tensor,
    num_tokens_post_padded: torch.Tensor,
    top_k: int,
    config: Dict[str, Any],
    use_fp8: bool,
):
    assert top_k >= 1
    assert B.shape[1] == A.shape[0]
    assert B.shape[2] == A.shape[1]
    assert B.shape[0] == sorted_token_ids.shape[1]
    assert sorted_token_ids.shape[0] == A.shape[0]
    assert C.shape[0] == sorted_token_ids.shape[0]
    assert C.shape[1] == top_k
    assert C.shape[2] == B.shape[1]
    assert A.is_contiguous()
    assert B.is_contiguous()
    assert C.is_contiguous()
    assert sorted_token_ids.is_contiguous()
    assert expert_ids.is_contiguous()
    if use_fp8:
        assert a_scale is not None
        assert b_scale is not None
    else:
        assert a_scale is None
        assert b_scale is None
    num_valid_tokens = A.shape[0]
    N = B.shape[1]
    K = B.shape[2]
    EM = B.shape[0]
    grid = lambda META: (
        triton.cdiv(EM, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    fused_moe_kernel[grid](
        A,
        B,
        C,
        a_scale,
        b_scale,
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
        C.stride(1),
        use_fp8=use_fp8,
        **config,
    )
