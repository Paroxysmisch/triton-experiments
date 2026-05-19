.Tensor): output tensor
        lora_indices_tensor (torch.Tensor): (batch_size,). The LoRA index
            corresponding to each batch. An index of -1 means no lora should be
            applied.
        add_inputs (bool, optional): adds the final lora results to the input
            Defaults to True.
    """
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]
    assert inputs.size(1) == lora_b_weights.size(-1)
    assert inputs.is_contiguous()
    assert lora_b_weights.is_contiguous()

    if output_tensor is None:
        output_tensor = torch.empty_like(inputs)
    else:
        assert output_tensor.is_contiguous()
        assert output_tensor.shape == inputs.shape

    BATCH, K = inputs.shape
    N, K = lora_b_weights.shape
    assert K % 8 == 0, "Hidden size must be divisible by 8"
    # type cast
    CAST_TYPE = False
    if inputs.dtype == torch.float32 and lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]:
        CAST_TYPE = True
    # split n reduce memory
    SPLIT_N = 16
    if N < 1024:
        SPLIT_N = triton.next_power_of_2(N)
    if SPLIT_N >= 16:
        BLOCK_N = SPLIT_N
    else:
        BLOCK_N = N
    # Make sure hidden_size could be evenly divided by BLOCK_N*BLOCK_K
    EVEN_K = K % (8 * BLOCK_N) == 0
    # 16 can be replaced by any power-of-two
    BLOCK_K = min(16, triton.next_power_of_2(K))
    grid = (triton.cdiv(N, BLOCK_N), BATCH)
    _bgmv_expand_kernel[grid](
        inputs,
        lora_b_weights,
        output_tensor,
        N,
        K,
        lora_indices_tensor,
        inputs.stride(0),
        inputs.stride(1),
        lora_b_weights.stride(0),
        lora_b_weights.stride(1),
        lora_b_weights.stride(2),
        output_tensor.stride(0),
        output_tensor.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        SPLIT_N=SPLIT_N,
        EVEN_K=EVEN_K,
        ADD_INPUTS=add_inputs,
        CAST_TYPE=CAST_TYPE,
        num_warps=4,
    )
    return
