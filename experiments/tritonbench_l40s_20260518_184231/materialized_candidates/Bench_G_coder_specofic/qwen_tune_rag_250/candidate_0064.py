2)
    # prepare for triton
    if inputs.dtype == torch.float32 and lora_b_weights.dtype in [
        torch.float16,
        torch.bfloat16,
    ]:
        inputs = inputs.to(lora_b_weights.dtype)
    # -------------------
    # | block    | block |
    # |         0,0         |
    # |         0,1         |
    # |                     |
    # | block m,n -> (m*n, 1)|
    # -------------------
    #  0,0  0,1  ... m,n
    # BLOCK_N * BLOCK_K
    BLOCK_N = 32
    BLOCK_K = 32
    # how many block_N in a group
    SPLIT_N = 8
    EVEN_K = lora_b_weights.size(-1) % BLOCK_K == 0
    # -------------------
    # | block    | block |
    # |         0,0         |
    # |         0,1         |
    # |                     |
    # | block m,n -> (1, m*n)|
    # -------------------
    #  0,0  0,1  ... m,n
    # 1,0  1,1  ... m,n
    #  m,n
    # SPLIT_K = 4
    # EVEN_N = lora_b_weights.size(-2) % BLOCK_N == 0
    # prepare kernel
    # here we can use 2d launch grid, since triton kernel is designed for this
    grid = (
        triton.cdiv(lora_b_weights.size(-2), BLOCK_N * SPLIT_N),
        lora_indices_tensor.size(0),
    )
    meta = {
        "BLOCK_N": BLOCK_N,
        "BLOCK_K": BLOCK_K,
        "SPLIT_N": SPLIT_N,
        "EVEN_K": EVEN_K,
        "ADD_INPUTS": add_inputs,
        "CAST_TYPE": False,
    }
    if inputs.dtype == torch.float32 and lora_b_weights.dtype == torch.float16:
        meta["CAST_TYPE"] = True
    # enqueue kernel
    _bgmv_expand_slice_kernel[grid](
        inputs,
        lora_b_weights,
        output_tensor,
        lora_indices_tensor,
        inputs.stride(0),
        inputs.stride(1),
        lora_b_weights.stride(0),
        lora_b_weights.stride(1),
        lora_b_weights.stride(2),
        output_tensor.stride(0),
        output_tensor.stride(1),
        slice_offset=slice_offset,
        **meta,
    )
    return
