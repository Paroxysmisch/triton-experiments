import triton
import triton.language as tl

# Kernel to perform batched GEMV with LoRA weights
@triton.jit
def _bgmv_expand_kernel(input_ptr, lora_ptr, out_ptr, lora_indices,
                        N, K, stride_in, stride_lora, stride_out,
                        BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
                        CAST_TYPE: tl.constexpr):
    pid = tl.program_id(0)
    batch_id = tl.program_id(1)

    # Determine the range of N to process
    start_n = pid * BLOCK_N
    end_n = min(start_n + BLOCK_N, N)

    # Load lora index for this batch
    lora_index = tl.load(lora_indices + batch_id)
    
    # If lora_index is -1, skip computation
    if lora_index == -1:
        return

    # Pointers to input and output slices
    input_ptr = input_ptr + batch_id * stride_in
    lora_ptr = lora_ptr + lora_index * stride_lora
    out_ptr = out_ptr + batch_id * stride_out

    # Initialize output accumulator
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)

    # Loop over K dimension in blocks
    for k in range(0, K, BLOCK_K):
        # Load input and lora weights
        input_vals = tl.load(input_ptr + k)
        lora_vals = tl.load(lora_ptr + k)

        # Optionally cast input and lora weights
        if CAST_TYPE:
            input_vals = input_vals.to(CAST_TYPE)
            lora_vals = lora_vals.to(CAST_TYPE)

        # Compute partial dot product
        acc += tl.dot(input_vals, lora_vals)

    # Write back to output
    tl.store(out_ptr + start_n, acc)

# Wrapper function to set up and launch the kernel
def _bgmv_expand(input, lora, output, lora_indices, BLOCK_N=128, BLOCK_K=32, CAST_TYPE=None):
    # Ensure input tensors are contiguous
    input = input.contiguous()
    lora = lora.contiguous()
    output = output.contiguous()

    # Get dimensions
    B, N, K = input.shape

    # Determine block sizes and grid
    grid = (triton.cdiv(N, BLOCK_N), B)

    # Launch the kernel
    _bgmv_expand_kernel[grid](
        input_ptr=input.data_ptr(),
        lora_ptr=lora.data_ptr(),
        out_ptr=output.data_ptr(),
        lora_indices=lora_indices.data_ptr(),
        N=N, K=K,
        stride_in=input.stride(0),
        stride_lora=lora.stride(0),
        stride_out=output.stride(0),
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        CAST_TYPE=CAST_TYPE
    )
