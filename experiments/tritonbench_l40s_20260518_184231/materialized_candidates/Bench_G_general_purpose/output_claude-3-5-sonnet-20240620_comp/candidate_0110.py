import triton
import triton.language as tl
import torch

@triton.jit
def _bgmv_expand_kernel(
    input_ptr, lora_ptr, out_ptr,
    lora_indices_ptr,
    N, K,
    input_batch_stride, input_row_stride,
    lora_batch_stride, lora_row_stride,
    out_batch_stride, out_row_stride,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    ADD_TO_OUTPUT: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    batch_id = pid // tl.cdiv(N, BLOCK_N)
    block_id = pid % tl.cdiv(N, BLOCK_N)

    # Load lora index for this batch
    lora_index = tl.load(lora_indices_ptr + batch_id)
    
    # Early exit if lora_index is -1
    if lora_index == -1:
        return

    # Compute offsets
    input_offset = batch_id * input_batch_stride
    lora_offset = lora_index * lora_batch_stride
    out_offset = batch_id * out_batch_stride + block_id * BLOCK_N

    # Initialize accumulator
    acc = tl.zeros([BLOCK_N], dtype=tl.float32)
    
    # Block pointers
    input_block_ptr = input_ptr + input_offset
    lora_block_ptr = lora_ptr + lora_offset + block_id * BLOCK_N * lora_row_stride

    # Loop over K dimension
    for k in range(0, K, BLOCK_K):
        k_remaining = min(BLOCK_K, K - k)
        
        # Load input block
        i_ptr = input_block_ptr + k * input_row_stride
        i = tl.load(i_ptr + tl.arange(0, k_remaining))
        if CAST_TYPE:
            i = i.to(tl.float32)

        # Load LoRA weights block
        w_ptrs = lora_block_ptr + k * lora_row_stride + tl.arange(0, BLOCK_N)[:, None] * lora_row_stride + tl.arange(0, k_remaining)[None, :]
        w = tl.load(w_ptrs)
        if CAST_TYPE:
            w = w.to(tl.float32)

        # Compute matrix-vector product for this block
        acc += tl.sum(w * i[None, :], axis=1)

    # Write output
    out_ptr = out_ptr + out_offset
    if ADD_TO_OUTPUT:
        acc += tl.load(out_ptr + tl.arange(0, BLOCK_N))
    
    tl.store(out_ptr + tl.arange(0, BLOCK_N), acc)

def _bgmv_expand(input_tensor, lora_weights, lora_indices, output_tensor=None, split_n=1):
    """
    Wrapper function for batched generalized matrix-vector multiplication with LoRA weights.
    
    Args:
        input_tensor: Input tensor of shape [batch_size, K]
        lora_weights: LoRA weights tensor of shape [num_loras, N, K]
        lora_indices: Tensor of indices indicating which LoRA weight to use for each batch
        output_tensor: Optional output tensor of shape [batch_size, N]
        split_n: Number of splits along N dimension for parallel processing
    """
    batch_size = input_tensor.shape[0]
    K = input_tensor.shape[1]
    N = lora_weights.shape[1]
    
    # Ensure inputs are contiguous
    input_tensor = input_tensor.contiguous()
    lora_weights = lora_weights.contiguous()
    
    # Create output tensor if not provided
    if output_tensor is None:
        output_tensor = torch.zeros((batch_size, N), device=input_tensor.device, dtype=input_tensor.dtype)
    else:
        output_tensor = output_tensor.contiguous()

    # Determine optimal block sizes
    BLOCK_K = min(128, triton.next_power_of_2(K))
    BLOCK_N = min(128, triton.next_power_of_2(N // split_n))
    
    # Check if type casting is needed
    CAST_TYPE = input_tensor.dtype != lora_weights.dtype
    
    # Launch kernel
    grid = (triton.cdiv(N, BLOCK_N) * batch_size,)
    _bgmv_expand_kernel[grid](
        input_tensor, lora_weights, output_tensor,
        lora_indices,
        N, K,
        input_tensor.stride(0), input_tensor.stride(1),
        lora_weights.stride(0), lora_weights.stride(1),
        output_tensor.stride(0), output_tensor.stride(1),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        ADD_TO_OUTPUT=output_tensor is not None,
        CAST_TYPE=CAST_TYPE,
    )
    
    return output_tensor
