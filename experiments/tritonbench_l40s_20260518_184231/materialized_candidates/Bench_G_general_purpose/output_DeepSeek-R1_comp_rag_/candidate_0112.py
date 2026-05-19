import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    lora_indices,
    xm_stride,
    xk_stride,
    l0_stride,
    lora_k_stride,
    lora_n_stride,
    cm_stride,
    cn_stride,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    """Optimized batched GEMV with LoRA weight selection and output accumulation."""
    # Determine block indices and batch
    pid_sn = tl.program_id(0)
    cur_batch = tl.program_id(1)
    lora_index = tl.load(lora_indices + cur_batch)
    
    # Skip computation for invalid LoRA indices
    if lora_index == -1:
        return

    # Offsets for block processing
    offset_k = tl.arange(0, BLOCK_K)
    offset_n = tl.arange(0, BLOCK_N)
    
    # Load input tensor block with K dimension handling
    input_block_ptr = input_ptr + cur_batch * xm_stride + offset_k * xk_stride
    if EVEN_K:
        tiled_a = tl.load(input_block_ptr)
    else:
        tiled_a = tl.load(input_block_ptr, mask=offset_k < K, other=0)
    
    # Type casting if required
    if CAST_TYPE:
        tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
    
    # Split N dimension for large hidden sizes
    split_n_length = tl.cdiv(N, SPLIT_N)
    b_ptr_base = lora_ptr + l0_stride * lora_index + pid_sn * split_n_length * lora_k_stride
    c_ptr_base = out_ptr + cur_batch * cm_stride + pid_sn * split_n_length

    # Process blocks in N dimension
    for n in range(0, split_n_length, BLOCK_N):
        current_n = n + offset_n
        current_n_c = tl.max_contiguous(current_n, BLOCK_N)
        mask_n = current_n < split_n_length
        
        # Load LoRA weights with masking
        b_ptr = b_ptr_base + current_n_c[:, None] * lora_k_stride + offset_k[None, :] * lora_n_stride
        b_mask = mask_n[:, None] & (offset_k[None, :] < K)
        tiled_b = tl.load(b_ptr, mask=b_mask, other=0.0)
        
        # Compute and accumulate results
        accumulator = tl.sum(tiled_a * tiled_b, axis=1)
        
        # Handle input addition if required
        if ADD_INPUTS:
            c_ptr = c_ptr_base + current_n * cn_stride
            tiled_out = tl.load(c_ptr, mask=mask_n, other=0.0)
            accumulator += tiled_out
        
        # Store results
        tl.store(c_ptr_base + current_n * cn_stride, accumulator, mask=mask_n)

@torch.inference_mode()
def _bgmv_expand(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    add_inputs: bool = True,
) -> None:
    """Configures and launches the GEMV kernel with LoRA support."""
    # Input validation and tensor preparation
    assert inputs.is_contiguous() and output_tensor.is_contiguous()
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [torch.float16, torch.bfloat16]
    assert inputs.size(1) == lora_b_weights.size(-1)

    # Handle different LoRA weight dimensions
    if lora_b_weights.ndim == 4:
        lora_b_weights = lora_b_weights.squeeze(1)
    N, K = lora_b_weights.shape[-2:]

    # Kernel configuration parameters
    BLOCK_K = triton.next_power_of_2(K)
    EVEN_K = (K % BLOCK_K) == 0
    CAST_TYPE = inputs.dtype == torch.float32 and lora_b_weights.dtype != torch.float32
    batches = lora_indices_tensor.size(0)

    # Grid configuration and kernel launch
    grid = (triton.cdiv(N, 128), batches)  # 128 as default SPLIT_N
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
        BLOCK_K=BLOCK_K,
        EVEN_K=EVEN_K,
        ADD_INPUTS=add_inputs,
        CAST_TYPE=CAST_TYPE,
        SPLIT_N=128,
        BLOCK_N=128,
    )
