import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_slice_kernel(
    input_ptr, lora_ptr, out_ptr,  # Pointers to input/output tensors
    N, K,                          # Matrix dimensions
    lora_indices,                  # Batch-specific LoRA indices
    xm_stride, xk_stride,          # Input strides
    l0_stride, lora_k_stride, lora_n_stride,  # LoRA weight strides
    cm_stride, cn_stride,          # Output strides
    slice_offset,                  # Output slice offset
    BLOCK_N: tl.constexpr,        # Block size for N dimension
    BLOCK_K: tl.constexpr,        # Block size for K dimension
    SPLIT_N: tl.constexpr,        # Number of splits for N dimension
    EVEN_K: tl.constexpr,         # Whether K dimension is evenly divisible
    ADD_INPUTS: tl.constexpr,     # Whether to add to existing outputs
    CAST_TYPE: tl.constexpr,      # Whether to perform type casting
):
    # Get program IDs for batch and split dimension
    pid_sn = tl.program_id(axis=0)
    cur_batch = tl.program_id(axis=1)
    
    # Load LoRA index for current batch
    lora_index = tl.load(lora_indices + cur_batch)
    if lora_index == -1:
        return
        
    # Create offset arrays for blocked computation
    offset_k = tl.arange(0, BLOCK_K)
    offset_n = tl.arange(0, BLOCK_N)
    
    # Load input vector with handling for uneven K
    if EVEN_K:
        tiled_a = tl.load(input_ptr + cur_batch * xm_stride + offset_k * xk_stride)
    else:
        tiled_a = tl.load(
            input_ptr + cur_batch * xm_stride + offset_k * xk_stride,
            mask=offset_k < K,
            other=0,
        )
    
    # Calculate split length and perform type casting if needed
    split_n_length = tl.cdiv(N, SPLIT_N)
    if CAST_TYPE:
        tiled_a = tiled_a.to(lora_ptr.dtype.element_ty)
    
    # Calculate base pointers for LoRA weights and output
    b_ptr = lora_ptr + l0_stride * lora_index + pid_sn * split_n_length * lora_k_stride
    c_ptr = out_ptr + cur_batch * cm_stride + pid_sn * split_n_length + slice_offset * cn_stride
    
    # Process blocks along N dimension
    for n in range(0, split_n_length, BLOCK_N):
        current_n = n + offset_n
        
        # Create masks for boundary handling
        b_ptr_mask = (current_n[:, None] < split_n_length) & (offset_k[None, :] < K)
        c_mask = current_n < split_n_length
        
        # Load LoRA weights
        tiled_b = tl.load(
            b_ptr + current_n[:, None] * lora_k_stride + offset_k[None, :] * lora_n_stride,
            mask=b_ptr_mask,
            other=0.0,
        )
        
        # Compute matrix-vector product and handle accumulation
        if ADD_INPUTS:
            tiled_out = tl.load(c_ptr + current_n * cn_stride, mask=c_mask)
            accumulator = tl.sum(tiled_a * tiled_b, 1) + tiled_out
        else:
            accumulator = tl.sum(tiled_a * tiled_b, 1)
            
        # Store results
        tl.store(c_ptr + current_n * cn_stride, accumulator, mask=c_mask)

@torch.inference_mode()
def _bgmv_expand_slice(
    inputs: torch.Tensor,
    lora_b_weights: torch.Tensor,
    output_tensor: torch.Tensor,
    lora_indices_tensor: torch.Tensor,
    slice_offset: int,
    slice_size: int,
    add_inputs: bool = True,
) -> None:
    # Input validation
    assert inputs.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora_b_weights.dtype in [torch.float16, torch.bfloat16]
    assert inputs.size(1) == lora_b_weights.size(-1)
    assert slice_size == lora_b_weights.size(-2)
    assert inputs.is_contiguous()
    assert output_tensor.is_contiguous()
    
    # Handle LoRA weights dimensionality
    if lora_b_weights.ndim == 4:
        assert lora_b_weights.size(1) == 1
        lora_b_weights = lora_b_weights.squeeze(dim=1)
    else:
        assert lora_b_weights.ndim == 3
    
    assert lora_b_weights.is_contiguous()
    
    # Get dimensions and compute block sizes
    N, K = lora_b_weights.shape[-2:]
    BLOCK_K = triton.next_power_of_2(K)
    EVEN_K = K % BLOCK_K == 0
    ADD_INPUTS = add_inputs
    CAST_TYPE = inputs.dtype == torch.float32 and lora_b_weights.dtype in [torch.float16, torch.bfloat16]
    
    batches = lora_indices_tensor.size(0)
    
    # Get configuration based on problem size
    config = get_lora_op_configs("expand", batches, N)
    
    # Define grid for kernel launch
    grid = lambda META: (META["SPLIT_N"], batches)
    
    # Launch kernel
    _bgmv_expand_slice_kernel[grid](
        inputs,
        lora_b_weights,
        output_tensor,
        N, K,
        lora_indices_tensor,
        inputs.stride(0),
        inputs.stride(1),
        lora_b_weights.stride(0),
        lora_b_weights.stride(1),
        lora_b_weights.stride(2),
        output_tensor.stride(0),
        output_tensor.stride(1),
        slice_offset,
        BLOCK_K=BLOCK_K,
        EVEN_K=EVEN_K,
        ADD_INPUTS=ADD_INPUTS,
        CAST_TYPE=CAST_TYPE,
        **config,
    )
