import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_kernel(
    # Pointers to matrices
    input_ptr,          # Input matrix pointer
    weight_ptr,         # LoRA weights pointer
    output_ptr,         # Output matrix pointer
    # Matrix dimensions
    batch_size,         # Batch size
    in_features,        # Input feature dimension
    out_features,       # Output feature dimension
    # Strides
    input_batch_stride,
    input_row_stride,
    weight_batch_stride,
    weight_row_stride,
    output_batch_stride,
    output_row_stride,
    # Options
    ADD_INPUTS: tl.constexpr,  # Whether to add inputs to outputs
    DTYPE_IN: tl.constexpr,    # Input data type
    DTYPE_OUT: tl.constexpr,   # Output data type
    BLOCK_SIZE: tl.constexpr,  # Block size for tiling
):
    # Program ID
    pid = tl.program_id(0)
    batch_id = pid // out_features
    out_feature_id = pid % out_features

    # Compute pointers
    input_batch_ptr = input_ptr + batch_id * input_batch_stride
    weight_batch_ptr = weight_ptr + batch_id * weight_batch_stride
    output_batch_ptr = output_ptr + batch_id * output_batch_stride

    # Initialize accumulator
    acc = 0.0

    # Load input and weight vectors and compute dot product
    for block_start in range(0, in_features, BLOCK_SIZE):
        block_end = min(block_start + BLOCK_SIZE, in_features)
        
        # Create mask for valid elements
        mask = tl.arange(0, BLOCK_SIZE) < (block_end - block_start)
        
        # Load input and weight blocks
        input_block = tl.load(
            input_batch_ptr + (block_start + tl.arange(0, BLOCK_SIZE)) * input_row_stride,
            mask=mask, other=0.0
        )
        weight_block = tl.load(
            weight_batch_ptr + out_feature_id * weight_row_stride + 
            (block_start + tl.arange(0, BLOCK_SIZE)),
            mask=mask, other=0.0
        )
        
        # Accumulate dot product
        acc += tl.sum(input_block * weight_block, 0)

    # Write output
    output_idx = output_batch_ptr + out_feature_id * output_row_stride
    if ADD_INPUTS:
        current_output = tl.load(output_idx)
        acc = acc + current_output

    # Store result
    tl.store(output_idx, acc.to(DTYPE_OUT))

@torch.inference_mode()
def _bgmv_expand(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    add_inputs: bool = False
) -> None:
    # Get dimensions
    batch_size = input_tensor.shape[0]
    in_features = input_tensor.shape[1]
    out_features = weight_tensor.shape[1]

    # Compute strides
    input_batch_stride = input_tensor.stride(0)
    input_row_stride = input_tensor.stride(1)
    weight_batch_stride = weight_tensor.stride(0)
    weight_row_stride = weight_tensor.stride(1)
    output_batch_stride = output_tensor.stride(0)
    output_row_stride = output_tensor.stride(1)

    # Determine block size (can be tuned)
    BLOCK_SIZE = 128

    # Launch kernel
    grid = (batch_size * out_features,)
    _bgmv_expand_kernel[grid](
        input_tensor.data_ptr(),
        weight_tensor.data_ptr(),
        output_tensor.data_ptr(),
        batch_size,
        in_features,
        out_features,
        input_batch_stride,
        input_row_stride,
        weight_batch_stride,
        weight_row_stride,
        output_batch_stride,
        output_row_stride,
        ADD_INPUTS=add_inputs,
        DTYPE_IN=input_tensor.dtype,
        DTYPE_OUT=output_tensor.dtype,
        BLOCK_SIZE=BLOCK_SIZE,
    )
