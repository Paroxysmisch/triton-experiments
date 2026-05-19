import triton
import triton.language as tl
import torch

@triton.jit
def _bgmv_expand_slice_kernel(
    # Pointers to matrices
    input_ptr,          # Input tensor pointer
    weight_ptr,         # LoRA weight tensor pointer
    output_ptr,        # Output tensor pointer
    # Matrix dimensions
    batch_size,        # Batch size
    in_features,       # Input feature dimension
    out_features,      # Output feature dimension
    # Strides for tensors
    input_batch_stride,
    input_row_stride,
    weight_row_stride,
    output_batch_stride,
    output_row_stride,
    # Additional parameters
    block_size: tl.constexpr,  # Size of block for computation
    input_dtype: tl.constexpr, # Input data type
    weight_dtype: tl.constexpr, # Weight data type
    output_dtype: tl.constexpr, # Output data type
    accumulate: tl.constexpr,  # Whether to accumulate results
):
    # Program ID
    pid = tl.program_id(0)
    
    # Calculate batch and feature indices
    batch_id = pid // ((out_features + block_size - 1) // block_size)
    feature_id = (pid % ((out_features + block_size - 1) // block_size)) * block_size
    
    # Compute pointers
    input_offset = batch_id * input_batch_stride
    weight_offset = feature_id * weight_row_stride
    output_offset = batch_id * output_batch_stride + feature_id * output_row_stride
    
    # Load input vector for this batch
    input_block = tl.load(
        input_ptr + input_offset + tl.arange(0, in_features) * input_row_stride,
        mask=tl.arange(0, in_features) < in_features,
        dtype=input_dtype
    )
    
    # Initialize accumulator
    acc = tl.zeros([block_size], dtype=tl.float32)
    
    # Main matrix multiplication loop
    for i in range(0, in_features, block_size):
        # Load weight matrix block
        weight_block = tl.load(
            weight_ptr + weight_offset + tl.arange(0, block_size)[:, None] * weight_row_stride + tl.arange(0, block_size)[None, :],
            mask=(feature_id + tl.arange(0, block_size)[:, None] < out_features) & (i + tl.arange(0, block_size)[None, :] < in_features),
            dtype=weight_dtype
        )
        
        # Perform matrix-vector multiplication
        acc += tl.dot(weight_block, input_block[i:i + block_size])
    
    # Handle accumulation if required
    if accumulate:
        existing_output = tl.load(
            output_ptr + output_offset + tl.arange(0, block_size),
            mask=feature_id + tl.arange(0, block_size) < out_features,
            dtype=output_dtype
        )
        acc += existing_output
    
    # Store results
    tl.store(
        output_ptr + output_offset + tl.arange(0, block_size),
        acc,
        mask=feature_id + tl.arange(0, block_size) < out_features
    )

def _bgmv_expand_slice(
    input_tensor: torch.Tensor,
    weight_tensor: torch.Tensor,
    output_tensor: torch.Tensor,
    accumulate: bool = False
) -> None:
    """
    Wrapper function for the batched Generalized Matrix-Vector Multiply kernel.
    
    Args:
        input_tensor: Input tensor of shape [batch_size, in_features]
        weight_tensor: LoRA weight tensor of shape [out_features, in_features]
        output_tensor: Output tensor of shape [batch_size, out_features]
        accumulate: Whether to accumulate results in output tensor
    """
    batch_size, in_features = input_tensor.shape
    out_features = weight_tensor.shape[0]
    
    # Define block size (can be tuned for performance)
    BLOCK_SIZE = 32
    
    # Calculate grid size
    grid = (batch_size * ((out_features + BLOCK_SIZE - 1) // BLOCK_SIZE),)
    
    # Launch kernel
    _bgmv_expand_slice_kernel[grid](
        input_tensor,
        weight_tensor,
        output_tensor,
        batch_size,
        in_features,
        out_features,
        input_tensor.stride(0),
        input_tensor.stride(1),
        weight_tensor.stride(0),
        output_tensor.stride(0),
        output_tensor.stride(1),
        BLOCK_SIZE,
        input_tensor.dtype,
        weight_tensor.dtype,
        output_tensor.dtype,
        accumulate,
    )
