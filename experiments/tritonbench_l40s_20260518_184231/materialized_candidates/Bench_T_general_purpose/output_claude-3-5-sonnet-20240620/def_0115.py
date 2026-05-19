import torch
import triton
import triton.language as tl

@triton.jit
def tanh_linear_kernel(
    # Pointers to matrices
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    # Matrix dimensions
    batch_size,
    in_features,
    out_features,
    # Strides for memory access
    input_batch_stride,
    input_feature_stride,
    weight_out_stride,
    weight_in_stride,
    output_batch_stride,
    output_feature_stride,
    # Whether bias is provided
    USE_BIAS: tl.constexpr,
    # Block sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Batch and output feature indices
    batch_idx = pid // (out_features // BLOCK_SIZE_M)
    out_feature_start = (pid % (out_features // BLOCK_SIZE_M)) * BLOCK_SIZE_M

    # Pointers to current batch
    input_batch_ptr = input_ptr + batch_idx * input_batch_stride
    output_batch_ptr = output_ptr + batch_idx * output_batch_stride

    # Initialize accumulator
    acc = tl.zeros([BLOCK_SIZE_M], dtype=tl.float32)

    # Iterate over input features in blocks
    for k in range(0, in_features, BLOCK_SIZE_K):
        # Load input block
        input_block = tl.load(input_batch_ptr + k * input_feature_stride)
        
        # Load weight block
        weight_block = tl.load(
            weight_ptr + 
            out_feature_start[:, None] * weight_out_stride +
            (k + tl.arange(0, BLOCK_SIZE_K)[None, :]) * weight_in_stride
        )
        
        # Matrix multiplication
        acc += tl.dot(weight_block, input_block)

    # Add bias if provided
    if USE_BIAS:
        bias = tl.load(bias_ptr + out_feature_start)
        acc += bias

    # Apply tanh activation
    acc = tl.where(acc != float('inf'), tl.exp(2 * acc), float('inf'))
    acc = (acc - 1) / (acc + 1)

    # Store result
    tl.store(
        output_batch_ptr + out_feature_start * output_feature_stride,
        acc
    )

def tanh_linear(input: torch.Tensor, 
                weight: torch.Tensor, 
                bias: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Applies a linear transformation followed by tanh activation.
    
    Args:
        input: Input tensor of shape (*, in_features)
        weight: Weight matrix of shape (out_features, in_features)
        bias: Optional bias tensor of shape (out_features)
    
    Returns:
        Output tensor of shape (*, out_features)
    """
    assert input.dim() >= 2, "Input tensor must have at least 2 dimensions"
    assert weight.dim() == 2, "Weight matrix must be 2-dimensional"
    
    # Extract dimensions
    in_features = weight.size(1)
    out_features = weight.size(0)
    batch_dims = input.shape[:-1]
    batch_size = input.numel() // in_features

    # Reshape input to 2D if needed
    input_2d = input.reshape(batch_size, in_features)
    
    # Prepare output tensor
    output = torch.empty(batch_size, out_features, 
                        device=input.device, dtype=input.dtype)

    # Define grid and block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 1
    BLOCK_SIZE_K = 32
    
    grid = (triton.cdiv(batch_size * out_features, BLOCK_SIZE_M),)

    # Launch kernel
    tanh_linear_kernel[grid](
        input_2d.contiguous().data_ptr(),
        weight.contiguous().data_ptr(),
        bias.contiguous().data_ptr() if bias is not None else None,
        output.data_ptr(),
        batch_size,
        in_features,
        out_features,
        input_2d.stride(0),
        input_2d.stride(1),
        weight.stride(0),
        weight.stride(1),
        output.stride(0),
        output.stride(1),
        bias is not None,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )

    # Reshape output back to match input dimensions
    return output.reshape(*batch_dims, out_features)
