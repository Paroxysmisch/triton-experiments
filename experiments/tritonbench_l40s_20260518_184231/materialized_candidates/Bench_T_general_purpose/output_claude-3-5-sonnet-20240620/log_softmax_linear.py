import torch
import triton
import triton.language as tl

@triton.jit
def log_softmax_linear_kernel(
    # Pointers to matrices
    output_ptr, input_ptr, weight_ptr, bias_ptr,
    # Matrix dimensions
    batch_size, in_features, out_features,
    # Strides for the different matrices
    output_stride_batch, output_stride_out,
    input_stride_batch, input_stride_in,
    weight_stride_out, weight_stride_in,
    bias_stride,
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr,
):
    # Program ID
    pid = tl.program_id(0)
    
    # Batch index
    batch_idx = pid // (out_features // BLOCK_SIZE_N)
    # Block index
    block_idx = pid % (out_features // BLOCK_SIZE_N)
    
    # Initialize pointers to input and output
    input_block_ptr = input_ptr + batch_idx * input_stride_batch
    output_block_ptr = output_ptr + batch_idx * output_stride_batch + block_idx * BLOCK_SIZE_N
    
    # Load the weight matrix block
    weight_block_ptr = weight_ptr + block_idx * BLOCK_SIZE_N * weight_stride_out
    
    # Compute linear transformation
    acc = tl.zeros([BLOCK_SIZE_N], dtype=tl.float32)
    
    # Matrix multiplication
    for k in range(0, in_features, BLOCK_SIZE_M):
        # Load input and weight blocks
        input_vals = tl.load(input_block_ptr + k * input_stride_in)
        weight_vals = tl.load(weight_block_ptr + k * weight_stride_in)
        
        # Compute partial dot product
        acc += tl.dot(input_vals, weight_vals)
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + block_idx * BLOCK_SIZE_N)
        acc += bias
    
    # Compute max for numerical stability
    max_val = tl.max(acc)
    
    # Compute exp(x - max(x))
    exp_vals = tl.exp(acc - max_val)
    
    # Compute sum of exponentials
    sum_exp = tl.sum(exp_vals)
    
    # Compute log_softmax
    log_softmax_vals = acc - max_val - tl.log(sum_exp)
    
    # Store result
    tl.store(output_block_ptr, log_softmax_vals)

def log_softmax_linear(input, weight, bias=None, dim=-1, dtype=None):
    """
    Applies a linear transformation followed by log_softmax activation.
    
    Args:
        input (Tensor): Input tensor of shape (*, in_features)
        weight (Tensor): Weight matrix of shape (out_features, in_features)
        bias (Tensor, optional): Bias tensor of shape (out_features)
        dim (int): Dimension along which log_softmax will be computed (default: -1)
        dtype (torch.dtype, optional): Desired output dtype
    
    Returns:
        Tensor: Output tensor of shape (*, out_features)
    """
    if dtype is not None:
        input = input.to(dtype)
        weight = weight.to(dtype)
        if bias is not None:
            bias = bias.to(dtype)
    
    # Get dimensions
    in_features = weight.size(1)
    out_features = weight.size(0)
    batch_size = input.numel() // in_features
    
    # Ensure input is properly shaped
    input_reshaped = input.reshape(batch_size, in_features)
    
    # Create output tensor
    output = torch.empty((batch_size, out_features), device=input.device, dtype=input.dtype)
    
    # Configure grid and block sizes
    BLOCK_SIZE_M = 32
    BLOCK_SIZE_N = 32
    grid = (batch_size * (out_features // BLOCK_SIZE_N),)
    
    # Launch kernel
    log_softmax_linear_kernel[grid](
        output, input_reshaped, weight, bias,
        batch_size, in_features, out_features,
        output.stride(0), output.stride(1),
        input_reshaped.stride(0), input_reshaped.stride(1),
        weight.stride(0), weight.stride(1),
        1 if bias is not None else 0,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
    )
    
    # Reshape output to match input dimensions
    output_shape = list(input.shape)
    output_shape[-1] = out_features
    output = output.reshape(output_shape)
    
    return output
