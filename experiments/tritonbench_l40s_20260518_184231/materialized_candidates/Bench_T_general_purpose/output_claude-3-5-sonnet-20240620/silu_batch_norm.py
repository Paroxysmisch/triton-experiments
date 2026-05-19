import torch
import triton
import triton.language as tl

@triton.jit
def silu_batchnorm_kernel(
    # Pointers to matrices
    x_ptr,          # Input tensor
    mean_ptr,       # Running mean
    var_ptr,        # Running variance
    weight_ptr,     # Optional weight for scaling
    bias_ptr,       # Optional bias
    output_ptr,     # Output tensor
    # Matrix dimensions
    n_elements,     # Total number of elements
    channels,       # Number of channels
    # Parameters
    eps,           # Epsilon for numerical stability
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input elements
    x = tl.load(x_ptr + offsets, mask=mask)
    
    # Calculate channel index for each element
    channel_idx = (offsets % channels).to(tl.int32)
    
    # Load channel-specific parameters
    mean = tl.load(mean_ptr + channel_idx, mask=mask)
    var = tl.load(var_ptr + channel_idx, mask=mask)
    
    # Apply batch normalization
    x_normalized = (x - mean) / tl.sqrt(var + eps)
    
    # Apply weight and bias if provided
    if weight_ptr is not None:
        weight = tl.load(weight_ptr + channel_idx, mask=mask)
        x_normalized = x_normalized * weight
    
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + channel_idx, mask=mask)
        x_normalized = x_normalized + bias
    
    # Apply SiLU activation: x * sigmoid(x)
    sigmoid_x = 1 / (1 + tl.exp(-x_normalized))
    output = x_normalized * sigmoid_x
    
    # Store result
    tl.store(output_ptr + offsets, output, mask=mask)

def silu_batch_norm(input, running_mean, running_var, weight=None, bias=None, 
                   training=False, momentum=0.1, eps=1e-5):
    """
    Applies Batch Normalization followed by SiLU activation.
    
    Args:
        input (Tensor): Input tensor to be normalized
        running_mean (Tensor): Running mean used for inference mode
        running_var (Tensor): Running variance used for inference mode
        weight (Tensor, optional): Weight tensor for scaling
        bias (Tensor, optional): Bias tensor for shifting
        training (bool, optional): Whether in training mode. Default: False
        momentum (float, optional): Momentum for running stats. Default: 0.1
        eps (float, optional): Small value for numerical stability. Default: 1e-5
    
    Returns:
        Tensor: The normalized and activated output tensor
    """
    if training:
        # Calculate batch statistics
        dims = [0] + list(range(2, input.dim()))
        batch_mean = input.mean(dims)
        batch_var = input.var(dims, unbiased=False)
        
        # Update running statistics
        running_mean.mul_(1 - momentum).add_(batch_mean * momentum)
        running_var.mul_(1 - momentum).add_(batch_var * momentum)
    else:
        batch_mean = running_mean
        batch_var = running_var
    
    # Prepare output tensor
    output = torch.empty_like(input)
    
    # Grid and block sizes
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(input.numel(), BLOCK_SIZE),)
    
    # Launch kernel
    silu_batchnorm_kernel[grid](
        input.contiguous().data_ptr(),
        batch_mean.contiguous().data_ptr(),
        batch_var.contiguous().data_ptr(),
        weight.contiguous().data_ptr() if weight is not None else None,
        bias.contiguous().data_ptr() if bias is not None else None,
        output.contiguous().data_ptr(),
        input.numel(),
        input.size(1),  # Number of channels
        eps,
        BLOCK_SIZE,
    )
    
    return output
