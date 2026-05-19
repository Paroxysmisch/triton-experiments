import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid_batch_norm_kernel(
    input_ptr, output_ptr,
    mean_ptr, var_ptr,
    weight_ptr, bias_ptr,
    batch_size, num_channels, seq_length,
    eps,
    BLOCK_SIZE: tl.constexpr
):
    # Get program ID
    pid = tl.program_id(0)
    
    # Calculate starting offset for this program instance
    offset = pid * BLOCK_SIZE
    
    # Load input values
    input_block = tl.load(input_ptr + offset, mask=offset < batch_size * num_channels * seq_length)
    
    # Calculate channel index for current element
    channel_idx = (offset // seq_length) % num_channels
    
    # Load channel-specific parameters
    mean = tl.load(mean_ptr + channel_idx)
    var = tl.load(var_ptr + channel_idx)
    
    # Load weight and bias if they exist
    weight = tl.load(weight_ptr + channel_idx) if weight_ptr is not None else 1.0
    bias = tl.load(bias_ptr + channel_idx) if bias_ptr is not None else 0.0
    
    # Normalize input
    normalized = (input_block - mean) / tl.sqrt(var + eps)
    scaled = normalized * weight + bias
    
    # Apply sigmoid activation
    output = 1.0 / (1.0 + tl.exp(-scaled))
    
    # Store result
    tl.store(output_ptr + offset, output, mask=offset < batch_size * num_channels * seq_length)

def sigmoid_batch_norm(input, running_mean, running_var, weight=None, bias=None, 
                      training=False, momentum=0.1, eps=1e-5):
    # Input validation and shape extraction
    assert input.dim() in [2, 3], "Input must be 2D or 3D tensor"
    
    batch_size = input.size(0)
    num_channels = input.size(1)
    seq_length = input.size(2) if input.dim() == 3 else 1
    
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Initialize output tensor
    output = torch.empty_like(input)
    
    # Calculate statistics during training
    if training:
        # Calculate batch statistics
        dims = [0, 2] if input.dim() == 3 else [0]
        batch_mean = input.mean(dim=dims)
        batch_var = input.var(dim=dims, unbiased=False)
        
        # Update running statistics
        running_mean.mul_(1 - momentum).add_(batch_mean * momentum)
        running_var.mul_(1 - momentum).add_(batch_var * momentum)
        
        mean = batch_mean
        var = batch_var
    else:
        mean = running_mean
        var = running_var
    
    # Determine grid and block sizes
    total_elements = batch_size * num_channels * seq_length
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(total_elements, BLOCK_SIZE),)
    
    # Launch kernel
    sigmoid_batch_norm_kernel[grid](
        input, output,
        mean, var,
        weight, bias,
        batch_size, num_channels, seq_length,
        eps,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return output
