import torch
import triton
import triton.language as tl

# Triton kernel for batch normalization
@triton.jit
def batch_norm_kernel(
    input_ptr,  # Pointer to input tensor
    output_ptr,  # Pointer to output tensor
    running_mean_ptr,  # Pointer to running mean
    running_var_ptr,  # Pointer to running variance
    weight_ptr,  # Pointer to weight (scale)
    bias_ptr,  # Pointer to bias (shift)
    batch_size,  # Number of batches
    channels,  # Number of channels
    height,  # Height of the input tensor
    width,  # Width of the input tensor
    eps,  # Epsilon value for numerical stability
    momentum,  # Momentum for running mean and variance
    training,  # Whether to use training mode
    BLOCK_SIZE: tl.constexpr  # Number of elements each program should process
):
    # Identify the program ID
    pid = tl.program_id(axis=0)
    
    # Calculate the channel index for this program
    channel = pid % channels
    batch = pid // channels
    
    # Calculate the offsets for the input and output tensors
    input_offset = (batch * channels + channel) * height * width
    output_offset = input_offset
    
    # Create a mask to guard memory operations against out-of-bounds accesses
    offsets = input_offset + tl.arange(0, BLOCK_SIZE)
    mask = offsets < (batch_size * channels * height * width)
    
    # Load input data
    input_data = tl.load(input_ptr + offsets, mask=mask)
    
    # Load running mean and variance
    running_mean = tl.load(running_mean_ptr + channel)
    running_var = tl.load(running_var_ptr + channel)
    
    # Load weight and bias if provided
    weight = tl.load(weight_ptr + channel) if weight_ptr is not None else 1.0
    bias = tl.load(bias_ptr + channel) if bias_ptr is not None else 0.0
    
    # Compute the batch mean and variance if in training mode
    if training:
        batch_mean = tl.sum(input_data, axis=0) / (height * width)
        batch_var = tl.sum((input_data - batch_mean) ** 2, axis=0) / (height * width)
        
        # Update running mean and variance
        running_mean = (1 - momentum) * running_mean + momentum * batch_mean
        running_var = (1 - momentum) * running_var + momentum * batch_var
        
        # Save updated running mean and variance
        tl.store(running_mean_ptr + channel, running_mean)
        tl.store(running_var_ptr + channel, running_var)
    
    # Normalize the input data
    normalized_data = (input_data - running_mean) * tl.rsqrt(running_var + eps)
    
    # Apply weight and bias
    output_data = normalized_data * weight + bias
    
    # Store the output data
    tl.store(output_ptr + offsets, output_data, mask=mask)

# Wrapper function for batch normalization
def batch_norm(input: torch.Tensor, running_mean: torch.Tensor, running_var: torch.Tensor, weight: torch.Tensor = None, bias: torch.Tensor = None, training: bool = False, momentum: float = 0.1, eps: float = 1e-05) -> torch.Tensor:
    # Preallocate the output tensor
    output = torch.empty_like(input)
    
    # Get the dimensions of the input tensor
    batch_size, channels, height, width = input.shape
    
    # Define the grid size
    grid = lambda meta: (triton.cdiv(batch_size * channels, meta['BLOCK_SIZE']), )
    
    # Launch the kernel
    batch_norm_kernel[grid](
        input, output, running_mean, running_var, weight, bias,
        batch_size, channels, height, width, eps, momentum, training,
        BLOCK_SIZE=1024
    )
    
    return output
