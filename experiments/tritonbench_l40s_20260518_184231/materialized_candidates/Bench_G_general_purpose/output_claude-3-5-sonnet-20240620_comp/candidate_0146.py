import torch
import triton
import triton.language as tl

@triton.jit
def fused_add_mul_activation_kernel(
    x_ptr,          # pointer to input tensor
    bias_ptr,       # pointer to bias tensor
    in_ptr,         # pointer to additional input tensor
    output_ptr,     # pointer to output tensor
    n_elements,     # number of elements in the tensor
    multiplier,     # scaling factor
    act_type,       # activation type (0: none, 1: relu, 2: sigmoid)
    BLOCK_SIZE: tl.constexpr,  # size of block for processing
):
    # Calculate the program ID and define the block of data to process
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Create a mask for valid elements
    mask = offsets < n_elements
    
    # Load data with the mask
    x = tl.load(x_ptr + offsets, mask=mask)
    bias = tl.load(bias_ptr + offsets, mask=mask)
    in_val = tl.load(in_ptr + offsets, mask=mask)
    
    # Perform fused operations
    output = x + bias + multiplier * in_val
    
    # Apply activation function
    if act_type == 1:  # ReLU
        output = tl.maximum(output, 0.0)
    elif act_type == 2:  # Sigmoid
        output = 1 / (1 + tl.exp(-output))
    
    # Store the result
    tl.store(output_ptr + offsets, output, mask=mask)

def fused_add_mul_activation_torch(in_out_tensor, bias_tensor, in_tensor, 
                                 multiplier=1.0, activation='none'):
    # Input validation
    assert in_out_tensor.is_cuda and bias_tensor.is_cuda and in_tensor.is_cuda, \
        "All tensors must be on GPU"
    assert in_out_tensor.shape == bias_tensor.shape == in_tensor.shape, \
        "All tensors must have the same shape"
    
    # Determine activation type
    act_type = {
        'none': 0,
        'relu': 1,
        'sigmoid': 2
    }.get(activation.lower(), 0)
    
    # Get tensor properties
    n_elements = in_out_tensor.numel()
    BLOCK_SIZE = 1024  # Can be tuned based on GPU architecture
    
    # Calculate grid size
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)
    
    # Create output tensor
    output = torch.empty_like(in_out_tensor)
    
    # Launch kernel
    fused_add_mul_activation_kernel[grid](
        in_out_tensor.data_ptr(),
        bias_tensor.data_ptr(),
        in_tensor.data_ptr(),
        output.data_ptr(),
        n_elements,
        multiplier,
        act_type,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    return output
