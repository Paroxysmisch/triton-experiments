import triton
import triton.language as tl

@triton.jit
def dropout_sigmoid_linear_kernel(input_ptr, weight_ptr, bias_ptr, output_ptr, p, n_elements):
    # Get the index of the current element
    idx = tl.program_id(0)
    
    # Load input, weight, and bias
    input_val = tl.load(input_ptr + idx)
    weight_val = tl.load(weight_ptr)
    bias_val = tl.load(bias_ptr) if bias_ptr is not None else 0.0
    
    # Apply linear transformation
    linear_output = tl.dot(weight_val, input_val) + bias_val
    
    # Apply sigmoid activation
    sigmoid_output = 1 / (1 + tl.exp(-linear_output))
    
    # Apply dropout
    if tl.random.uniform(0, 1) < p:
        sigmoid_output = 0.0  # Zero out the output based on dropout probability
    
    # Store the result
    tl.store(output_ptr + idx, sigmoid_output)

import torch

def dropout_sigmoid_linear(input: torch.Tensor, weight: torch.Tensor, bias=None, p=0.5, training=True, inplace=False) -> torch.Tensor:
    # Ensure input and weight are on the same device
    assert input.device == weight.device, "Input and weight must be on the same device"
    
    # Prepare output tensor
    output = input.new_zeros((input.size(0), weight.size(0)))  # Shape: (batch_size, out_features)
    
    # Call Triton kernel
    if training:
        # Apply dropout only during training
        dropout_sigmoid_linear_kernel[(input.size(0),)](input, weight, bias, output, p, input.size(0))
    else:
        # Apply without dropout
        output = torch.sigmoid(input @ weight.t() + (bias if bias is not None else 0))
    
    return output
