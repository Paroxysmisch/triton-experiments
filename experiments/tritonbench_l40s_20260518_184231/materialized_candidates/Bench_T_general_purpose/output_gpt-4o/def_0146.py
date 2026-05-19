import triton
import triton.language as tl
import torch

@triton.jit
def linear_elu_kernel(
    input_ptr, weight_ptr, bias_ptr, output_ptr,
    alpha, n_elements, n_features, 
    inplace, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    
    # Offsets for each block
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    
    # Load input, weight, and bias
    input = tl.load(input_ptr + offsets, mask=offsets < n_elements, other=0.0)
    weight = tl.load(weight_ptr + offsets, mask=offsets < n_features, other=0.0)
    bias = tl.load(bias_ptr + offsets, mask=offsets < n_features, other=0.0) if bias_ptr else 0.0
    
    # Linear transformation
    linear_out = tl.dot(input, weight) + bias
    
    # ELU activation
    elu_out = tl.where(linear_out > 0, linear_out, alpha * (tl.exp(linear_out) - 1))
    
    # Store the result
    if inplace:
        tl.store(input_ptr + offsets, elu_out, mask=offsets < n_elements)
    else:
        tl.store(output_ptr + offsets, elu_out, mask=offsets < n_elements)

def elu_linear(input, weight, bias=None, alpha=1.0, inplace=False):
    assert input.dim() == 2, "Input must be a 2D tensor"
    assert weight.dim() == 2, "Weight must be a 2D tensor"
    
    n_elements, n_features = input.shape
    output = input if inplace else torch.empty_like(input)
    
    # Get pointers to the data
    input_ptr = input.data_ptr()
    weight_ptr = weight.data_ptr()
    bias_ptr = bias.data_ptr() if bias is not None else None
    output_ptr = output.data_ptr()
    
    # Launch kernel
    grid = lambda meta: (triton.cdiv(n_elements, meta['BLOCK_SIZE']),)
    linear_elu_kernel[grid](
        input_ptr, weight_ptr, bias_ptr, output_ptr,
        alpha, n_elements, n_features,
        inplace, BLOCK_SIZE=1024
    )
    
    return output
