import torch
import triton
import triton.language as tl

@triton.jit
def relu_kernel(output_ptr, input_ptr, n_elements, inplace: tl.constexpr):
    # Compute element-wise ReLU
    # row_id represents the index of the current element in the tensor
    row_id = tl.program_id(0)
    
    # Ensure we are within the bounds of the input tensor
    if row_id < n_elements:
        # Load the input value
        x = tl.load(input_ptr + row_id)
        
        # Apply ReLU function: max(0, x)
        result = tl.maximum(x, 0.0)
        
        if inplace:
            # If inplace operation is requested, write directly to input
            tl.store(input_ptr + row_id, result)
        else:
            # Otherwise, write to the output tensor
            tl.store(output_ptr + row_id, result)

def relu(input, inplace=False):
    # Get the total number of elements in the input tensor
    n_elements = input.numel()

    # Allocate output tensor if inplace=False
    if not inplace:
        output = torch.empty_like(input)
    else:
        output = input

    # Launch the Triton kernel
    relu_kernel[(n_elements,)](output, input, n_elements, inplace=inplace)

    return output
