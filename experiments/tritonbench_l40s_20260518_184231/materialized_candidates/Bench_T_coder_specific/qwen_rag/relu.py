import triton
import triton.language as tl

@triton.jit
def relu_kernel(output_ptr, input_ptr, n_elements: tl.constexpr, BLOCK_SIZE: tl.constexpr):
    # Get the index of the current element
    pid = tl.program_id(axis=0)
    coords = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    
    # Load elements from input tensor
    inputs = tl.load(input_ptr + coords, mask=coords < n_elements, other=0.0)
    
    # Apply ReLU operation
    outputs = tl.maximum(inputs, 0.0)
    
    # Store results in output tensor
    tl.store(output_ptr + coords, outputs, mask=coords < n_elements)
