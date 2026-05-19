import triton
import triton.language as tl

@triton.jit
def add_kernel(
    input_ptr,
    other_ptr,
    output_ptr,
    n_elements,
    alpha,
    BLOCK_SIZE: tl.constexpr
):
    """
    Triton kernel to add 'other', scaled by 'alpha', to 'input'.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_ptr: Pointer to the other tensor or scalar.
    - output_ptr: Pointer to the output tensor.
    - n_elements: Total number of elements in the tensors.
    - alpha: The multiplier for 'other'.
    - BLOCK_SIZE: Block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input and other values
    input_val = tl.load(input_ptr + offsets, mask=mask)
    other_val = tl.load(other_ptr + offsets, mask=mask)
    
    # Perform the addition with scaling
    output_val = input_val + alpha * other_val
    
    # Store the result
    tl.store(output_ptr + offsets, output_val, mask=mask)
