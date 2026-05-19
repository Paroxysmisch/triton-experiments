import triton
import triton.language as tl

@triton.jit
def mul_sub_kernel(
    input_ptr, other_mul_ptr, other_sub_ptr, alpha, output_ptr,
    n_elements: tl.constexpr
):
    """
    A kernel to perform element-wise multiplication and subtraction on tensors.
    
    Parameters:
    - input_ptr: Pointer to the input tensor.
    - other_mul_ptr: Pointer to the tensor or scalar to multiply with `input`.
    - other_sub_ptr: Pointer to the tensor or scalar to subtract from the multiplication result.
    - alpha: Multiplier for `other_sub`.
    - output_ptr: Pointer to the output tensor.
    - n_elements: Total number of elements in the tensors.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    
    # Load input values
    input_val = tl.load(input_ptr + offsets, mask=mask)
    
    # Load other_mul values
    if isinstance(other_mul, torch.Tensor):
        other_mul_val = tl.load(other_mul_ptr + offsets, mask=mask)
    else:
        other_mul_val = other_mul
    
    # Load other_sub values
    if isinstance(other_sub, torch.Tensor):
        other_sub_val = tl.load(other_sub_ptr + offsets, mask=mask)
    else:
        other_sub_val = other_sub
    
    # Perform the operation
    result = (input_val * other_mul_val) - (alpha * other_sub_val)
    
    # Store the result
    tl.store(output_ptr + offsets, result, mask=mask)
