import triton
import triton.language as tl
import torch
import math

# Triton kernel for fused hstack and division
@triton.jit
def fused_hstack_div_kernel(X_ptr, Y_ptr, divisor_ptr, N, M, stride_X, stride_Y, stride_D, BLOCK_SIZE: tl.constexpr):
    """
    A kernel to perform horizontal stacking and element-wise division on tensors.
    
    Parameters:
    - X_ptr: Pointer to the input tensor X.
    - Y_ptr: Pointer to the output tensor Y.
    - divisor_ptr: Pointer to the divisor tensor D.
    - N: Number of elements in the stacked tensor X.
    - M: Number of elements in the divisor tensor D.
    - stride_X: Stride for accessing elements in X.
    - stride_Y: Stride for accessing elements in Y.
    - stride_D: Stride for accessing elements in D.
    - BLOCK_SIZE: The block size for Triton kernel execution.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    
    X = tl.load(X_ptr + offsets * stride_X, mask=mask)
    D = tl.load(divisor_ptr + offsets // M * stride_D, mask=mask)
    
    # Perform element-wise division with optional rounding
    if rounding_mode == 'trunc':
        Y = X // D
    elif rounding_mode == 'floor':
        Y = tl.floor(X / D)
    else:
        Y = X / D
    
    tl.store(Y_ptr + offsets * stride_Y, Y, mask=mask)

# Wrapper function for fused hstack and division
def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    """
    A wrapper function to perform horizontal stacking and element-wise division on a sequence of tensors.
    
    Parameters:
    - tensors: Sequence of tensors to be horizontally stacked.
    - divisor: Tensor or number to divide the stacked tensor by.
    - rounding_mode: Optional rounding mode ('trunc', 'floor').
    - out: Optional output tensor.
    
    Returns:
    - Resulting tensor after fusion of hstack and division.
    """
    if len(tensors) == 0:
        raise ValueError("At least one tensor must be provided.")
    
    # Determine the shape of the resulting tensor after hstack
    total_elements = sum(tensor.numel() for tensor in tensors)
    max_shape = max(tensor.shape[1:] for tensor in tensors)
    result_shape = (total_elements,) + max_shape
    
    # Create output tensor if not provided
    if out is None:
        out = torch.empty(result_shape, dtype=tensors[0].dtype, device=tensors[0].device)
    
    # Prepare pointers and strides
    X_ptr = tensors[0].data_ptr()
    stride_X = tensors[0].stride(0)
    Y_ptr = out.data_ptr()
    stride_Y = out.stride(0)
    
    # Compute the number of elements in the divisor tensor
    M = divisor.numel()
    
    # Get the stride for the divisor tensor
    stride_D = divisor.stride(0)
    
    # Launch the Triton kernel
    num_blocks = math.ceil(total_elements / BLOCK_SIZE)
    fused_hstack_div_kernel[(num_blocks, 1, 1)](X_ptr, Y_ptr, divisor.data_ptr(), total_elements, M, stride_X, stride_Y, stride_D, BLOCK_SIZE)
    
    return out
