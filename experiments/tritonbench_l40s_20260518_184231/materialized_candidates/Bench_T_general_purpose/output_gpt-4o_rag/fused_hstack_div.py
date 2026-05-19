import triton
import triton.language as tl
import torch

# Kernel for fused horizontal stacking and division
@triton.jit
def fused_hstack_div_kernel(X_ptrs, D_ptr, Y_ptr, row_size, n_cols, rounding_mode: tl.constexpr):
    """
    A kernel to perform horizontal stacking of tensors and element-wise division.
    
    Parameters:
    - X_ptrs: List of pointers to the input tensors to be stacked.
    - D_ptr: Pointer to the divisor tensor.
    - Y_ptr: Pointer to the output tensor.
    - row_size: Number of elements in a row after stacking.
    - n_cols: Number of columns (tensors) to stack.
    - rounding_mode: Rounding mode for division.
    """
    row_idx = tl.program_id(axis=0)
    col_idx = tl.arange(0, n_cols)
    
    # Load horizontally stacked row
    row = tl.load(X_ptrs + row_idx * row_size + col_idx)
    
    # Load divisor
    divisor = tl.load(D_ptr + row_idx * row_size + col_idx)
    
    # Perform division with optional rounding
    if rounding_mode == 'trunc':
        result = tl.div(row, divisor, rounding_mode='trunc')
    elif rounding_mode == 'floor':
        result = tl.div(row, divisor, rounding_mode='floor')
    else:
        result = row / divisor
    
    # Store result
    tl.store(Y_ptr + row_idx * row_size + col_idx, result)

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    """
    Wrapper function for fused horizontal stacking and division using Triton.
    
    Parameters:
    - tensors: Sequence of tensors to be horizontally stacked.
    - divisor: Tensor or number to divide the stacked tensor by.
    - rounding_mode: Optional rounding mode ('trunc', 'floor', or None).
    - out: Optional output tensor.
    
    Returns:
    - Resulting tensor after stacking and division.
    """
    # Ensure all tensors have compatible shapes for stacking
    assert all(tensor.shape[1:] == tensors[0].shape[1:] for tensor in tensors), "Incompatible shapes for stacking"
    
    # Horizontally stack tensors
    stacked_tensor = torch.hstack(tensors)
    
    # Determine output shape
    if out is None:
        out = torch.empty_like(stacked_tensor)
    
    # Determine grid and block sizes
    row_size = stacked_tensor.shape[1]
    n_rows = stacked_tensor.shape[0]
    n_cols = len(tensors)
    
    # Launch Triton kernel
    grid = (n_rows,)
    fused_hstack_div_kernel[grid](
        [tensor.data_ptr() for tensor in tensors], 
        divisor.data_ptr() if isinstance(divisor, torch.Tensor) else divisor,
        out.data_ptr(), 
        row_size, 
        n_cols, 
        rounding_mode
    )
    
    return out
