import torch
import triton
import triton.language as tl

@triton.jit
def fused_hstack_div_kernel(
    x_ptrs, d_ptrs, y_ptrs,
    stride_x, stride_d, stride_y,
    num_cols, num_stacks,
    rounding_mode: tl.constexpr
):
    # Compute row and column indices
    row_idx = tl.program_id(0)
    col_idx = tl.arange(0, num_cols)
    
    # Load data from input tensors
    x = tl.load(x_ptrs + row_idx * stride_x + col_idx, mask=col_idx < num_cols)
    d = tl.load(d_ptrs + row_idx * stride_d + col_idx, mask=col_idx < num_cols)
    
    # Perform division
    if rounding_mode == 'trunc':
        y = tl.math.trunc(x / d)
    elif rounding_mode == 'floor':
        y = tl.math.floor(x / d)
    else:
        y = x / d  # Default to true division
    
    # Store result
    tl.store(y_ptrs + row_idx * stride_y + col_idx, y, mask=col_idx < num_cols)

def fused_hstack_div(tensors, divisor, *, rounding_mode=None, out=None):
    # Validate inputs
    if not tensors:
        raise ValueError("The 'tensors' sequence must contain at least one tensor.")
    
    # Perform horizontal stacking
    stacked_tensor = torch.hstack(tensors)
    
    # Check divisor type and broadcast if necessary
    if isinstance(divisor, (int, float)):
        divisor_tensor = torch.full_like(stacked_tensor, divisor, dtype=stacked_tensor.dtype)
    else:
        divisor_tensor = divisor
        if divisor_tensor.shape != stacked_tensor.shape:
            divisor_tensor = divisor_tensor.expand_as(stacked_tensor)
    
    # Prepare output tensor
    if out is None:
        out = torch.empty_like(stacked_tensor)
    
    # Launch Triton kernel
    num_rows, num_cols = stacked_tensor.shape
    grid = (num_rows,)
    stride_x = stacked_tensor.stride(0)
    stride_d = divisor_tensor.stride(0)
    stride_y = out.stride(0)
    
    fused_hstack_div_kernel[grid](
        stacked_tensor, divisor_tensor, out,
        stride_x, stride_d, stride_y,
        num_cols, len(tensors),
        rounding_mode=rounding_mode
    )
    
    return out

# Example usage
# tensors = [torch.randn(2, 3), torch.randn(2, 4)]
# divisor = torch.tensor([[2.0, 2.0, 2.0, 2.0, 2.0, 2.0, 2.0]])
# result = fused_hstack_div(tensors, divisor, rounding_mode='floor')
# print(result)
