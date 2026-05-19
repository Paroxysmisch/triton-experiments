import torch
import triton
import triton.language as tl

@triton.jit
def fused_mv_sigmoid_sub_kernel(
    A_ptr, v_ptr, other_ptr, out_ptr,
    n, m, alpha, other_is_scalar,
    stride_am, stride_av, stride_ao,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Define offsets for this program instance
    row_start = pid * BLOCK_SIZE
    row_end = tl.min(row_start + BLOCK_SIZE, n)
    
    # Initialize accumulators for matrix-vector multiplication
    acc = tl.zeros([BLOCK_SIZE], dtype=tl.float32)
    
    # Iterate over the columns of A and elements of v
    for k in range(0, m):
        # Load A[row_start:row_end, k] and v[k]
        a = tl.load(A_ptr + row_start * stride_am + k, mask=row_start < row_end)
        v = tl.load(v_ptr + k * stride_av)
        
        # Accumulate the product
        acc += a * v
    
    # Apply sigmoid function
    s = 1 / (1 + tl.exp(-acc))
    
    # Load `other` and apply scaling
    if other_is_scalar:
        other = tl.load(other_ptr)
    else:
        other = tl.load(other_ptr + row_start * stride_ao, mask=row_start < row_end)
    
    # Compute the final result
    y = s - alpha * other
    
    # Store the result
    tl.store(out_ptr + row_start * stride_ao, y, mask=row_start < row_end)

def fused_mv_sigmoid_sub(input, vec, other, alpha=1, *, out=None):
    # Check input dimensions
    assert input.dim() == 2, "input must be a 2D tensor"
    assert vec.dim() == 1, "vec must be a 1D tensor"
    assert input.size(1) == vec.size(0), "input and vec dimensions are incompatible"
    
    n, m = input.shape
    
    # Determine if `other` is a scalar or tensor
    other_is_scalar = isinstance(other, (int, float))
    
    # Allocate output tensor if necessary
    if out is None:
        out = torch.empty(n, dtype=input.dtype, device=input.device)
    
    # Define grid size
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n, BLOCK_SIZE),)
    
    # Launch Triton kernel
    fused_mv_sigmoid_sub_kernel[grid](
        input, vec, other, out,
        n, m, alpha, other_is_scalar,
        input.stride(0), vec.stride(0), out.stride(0),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
