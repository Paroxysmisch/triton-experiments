import triton
import triton.language as tl
import torch

# Constants
BLOCK_SIZE = 128
NUM_WARPS = 4

@triton.jit
def _geglu_tanh_forward_kernel(a_ptr, b_ptr, c_ptr, n_cols, **meta):
    row_idx = tl.program_id(0)
    # Pointers to the row of a, b, and c
    a_row_ptr = a_ptr + row_idx * n_cols
    b_row_ptr = b_ptr + row_idx * n_cols
    c_row_ptr = c_ptr + row_idx * n_cols
    
    # Block of elements
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load a and b
    a = tl.load(a_row_ptr + offsets, mask=mask)
    b = tl.load(b_row_ptr + offsets, mask=mask)
    
    # Compute GEGLU using tanh approximation
    c = 0.5 * a * (1 + tl.math.tanh(tl.math.sqrt(2 / 3.141592653589793) * (a + 0.044715 * a * a * a)))
    
    # Store the result
    tl.store(c_row_ptr + offsets, c, mask=mask)

def geglu_forward(a, b):
    n_rows, n_cols = a.shape
    c = torch.empty_like(a)
    
    grid = (n_rows,)
    _geglu_tanh_forward_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        c_ptr=c,
        n_cols=n_cols,
        num_warps=NUM_WARPS,
        num_stages=1
    )
    
    return c

@triton.jit
def _geglu_tanh_backward_kernel(a_ptr, b_ptr, dc_ptr, da_ptr, db_ptr, n_cols, **meta):
    row_idx = tl.program_id(0)
    # Pointers to the row of a, b, dc, da, and db
    a_row_ptr = a_ptr + row_idx * n_cols
    b_row_ptr = b_ptr + row_idx * n_cols
    dc_row_ptr = dc_ptr + row_idx * n_cols
    da_row_ptr = da_ptr + row_idx * n_cols
    db_row_ptr = db_ptr + row_idx * n_cols
    
    # Block of elements
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    # Load a, b, and dc
    a = tl.load(a_row_ptr + offsets, mask=mask)
    b = tl.load(b_row_ptr + offsets, mask=mask)
    dc = tl.load(dc_row_ptr + offsets, mask=mask)
    
    # Recompute intermediates
    tanh_intermediate = tl.math.tanh(tl.math.sqrt(2 / 3.141592653589793) * (a + 0.044715 * a * a * a))
    derivative = 0.5 * (1 + tanh_intermediate) + 0.5 * a * (1 - tanh_intermediate * tanh_intermediate) * (tl.math.sqrt(2 / 3.141592653589793) * (1 + 3 * 0.044715 * a * a))
    
    # Compute gradients
    da = dc * derivative
    db = da * b
    
    # Store the gradients
    tl.store(da_row_ptr + offsets, da, mask=mask)
    tl.store(db_row_ptr + offsets, db, mask=mask)

def geglu_backward(a, b, dc):
    n_rows, n_cols = a.shape
    da = torch.empty_like(a)
    db = torch.empty_like(b)
    
    grid = (n_rows,)
    _geglu_tanh_backward_kernel[grid](
        a_ptr=a,
        b_ptr=b,
        dc_ptr=dc,
        da_ptr=da,
        db_ptr=db,
        n_cols=n_cols,
        num_warps=NUM_WARPS,
        num_stages=1
    )
    
    return da, db
