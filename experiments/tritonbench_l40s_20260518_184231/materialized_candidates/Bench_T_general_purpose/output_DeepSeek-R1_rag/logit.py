import torch
import triton
import triton.language as tl

@triton.jit
def logit_kernel(
    output_ptr,
    input_ptr,
    input_row_stride,
    output_row_stride,
    n_cols,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    x = tl.load(input_ptrs, mask=col_offsets < n_cols, other=0.0)

    has_eps = ~tl.is_nan(eps)
    if has_eps:
        lower_bound = eps
        upper_bound = 1.0 - eps
        z = tl.minimum(tl.maximum(x, lower_bound), upper_bound)
    else:
        z = x

    one = 1.0
    logit_val = tl.log(z / (one - z))

    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, logit_val, mask=col_offsets < n_cols)

def logit(input, eps=None, *, out=None):
    input_reshaped = input.reshape(-1, input.shape[-1])
    n_rows, n_cols = input_reshaped.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    
    if eps is None:
        kernel_eps = float('nan')
    else:
        kernel_eps = float(eps)
    
    if out is None:
        out = torch.empty_like(input_reshaped)
    else:
        out = out.reshape_as(input_reshaped)
    
    logit_kernel[(n_rows,)](
        out,
        input_reshaped,
        input_reshaped.stride(0),
        out.stride(0),
        n_cols,
        kernel_eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    
    out = out.reshape(input.shape)
    return out
