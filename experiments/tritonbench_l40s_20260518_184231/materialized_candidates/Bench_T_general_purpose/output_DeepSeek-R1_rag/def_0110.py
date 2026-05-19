import torch
import triton
import triton.language as tl

@triton.jit
def exp_mean_kernel(
    input_ptr,  # pointer to the input tensor
    output_ptr,  # pointer to the output tensor
    input_row_stride,  # stride between rows of the input
    input_col_stride,  # stride between columns of the input
    output_row_stride,  # stride between rows of the output
    n_rows,  # number of rows in the input (collapsed leading dimensions)
    n_cols,  # number of columns in the input (dimension to reduce)
    BLOCK_SIZE: tl.constexpr,  # number of columns each block handles
):
    row_idx = tl.program_id(0)
    if row_idx >= n_rows:
        return

    row_start = input_ptr + row_idx * input_row_stride
    sum_exp = 0.0
    for col_block_start in range(0, n_cols, BLOCK_SIZE):
        col_offsets = tl.arange(0, BLOCK_SIZE) + col_block_start
        mask = col_offsets < n_cols
        input_ptrs = row_start + col_offsets * input_col_stride
        x = tl.load(input_ptrs, mask=mask, other=0.0)
        x = x.to(tl.float32)  # Convert to float32 for higher precision
        exp_x = tl.exp(x)
        sum_exp += tl.sum(exp_x, axis=0)

    mean = sum_exp / n_cols
    output_ptr_row = output_ptr + row_idx * output_row_stride
    tl.store(output_ptr_row, mean)

def exp_mean(input: torch.Tensor, dim=None, keepdim=False, dtype=None, out=None) -> torch.Tensor:
    # Ensure input is contiguous
    input = input.contiguous()
    
    # Determine output dtype
    if dtype is None:
        dtype = input.dtype
    else:
        dtype = dtype
    
    # Handle output tensor
    if out is not None:
        assert out.dtype == dtype, "Output tensor dtype does not match specified dtype"
        assert out.is_contiguous(), "Output tensor must be contiguous"
    else:
        # Determine output shape
        if dim is None:
            output_shape = [] if not keepdim else [1] * input.ndim
        else:
            output_shape = list(input.shape)
            if keepdim:
                output_shape[dim] = 1
            else:
                output_shape.pop(dim)
        out = torch.empty(output_shape, dtype=dtype, device=input.device)
    
    # Handle the case when dim is None: reduce all dimensions
    if dim is None:
        input_reshaped = input.view(-1)
        n_rows = 1
        n_cols = input_reshaped.shape[0]
    else:
        # Validate dimension
        dim = dim if dim >= 0 else input.dim() + dim
        assert 0 <= dim < input.dim(), "dim out of range"
        # Permute the target dimension to the end and make contiguous
        permuted_dims = [d for d in range(input.dim()) if d != dim] + [dim]
        input_permuted = input.permute(permuted_dims).contiguous()
        # Collapse all leading dimensions
        input_reshaped = input_permuted.view(-1, input_permuted.shape[-1])
        n_rows = input_reshaped.shape[0]
        n_cols = input_reshaped.shape[1]
    
    # Launch kernel
    BLOCK_SIZE = 1024  # Tune this based on hardware specifics
    grid = (n_rows,)
    exp_mean_kernel[grid](
        input_reshaped, out,
        input_reshaped.stride(0), input_reshaped.stride(1),
        out.stride(0) if out.dim() > 0 else 0,
        n_rows, n_cols,
        BLOCK_SIZE
    )
    
    # Reshape the output to the desired shape
    if dim is not None:
        # Compute the output shape based on original dimensions and keepdim
        output_shape = list(input.shape)
        if keepdim:
            output_shape[dim] = 1
        else:
            output_shape.pop(dim)
        out = out.view(output_shape)
    else:
        # If dim is None, output is a scalar
        out = out.view(())
    
    return out
