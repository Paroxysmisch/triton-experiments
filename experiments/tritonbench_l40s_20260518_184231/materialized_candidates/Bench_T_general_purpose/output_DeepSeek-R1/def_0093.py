import torch
import triton
import triton.language as tl

@triton.jit
def softmax_log_kernel(
    output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float('inf'))

    # Compute log of the input row
    log_row = tl.log(row)

    # Subtract max for numerical stability
    max_log = tl.max(log_row, axis=0)
    log_row_minus_max = log_row - max_log

    # Compute exponentials
    exp_row = tl.exp(log_row_minus_max)

    # Sum exponentials
    sum_exp = tl.sum(exp_row, axis=0)

    # Compute softmax
    softmax_output = exp_row / sum_exp

    # Write output
    output_row_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_ptr + col_offsets
    tl.store(output_ptrs, softmax_output, mask=col_offsets < n_cols)

def softmax_log(input, dim=-1, dtype=None) -> torch.Tensor:
    if dtype is not None:
        input = input.to(dtype)
    else:
        dtype = input.dtype
    
    assert input.is_cuda, "Input tensor must be on CUDA device"
    dim = dim if dim >= 0 else input.dim() + dim

    if dim != input.dim() - 1:
        # Transpose to make the target dimension the last one
        input = input.transpose(dim, -1)
        output = softmax_log(input, dim=-1, dtype=dtype)
        return output.transpose(dim, -1)
    
    # Ensure input is contiguous after potential transpose
    input = input.contiguous()
    log_input = torch.log(input)
    output = torch.empty_like(log_input)
    
    n_rows = log_input.numel() // log_input.size(-1)
    n_cols = log_input.size(-1)
    
    BLOCK_SIZE = triton.next_power_of_2(n_cols)
    if BLOCK_SIZE > 4096:
        raise ValueError("Block size exceeds maximum allowed value")
    
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    elif BLOCK_SIZE >= 1024:
        num_warps = 8
    
    grid = (n_rows,)
    softmax_log_kernel[grid](
        output, log_input, log_input.stride(-2), output.stride(-2), n_cols,
        BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps
    )
    
    return output
