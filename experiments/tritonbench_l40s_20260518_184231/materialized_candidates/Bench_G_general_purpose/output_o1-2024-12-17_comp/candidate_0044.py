import torch
import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, 
    output_ptr,
    stride_in,
    stride_out,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row_id = tl.program_id(0)
    row_offset_in = row_id * stride_in
    row_offset_out = row_id * stride_out
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    x = tl.load(input_ptr + row_offset_in + offsets, mask=mask, other=-float('inf'))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    x_exp = tl.exp(x)
    s = tl.sum(x_exp, axis=0)
    y = x_exp / s
    tl.store(output_ptr + row_offset_out + offsets, y, mask=mask)

def softmax(input_tensor: torch.Tensor) -> torch.Tensor:
    n_rows, n_cols = input_tensor.shape
    output = torch.empty_like(input_tensor)
    # Compute next power of two for BLOCK_SIZE
    BLOCK_SIZE = 1 << (n_cols - 1).bit_length()
    # Choose number of warps
    num_warps = min(4, max(1, BLOCK_SIZE // 256))
    grid = (n_rows,)
    softmax_kernel[grid](
        input_tensor, 
        output,
        input_tensor.stride(0),
        output.stride(0),
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    return output
