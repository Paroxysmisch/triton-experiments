import triton
import triton.language as tl
import torch

@triton.jit
def square_kernel(
    input_ptr, output_ptr, row_stride_in, row_stride_out, n_cols, BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    row_start_in = input_ptr + row_idx * row_stride_in
    row_start_out = output_ptr + row_idx * row_stride_out
    
    offsets = tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_cols
    
    row = tl.load(row_start_in + offsets, mask=mask, other=0.0)
    square_output = row * row
    tl.store(row_start_out + offsets, square_output, mask=mask)

def square(x):
    n_rows, n_cols = x.shape
    BLOCK_SIZE = 1 << (n_cols - 1).bit_length()
    
    y = torch.empty_like(x)
    
    num_warps = 4 if BLOCK_SIZE > 64 else 2
    grid = (n_rows,)
    
    square_kernel[grid](
        x.data_ptr(), y.data_ptr(),
        x.stride(0), y.stride(0),
        n_cols, BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return y

# Example usage:
x = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]], device='cuda')
y = square(x)
print(y)
