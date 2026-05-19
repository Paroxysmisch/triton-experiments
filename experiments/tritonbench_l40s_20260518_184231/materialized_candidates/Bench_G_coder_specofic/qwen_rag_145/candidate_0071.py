import triton.language as tl
import torch

@triton.jit
def square_kernel(
    in_ptr, out_ptr, n_cols, BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    tile_start_ptr = in_ptr + row * n_cols
    tile_end_ptr = tile_start_ptr + n_cols 
    
    mask = tl.mask(tl.bitwise((tl.load(tile_start_ptr + i) for i in range(BLOCK_SIZE)),
                              mode='LT',
                              absolute_value=True)
                  
    in_row = tl.load(tile_start_ptr + tl.arange(0, BLOCK_SIZE))
    square_output = in_row * in_row
    square_output = tl.where(mask, 0, square_output)
    tl.store(out_ptr + row * n_cols, square_output)


def square(x):
    n_rows, n_cols = x.shape
    assert n_cols <= 2048, "BLOCK_SIZE should be at least as large as number of columns"
    BLOCK_SIZE = 2 ** tl.ceil(tl.log2(torch.tensor(n_cols)))
    assert BLOCK_SIZE <= 2048, "BLOCK_SIZE should not exceed 2048 due to shared memory limitations"

    y = torch.empty_like(x)
    num_warps = 4
    square_kernel[(n_rows,)](x, y, n_cols, BLOCK_SIZE=BLOCK_SIZE)
    return y
