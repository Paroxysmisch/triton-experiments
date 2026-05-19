import torch
import triton
import triton.language as tl

@triton.jit
def _softmax_kernel(INPUT_PTR, OUTPUT_PTR, row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_id = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    offset = row_id * row_stride
    mask = cols < n_cols

    x = tl.load(INPUT_PTR + offset + cols, mask=mask, other=-float("inf"))
    x_max = tl.max(x, axis=0)
    x = x - x_max
    expx = tl.exp(x)
    sum_expx = tl.sum(expx, axis=0)
    out = expx / sum_expx

    tl.store(OUTPUT_PTR + offset + cols, out, mask=mask)

def softmax(input, dim, dtype=None):
    if dtype is not None:
        input = input.to(dtype)

    # Handle negative dims
    dim = dim if dim >= 0 else (dim + input.dim())

    # Permute so that dim is last
    perm_dims = list(range(input.dim()))
    perm_dims[-1], perm_dims[dim] = perm_dims[dim], perm_dims[-1]
    input_t = input.permute(perm_dims)

    shape_t = input_t.shape
    row_count = 1
    for s in shape_t[:-1]:
        row_count *= s
    col_count = shape_t[-1]

    input_2d = input_t.reshape(row_count, col_count)
    output_2d = torch.empty_like(input_2d)

    BLOCK_SIZE = 1024
    grid = (row_count,)

    _softmax_kernel[grid](
        input_2d, 
        output_2d, 
        input_2d.stride(0), 
        col_count, 
        BLOCK_SIZE=BLOCK_SIZE
    )

    out_t = output_2d.reshape(shape_t)

    # Invert permutation to restore original dimensions
    inv_perm = [0] * input.dim()
    for i, j in enumerate(perm_dims):
        inv_perm[j] = i
    return out_t.permute(inv_perm)
