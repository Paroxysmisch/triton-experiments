import triton
import triton.language as tl
import torch

@triton.jit
def softmax_kernel(output_ptr, input_ptr, row_stride, n_cols, mask_ptr, BLOCK_SIZE):
    row_id = tl.program_id(0)
    row_start = row_id * row_stride
    col_offsets = tl.arange(0, n_cols)
    input_ptrs = input_ptr + row_start + col_offsets
    x = tl.load(input_ptrs)
    if mask_ptr is not None:
        mask = tl.load(mask_ptr + row_start + col_offsets)
        x = x + mask
    x_max = tl.max(x, axis=0)
    x = x - x_max
    exp_x = tl.exp(x)
    sum_exp_x = tl.sum(exp_x, axis=0)
    output = exp_x / sum_exp_x
    tl.store(output_ptr + row_start + col_offsets, output)

def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1):
    assert dim == -1, "Only support softmax on the last dimension"
    assert input.dim() == 2, "Only support 2D tensor"
    if mask is not None:
        assert input.shape == mask.shape, "Input and mask should have the same shape"
    n_rows, n_cols = input.shape
    output = torch.empty_like(input)
    if n_rows > 1024:
        BLOCK_SIZE = 1024
        grid_size = int(n_rows / BLOCK_SIZE)
        if n_rows % BLOCK_SIZE != 0:
            grid_size += 1
        softmax_kernel[grid_size, BLOCK_SIZE](output, input, n_cols, mask, BLOCK_SIZE)
    else:
        BLOCK_SIZE = n_rows
        softmax_kernel[1, BLOCK_SIZE](output, input, n_cols, mask, BLOCK_SIZE)
    return output
