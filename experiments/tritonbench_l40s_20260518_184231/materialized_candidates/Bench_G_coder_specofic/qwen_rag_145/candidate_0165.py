import torch
import triton
import triton.language as tl
import triton.compiler as tc
from triton.runtime import driver

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_rows, n_cols, mask_ptr, BLOCK_SIZE: tl.constexpr):
    row_start = tl.program_id(0)
    row_step = tl.num_programs(0)
    for row_idx in tl.range(row_start, n_rows, row_step):
        col_offsets = tl.arange(0, BLOCK_SIZE)
        input_ptrs = input_ptr + row_idx * input_row_stride + col_offsets
        output_ptrs = output_ptr + row_idx * output_row_stride + col_offsets
        mask = None
        if mask_ptr is not None:
            mask = mask_ptr + col_offsets < n_cols
        row = tl.load(input_ptrs, mask=mask, other=-float('inf'))
        row_minus_max = row - tl.max(row, axis=0)
        numerator = tl.exp(row_minus_max)
        denominator = tl.sum(numerator, axis=0)
        softmax_output = numerator / denominator
        tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(input: torch.Tensor, mask: torch.Tensor = None, dim=-1):
    assert isinstance(input, torch.Tensor), "Input must be a Tensor"
    assert len(input.shape) <= 2, "Input must be at most a 2D tensor"
    assert dim == -1, "Only the last dimension is allowed for softmax"
    if len(input.shape) == 1:
        input = input.unsqueeze(0)
    input = input.contiguous()
    if mask is not None:
        assert mask.shape == input.shape, "Mask and input tensor must have the same shape"
        mask = mask.contiguous()
    device = input.device
    output = torch.empty_like(input)

    if input.numel() > 256*1024: 
        num_programs = 8
    elif input.numel() > 1024*64:
        num_programs = 4
    else:
        num_programs = 2

    kernel = softmax_kernel[(num_programs, 1, 1)]
    BLOCK_SIZE = triton.next_power_of_2(input.shape[1])
    grid = lambda meta: (driver.active.utils.cuda.get_n_warps_per_cta(meta) * num_programs, 1, 1)
    kernel[grid](output, input, input.stride(0), output.stride(0), input.shape[0], input.shape[1], mask.int() if mask is not None else None, BLOCK_SIZE)
    return output
