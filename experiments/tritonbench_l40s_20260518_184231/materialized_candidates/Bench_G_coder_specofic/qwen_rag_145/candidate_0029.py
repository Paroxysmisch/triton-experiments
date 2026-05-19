import torch
import triton
import triton.language as tl
from triton import launch_grid

@triton.jit
def softmax_kernel(output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE: tl.constexpr):
    row_idx = tl.program_id(0)
    row_start_ptr = input_ptr + row_idx * input_row_stride
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    row = tl.load(input_ptrs, mask=col_offsets < n_cols, other=-float("inf"))
    row_f32 = row.to(tl.float32)
    row_minus_max = row_f32 - tl.max(row_f32, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator
    output_row_start_ptr = output_ptr + row_idx * output_row_stride
    output_ptrs = output_row_start_ptr + col_offsets
    tl.store(output_ptrs, softmax_output.to(row.dtype), mask=col_offsets < n_cols)

def softmax(input_tensor: torch.Tensor):
    device = input_tensor.device
    input_ptr = triton.pointers.address_as_pointer(input_tensor.data_ptr())
    input_row_stride = input_tensor.stride(0) * input_tensor.element_size()
    output_tensor = torch.empty_like(input_tensor, device=device)
    output_ptr = triton.pointers.address_as_pointer(output_tensor.data_ptr())
    output_row_stride = output_tensor.stride(0) * output_tensor.element_size()
    n_cols = input_tensor.size(-1)
    BLOCK_SIZE = 2**(n_cols - 1).bit_length()
    num_warps = 4
    if BLOCK_SIZE >= 2048:
        num_warps = 8
    if BLOCK_SIZE >= 4096:
        num_warps = 16
    
    grid = launch_grid(n_rows=input_tensor.size(0),
                       num_warps=num_warps,
                       num_staging=1)
    softmax_kernel[grid](output_ptr, input_ptr, input_row_stride, output_row_stride, n_cols, BLOCK_SIZE)
    
    return output_tensor
