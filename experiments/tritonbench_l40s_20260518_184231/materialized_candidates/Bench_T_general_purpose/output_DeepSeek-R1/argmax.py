import torch
import triton
import triton.language as tl
from typing import Optional

@triton.jit
def argmax_kernel(
    input_ptr,
    output_ptr,
    input_row_stride,
    output_row_stride,
    row_size,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    input_row_ptr = input_ptr + row_idx * input_row_stride
    output_row_ptr = output_ptr + row_idx * output_row_stride

    max_value = -tl.inf(tl.float32)
    max_index = 0
    for i in range(0, row_size, BLOCK_SIZE):
        offsets = i + tl.arange(0, BLOCK_SIZE)
        mask = offsets < row_size
        current_values = tl.load(input_row_ptr + offsets, mask=mask, other=-tl.inf(tl.float32))
        current_indices = offsets

        current_max_value = tl.max(current_values, axis=0)
        current_max_pos = tl.argmax(current_values, axis=0)
        current_max_index = current_indices[current_max_pos]

        if (current_max_value > max_value) or ((current_max_value == max_value) & (current_max_index < max_index)):
            max_value = current_max_value
            max_index = current_max_index

    tl.store(output_row_ptr, max_index)

def argmax(input: torch.Tensor, dim: Optional[int], keepdim: bool = False) -> torch.Tensor:
    if dim is None:
        input_flat = input.flatten().contiguous()
        if input_flat.numel() == 0:
            return torch.empty((), dtype=torch.long, device=input.device)
        output = torch.empty((), dtype=torch.long, device=input.device)
        grid = lambda meta: (1,)
        argmax_kernel[grid](input_flat, output, input_flat.stride(0), output.stride(0), input_flat.size(0), BLOCK_SIZE=1024)
        return output
    else:
        if dim < 0:
            dim += input.dim()
        assert 0 <= dim < input.dim(), f"Dimension {dim} out of range for tensor of dimension {input.dim()}"
        input_contig = input.contiguous()
        original_shape = input_contig.shape
        size_dim = original_shape[dim]
        other_dims = list(original_shape)
        del other_dims[dim]
        num_rows = 1
        for d in other_dims:
            num_rows *= d
        if num_rows == 0:
            output_shape = list(original_shape)
            if keepdim:
                output_shape[dim] = 1
            else:
                del output_shape[dim]
            return torch.empty(output_shape, dtype=torch.long, device=input.device)
        input_reshaped = input_contig.view(num_rows, size_dim)
        output_shape = list(original_shape)
        if keepdim:
            output_shape[dim] = 1
        else:
            del output_shape[dim]
        output = torch.empty(output_shape, dtype=torch.long, device=input.device)
        if output.numel() == 0:
            return output
        if keepdim:
            output_reshaped = output.view(num_rows, 1)
        else:
            output_reshaped = output.view(num_rows)
        input_row_stride = input_reshaped.stride(0)
        output_row_stride = output_reshaped.stride(0)
        grid = (num_rows,)
        BLOCK_SIZE = 1024
        argmax_kernel[grid](input_reshaped, output_reshaped, input_row_stride, output_row_stride, size_dim, BLOCK_SIZE=BLOCK_SIZE)
        return output
