import torch
import triton
import triton.language as tl

@triton.jit
def fused_relu_fractional_max_pool2d(
    input, output, indices, channels, input_rows, input_cols, kernel_size, stride, index_offset, dot_dtype: tl.constexpr
):
    # Compute row and column indices
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    # Calculate indices and offsets
    idx = row_idx * input_cols + col_idx
    row_idx_stride = kernel_size * stride
    row_offset = idx * row_idx_stride
    row_start = row_offset
    row_end = row_offset + row_idx_stride
    col_offset = tl.arange(0, kernel_size) + idx * kernel_size
    col_start = col_offset
    col_end = col_offset + kernel_size

    # Initialize variables for maximum value and index
    value_max = tl.zeros([kernel_size, kernel_size], dtype=dot_dtype) - float("inf")
    idx_max = tl.zeros([kernel_size, kernel_size], dtype=tl.int32) + channels

    # Iterate through the input to find the maximum value and index
    for r in range(row_start, row_end, stride):
        for c in col_start, col_end:
            mask = (r < input_rows) & (c < input_cols)
            a = tl.load(input + r + c, mask=mask, other=float("-inf")).to(dot_dtype)
            b = tl.max(a, axis=0)
            idx_a = tl.load(indices + r + c, mask=mask, other=idx_max).to(tl.int32)
            idx_b = tl.where(b == a, c, idx_a)
            value_max = tl.where(b > value_max, b, value_max)
            idx_max = tl.where(b > value_max, idx_b, idx_max)

    # Store the result
    tl.store(output + row_idx * input_cols + col_idx, value_max, mask=(col_idx < input_cols))
    tl.store(indices + row_idx * input_cols + col_idx, idx_max, mask=(col_idx < input_cols))


def fused_fractional_max_pool2d_with_relu(input: torch.Tensor, kernel_size, output_size=None, output_ratio=None, return_indices=False) -> torch.Tensor:
    # Ensure input has four dimensions
    if input.dim() != 4:
        raise ValueError("Input must have four dimensions")

    # Prepare output tensor
    output = torch.empty(input.size(0), input.size(1), *output_size, device=input.device, dtype=input.dtype)
    index_dtype = torch.int32 if kernel_size < 65536 else torch.int64
    indices = torch.empty(input.size(0), input.size(1), *output_size, device=input.device, dtype=index_dtype)

    # Define constants
    stride = 1 if output_ratio is None else output_ratio[0] / input.size(2)
    index_offset = 0

    # Call Triton kernel
    fused_relu_fractional_max_pool2d[(output_size[0], output_size[1])](
        input,
        output,
        indices,
        input.size(1),
        input.size(2),
        input.size(3),
        kernel_size,
        stride,
        index_offset,
        input.dtype,
    )

    # Apply ReLU activation
    output = torch.nn.functional.relu(output)

    # Return results
    if return_indices:
        return output, indices
    else:
        return output
