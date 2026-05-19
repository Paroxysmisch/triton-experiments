import torch
import triton
import triton.language as tl

@triton.jit
def _softmax(output_ptr, input_ptr, input_stride, output_stride, n_rows, n_cols, n_depth, BLOCK_SIZE: tl.constexpr, LOG: tl.constexpr, CAUSAL: tl.constexpr, MASK_TYPE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    input_row_start = input_ptr + row_idx * input_stride
    output_row_start = output_ptr + row_idx * output_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = input_row_start + col_offsets

    mask = col_offsets < n_cols
    row = tl.load(input_ptrs, mask=mask, other=-float('inf'))

    row_minus_max = row - tl.max(row, axis=0)
    numerator = tl.exp(row_minus_max)
    denominator = tl.sum(numerator, axis=0)
    softmax_output = numerator / denominator

    if LOG:
        softmax_output = tl.log(softmax_output)

    output_ptrs = output_row_start + col_offsets
    tl.store(output_ptrs, softmax_output, mask=mask)

def softmax(input_tensor, log=False, causal=False, mask_type=None):
    n_batches, n_rows, n_cols = input_tensor.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    output_tensor = torch.empty_like(input_tensor)

    grid = (n_rows, n_batches)

    _softmax[grid](
        output_tensor,
        input_tensor,
        input_tensor.stride(1),
        output_tensor.stride(1),
        n_rows,
        n_cols,
        input_tensor.stride(2),
        BLOCK_SIZE,
        log,
        causal,
        mask_type
    )
    return output_tensor

@triton.jit
def _softmax_backward(grad_output_ptr, grad_input_ptr, output_ptr, input_stride, output_stride, grad_stride, n_rows, n_cols, n_depth, BLOCK_SIZE: tl.constexpr, LOG: tl.constexpr, CAUSAL: tl.constexpr, MASK_TYPE: tl.constexpr):
    row_idx = tl.program_id(0)
    col_idx = tl.program_id(1)

    output_row_start = output_ptr + row_idx * output_stride
    grad_output_row_start = grad_output_ptr + row_idx * grad_stride
    grad_input_row_start = grad_input_ptr + row_idx * input_stride

    col_offsets = tl.arange(0, BLOCK_SIZE)
    output_ptrs = output_row_start + col_offsets
    grad_output_ptrs = grad_output_row_start + col_offsets
    grad_input_ptrs = grad_input_row_start + col_offsets

    mask = col_offsets < n_cols
    output = tl.load(output_ptrs, mask=mask)
    grad_output = tl.load(grad_output_ptrs, mask=mask)

    if LOG:
        grad_input = grad_output - tl.exp(output) * tl.sum(grad_output, axis=0)
    else:
        grad_input = output * (grad_output - tl.sum(output * grad_output, axis=0))

    tl.store(grad_input_ptrs, grad_input, mask=mask)

def softmax_backward(grad_output, output, log=False, causal=False, mask_type=None):
    n_batches, n_rows, n_cols = grad_output.shape
    BLOCK_SIZE = triton.next_power_of_2(n_cols)

    grad_input = torch.empty_like(grad_output)

    grid = (n_rows, n_batches)

    _softmax_backward[grid](
        grad_output,
        grad_input,
        output,
        output.stride(1),
        grad_output.stride(1),
        grad_input.stride(1),
        n_rows,
        n_cols,
        grad_output.stride(2),
        BLOCK_SIZE,
        log,
        causal,
        mask_type
    )
    return grad_input

# Example usage:
input_tensor = torch.randn(2, 3, 4, device='cuda')
output_tensor = softmax(input_tensor)
grad_output = torch.randn_like(output_tensor)
grad_input = softmax_backward(grad_output, output_tensor)
