import torch
import triton
import triton.language as tl

@triton.jit
def softmax_mul_kernel(
    input_ptr, other_ptr, output_ptr,
    n_rows, n_cols,
    input_row_stride, input_col_stride,
    other_row_stride, other_col_stride,
    output_row_stride, output_col_stride,
    BLOCK_SIZE: tl.constexpr
):
    row_idx = tl.program_id(0)
    col_offsets = tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < n_cols

    # Load input row
    input_row_ptr = input_ptr + row_idx * input_row_stride
    input_vals = tl.load(input_row_ptr + col_offsets * input_col_stride, mask=mask, other=-float('inf'))

    # Compute max for numerical stability
    row_max = tl.max(input_vals, axis=0)
    input_vals -= row_max

    # Compute exponentials and sum
    exp_vals = tl.exp(input_vals)
    row_sum = tl.sum(exp_vals, axis=0)

    # Compute softmax
    softmax_vals = exp_vals / row_sum

    # Load other row
    other_row_ptr = other_ptr + row_idx * other_row_stride
    other_vals = tl.load(other_row_ptr + col_offsets * other_col_stride, mask=mask, other=0.0)

    # Multiply softmax by other values
    output_vals = softmax_vals * other_vals

    # Write output
    output_row_ptr = output_ptr + row_idx * output_row_stride
    tl.store(output_row_ptr + col_offsets * output_col_stride, output_vals, mask=mask)

def softmax_mul(input, other, dim, dtype=None, out=None) -> torch.Tensor:
    # Cast input to the specified dtype if provided
    if dtype is not None:
        input = input.to(dtype)
    else:
        dtype = input.dtype
    
    # Convert other to a tensor if it's a number and ensure it's on the same device and dtype
    if not isinstance(other, torch.Tensor):
        other = torch.tensor(other, device=input.device, dtype=dtype)
    else:
        other = other.to(device=input.device, dtype=dtype)
    
    # Broadcast other to the shape of input
    try:
        other = other.broadcast_to(input.shape)
    except RuntimeError:
        raise RuntimeError("The shape of 'other' must be broadcastable to the shape of 'input'")
    
    # Ensure contiguous tensors after broadcasting and reshaping
    original_shape = input.shape
    size = original_shape[dim]
    outer_dim = input.numel() // size  # Total number of elements divided by the softmax dimension size
    
    # Reshape input and other to 2D tensors
    input_2d = input.reshape(outer_dim, size).contiguous()
    other_2d = other.reshape(outer_dim, size).contiguous()
    
    # Allocate output tensor
    output = torch.empty_like(input_2d)
    
    # Determine the block size for the Triton kernel
    BLOCK_SIZE = triton.next_power_of_2(size)
    if BLOCK_SIZE > 1024:
        BLOCK_SIZE = 1024
    
    # Launch the Triton kernel
    grid = (outer_dim,)
    softmax_mul_kernel[grid](
        input_2d, other_2d, output,
        outer_dim, size,
        input_2d.stride(0), input_2d.stride(1),
        other_2d.stride(0), other_2d.stride(1),
        output.stride(0), output.stride(1),
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    # Reshape the output back to the original shape
    output = output.reshape(original_shape)
    
    # Handle the output tensor if provided
    if out is not None:
        out.copy_(output)
        return out
    return output
