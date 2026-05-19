import torch
import triton
import triton.language as tl

@triton.jit
def sigmoid(x):
    return 1 / (1 + tl.exp(-x))

@triton.jit
def sigmoid_argmax_kernel(output_ptr, input_ptr, input_row_stride, input_col_stride, n_rows, n_cols, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_rows * n_cols

    input_offsets = input_ptr + offsets * input_col_stride
    input_vals = tl.load(input_offsets, mask=mask, other=-float('inf'))

    # Apply sigmoid function
    sigmoid_vals = sigmoid(input_vals)

    # Find the maximum value and its index
    max_val, max_idx = tl.max_argmax(sigmoid_vals, axis=0, mask=mask)

    # Write the index back to the output tensor
    output_offsets = output_ptr + pid
    tl.store(output_offsets, max_idx, mask=mask)

def sigmoid_argmax(input, dim=None, keepdim=False):
    if dim is None:
        # Flatten the tensor and find the argmax
        input_flattened = input.flatten()
        max_idx = torch.argmax(input_flattened.sigmoid())
        return max_idx.item()

    # Ensure dim is within the valid range
    if dim < 0:
        dim = input.dim() + dim

    # Compute the shape of the output tensor
    output_shape = list(input.shape)
    if not keepdim:
        output_shape.pop(dim)

    # Allocate the output tensor
    output = torch.empty(output_shape, dtype=torch.int64, device=input.device)

    # Determine the block size
    BLOCK_SIZE = 1024

    # Determine the number of programs
    n_rows = input.shape[dim]
    n_cols = input.numel() // n_rows

    # Launch the kernel
    grid = (n_rows, 1, 1)
    sigmoid_argmax_kernel[grid](
        output,
        input,
        input.stride(dim),
        input.stride(0),
        n_rows,
        n_cols,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output

# Example usage
input_tensor = torch.randn(4, 5, device='cuda')
result = sigmoid_argmax(input_tensor, dim=1, keepdim=True)
print(result)
