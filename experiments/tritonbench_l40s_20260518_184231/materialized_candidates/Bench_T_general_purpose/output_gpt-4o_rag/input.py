import torch
import triton
import triton.language as tl

@triton.jit
def mean_kernel(input_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    # Get the row index for this program
    row_idx = tl.program_id(0)
    # Compute the start of the row in the input tensor
    row_start_ptr = input_ptr + row_idx * n_elements
    # Load elements of the row into shared memory
    col_offsets = tl.arange(0, BLOCK_SIZE)
    input_ptrs = row_start_ptr + col_offsets
    mask = col_offsets < n_elements
    row = tl.load(input_ptrs, mask=mask, other=0.0)
    # Compute the sum of the row
    row_sum = tl.sum(row, axis=0)
    # Compute the mean
    row_mean = row_sum / n_elements
    # Write the mean to the output tensor
    output_ptr[row_idx] = row_mean

def mean(input, dim, keepdim=False, dtype=None, out=None):
    if dtype is not None:
        input = input.to(dtype)

    if isinstance(dim, int):
        dim = (dim,)

    # Handle reduction over multiple dimensions
    reduced_shape = list(input.shape)
    for d in dim:
        reduced_shape[d] = 1 if keepdim else -1

    n_elements = 1
    for d in dim:
        n_elements *= input.shape[d]

    # Allocate output tensor
    if out is None:
        out_shape = input.shape if keepdim else [s for i, s in enumerate(input.shape) if i not in dim]
        out = torch.empty(out_shape, dtype=input.dtype, device=input.device)

    # Launch the kernel
    BLOCK_SIZE = 1024
    grid = (input.shape[0],)
    mean_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    return out

# Example usage
x = torch.randn(128, 256, device='cuda')
result = mean(x, dim=1, keepdim=True)
print(result.shape)  # Should be [128, 1] if keepdim=True
