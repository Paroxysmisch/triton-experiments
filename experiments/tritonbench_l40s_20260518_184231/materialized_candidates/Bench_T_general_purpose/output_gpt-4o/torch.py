import torch
import triton
import triton.language as tl

@triton.jit
def permute_kernel(input_ptr, output_ptr, shape, perm, numel, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    # Calculate the linear index for this block
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel

    # Compute multi-dimensional index from the linear index
    multi_index = tl.zeros(len(shape), dtype=tl.int32)
    strides = tl.cumprod(tl.cat([tl.tensor([1], dtype=tl.int32), shape[:-1]]))
    remaining = offsets
    for i in range(len(shape) - 1, -1, -1):
        multi_index[i] = remaining // strides[i]
        remaining = remaining % strides[i]

    # Permute the indices
    permuted_index = multi_index[perm]

    # Compute the linear index in the output tensor
    output_strides = tl.cumprod(tl.cat([tl.tensor([1], dtype=tl.int32), shape[perm][:-1]]))
    output_offset = tl.sum(permuted_index * output_strides)

    # Load from input and store to output
    input_value = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + output_offset, input_value, mask=mask)

def permute_copy(input, dims):
    # Ensure dims is a tuple
    if not isinstance(dims, tuple):
        dims = tuple(dims)

    # Calculate the shape of the output tensor
    input_shape = input.shape
    output_shape = tuple(input_shape[dim] for dim in dims)

    # Create the output tensor
    output = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Launch the Triton kernel
    numel = input.numel()
    BLOCK_SIZE = 1024  # Define a block size, adjust as needed
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    permute_kernel[grid](input, output, input_shape, dims, numel, BLOCK_SIZE=BLOCK_SIZE)

    return output

# Example usage:
x = torch.randn(2, 3, 4)
y = permute_copy(x, (2, 0, 1))
print(y.shape)  # Should print torch.Size([4, 2, 3])
