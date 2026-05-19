import torch
import triton
import triton.language as tl

device = 'cuda:0'

@triton.jit
def permute_copy_kernel(
    input_ptr,
    output_ptr,
    strides_in,
    strides_out,
    dims,
    numel,
    BLOCK_SIZE: tl.constexpr
):
    # Calculate the global index for this program instance
    idx = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < numel

    # Calculate the input multi-dimensional index from the flat index
    input_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    remaining = idx
    for i in range(len(dims)):
        dim_size = dims[i]
        input_offset += (remaining // dim_size) * strides_in[i]
        remaining = remaining % dim_size

    # Calculate the output multi-dimensional index from the flat index
    output_offset = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    remaining = idx
    for i in range(len(dims)):
        dim_size = dims[i]
        output_offset += (remaining // dim_size) * strides_out[i]
        remaining = remaining % dim_size

    # Load from input and store to output
    input_data = tl.load(input_ptr + input_offset, mask=mask, other=0)
    tl.store(output_ptr + output_offset, input_data, mask=mask)


def permute_copy(input: torch.Tensor, dims):
    # Calculate the shape of the output tensor
    output_shape = [input.shape[dim] for dim in dims]
    numel = input.numel()

    # Prepare an output tensor on the same device
    output = torch.empty(size=output_shape, device=device, dtype=input.dtype)

    # Calculate strides for input and output tensors
    strides_in = input.stride()
    strides_out = torch.empty(output_shape, device=device).stride()

    # Ensure both input and output tensors are on the GPU
    assert input.is_cuda and output.is_cuda, 'Both tensors must be on GPU'

    # Launch the Triton kernel
    BLOCK_SIZE = 1024
    grid = (triton.cdiv(numel, BLOCK_SIZE),)
    permute_copy_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        strides_in=strides_in,
        strides_out=strides_out,
        dims=torch.tensor(dims, device=device),
        numel=numel,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return output

# Example usage:
# input_tensor = torch.randn(2, 3, 4, device=device)
# output_tensor = permute_copy(input_tensor, (2, 0, 1))
