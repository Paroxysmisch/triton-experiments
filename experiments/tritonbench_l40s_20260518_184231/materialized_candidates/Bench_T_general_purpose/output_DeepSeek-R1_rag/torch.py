import torch
import triton
import triton.language as tl

@triton.jit
def permute_copy_kernel(
    input_ptr,
    output_ptr,
    input_ndim,
    input_shape_ptr,
    input_strides_ptr,
    inv_dims_ptr,
    output_numel,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    for idx in range(pid * BLOCK_SIZE, (pid + 1) * BLOCK_SIZE):
        if idx >= output_numel:
            return

        # Load metadata
        input_ndim = tl.load(input_ndim)
        input_shape = tl.load(input_shape_ptr + tl.arange(0, input_ndim))
        input_strides = tl.load(input_strides_ptr + tl.arange(0, input_ndim))
        inv_dims = tl.load(inv_dims_ptr + tl.arange(0, input_ndim))

        # Compute output indices
        output_indices = []
        remainder = idx
        for dim in range(input_ndim):
            stride = tl.prod(input_shape[dim+1:])  # output is contiguous
            output_indices.append(remainder // stride)
            remainder = remainder % stride

        # Apply inverse permutation to get input indices
        input_indices = [output_indices[inv_dims[dim]] for dim in range(input_ndim)]

        # Compute input offset
        input_offset = 0
        for dim in range(input_ndim):
            input_offset += input_indices[dim] * input_strides[dim]

        # Load and store
        val = tl.load(input_ptr + input_offset)
        tl.store(output_ptr + idx, val)

def permute_copy(input: torch.Tensor, dims):
    # Compute output shape
    output_shape = tuple(input.size(d) for d in dims)
    output = torch.empty(output_shape, device=input.device, dtype=input.dtype)

    # Compute inverse permutation
    inv_dims = [dims.index(i) for i in range(len(dims))]

    # Prepare metadata for the kernel
    input_ndim = input.ndim
    input_shape = torch.tensor(input.shape, device='cuda', dtype=torch.int64)
    input_strides = torch.tensor(input.stride(), device='cuda', dtype=torch.int64)
    inv_dims_tensor = torch.tensor(inv_dims, device='cuda', dtype=torch.int64)

    output_numel = output.numel()

    # Launch kernel
    BLOCK_SIZE = 128
    grid = (triton.cdiv(output_numel, BLOCK_SIZE),)
    permute_copy_kernel[grid](
        input_ptr=input,
        output_ptr=output,
        input_ndim=input_ndim,
        input_shape_ptr=input_shape,
        input_strides_ptr=input_strides,
        inv_dims_ptr=inv_dims_tensor,
        output_numel=output_numel,
        BLOCK_SIZE=BLOCK_SIZE
    )
    return output
