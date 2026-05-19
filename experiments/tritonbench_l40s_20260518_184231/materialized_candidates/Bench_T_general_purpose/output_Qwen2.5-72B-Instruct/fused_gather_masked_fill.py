import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill_kernel(
    input_ptr,  # Pointer to the input tensor
    index_ptr,  # Pointer to the index tensor
    mask_ptr,   # Pointer to the mask tensor
    output_ptr, # Pointer to the output tensor
    value,      # Value to fill in where mask is True
    dim,        # Dimension along which to index
    stride_input,  # Stride of the input tensor
    stride_index,  # Stride of the index tensor
    stride_mask,   # Stride of the mask tensor
    stride_output, # Stride of the output tensor
    input_shape,   # Shape of the input tensor
    index_shape,   # Shape of the index tensor
    mask_shape,    # Shape of the mask tensor
    output_shape,  # Shape of the output tensor
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_shape[0]

    input_offsets = tl.zeros((BLOCK_SIZE,), tl.int32)
    index_offsets = tl.zeros((BLOCK_SIZE,), tl.int32)
    mask_offsets = tl.zeros((BLOCK_SIZE,), tl.int32)

    for d in range(len(input_shape)):
        if d == dim:
            input_offsets += index_ptr[index_offsets] * stride_input[d]
        else:
            input_offsets += offsets % output_shape[d] * stride_input[d]
            index_offsets += offsets % output_shape[d] * stride_index[d]
            mask_offsets += offsets % output_shape[d] * stride_mask[d]

    input_values = tl.load(input_ptr + input_offsets, mask=mask)
    mask_values = tl.load(mask_ptr + mask_offsets, mask=mask)
    output_values = tl.where(mask_values, value, input_values)

    tl.store(output_ptr + offsets, output_values, mask=mask)

import torch
import triton
import triton.language as tl

def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    # Validate input shapes and dimensions
    if input.dim() != index.dim():
        raise ValueError("Input and index tensors must have the same number of dimensions.")
    if any(input.size(d) != index.size(d) for d in range(input.dim()) if d != dim):
        raise ValueError("The size of index at each dimension d must not exceed the size of input at that dimension, except at dimension dim.")
    if not mask.shape == input.shape:
        raise ValueError("The mask tensor must be broadcastable to the shape of the output tensor.")

    # Determine the output shape
    output_shape = list(input.shape)
    output_shape[dim] = index.shape[dim]

    # Allocate output tensor if not provided
    if out is None:
        out = torch.empty(output_shape, dtype=input.dtype, device=input.device)

    # Launch the Triton kernel
    grid = (triton.cdiv(out.numel(), 1024),)
    fused_gather_masked_fill_kernel[grid](
        input.contiguous().data_ptr(),
        index.contiguous().data_ptr(),
        mask.contiguous().data_ptr(),
        out.data_ptr(),
        value,
        dim,
        input.stride(0),
        index.stride(0),
        mask.stride(0),
        out.stride(0),
        input.shape,
        index.shape,
        mask.shape,
        out.shape,
        BLOCK_SIZE=1024
    )

    return out
