import torch
import triton
import triton.language as tl

@triton.jit
def permute_copy_kernel(
    input_ptr,
    output_ptr,
    input_strides_ptr,
    output_shape_ptr,
    inv_perm_ptr,
    ndim,
    input_numel,
    output_numel,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < output_numel

    # Load inverse permutation, output shape, and input strides
    inv_perm = tl.zeros((8,), dtype=tl.int32)
    output_shape = tl.zeros((8,), dtype=tl.int32)
    input_strides = tl.zeros((8,), dtype=tl.int32)
    for i in tl.static_range(8):
        if i < ndim:
            inv_perm = tl.load(inv_perm_ptr + i)
            output_shape = tl.load(output_shape_ptr + i)
            input_strides = tl.load(input_strides_ptr + i)

    # Compute output strides for contiguous tensor
    output_strides = tl.zeros((8,), dtype=tl.int32)
    stride = 1
    for i in tl.static_range(7, -1, -1):
        if i < ndim:
            output_strides = stride
            stride *= output_shape[i]

    # Calculate input offsets
    input_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32)
    for idx in tl.static_range(BLOCK_SIZE):
        if mask[idx]:
            linear_offset = offsets[idx]
            remaining = linear_offset
            output_indices = tl.zeros((8,), dtype=tl.int32)
            for i in tl.static_range(7, -1, -1):
                if i < ndim:
                    stride = output_strides[i]
                    if stride == 0:
                        index = 0
                    else:
                        index = remaining // stride
                        remaining = remaining % stride
                    output_indices = index
            # Compute input indices using inverse permutation
            input_indices = tl.zeros((8,), dtype=tl.int32)
            for d in tl.static_range(8):
                if d < ndim:
                    input_idx = output_indices[inv_perm[d]]
                    input_indices = input_idx
            # Calculate input offset
            input_offset = 0
            for d in tl.static_range(8):
                if d < ndim:
                    input_offset += input_indices[d] * input_strides[d]
            input_offsets = input_offset

    # Load and store
    input_vals = tl.load(input_ptr + input_offsets, mask=mask)
    tl.store(output_ptr + offsets, input_vals, mask=mask)

def permute_copy(input: torch.Tensor, dims: list[int]) -> torch.Tensor:
    # Validate dims
    assert len(dims) == input.ndim, "dims must permute all dimensions"
    assert sorted(dims) == list(range(input.ndim)), "dims must be a valid permutation"
    
    # Compute permuted shape and create output tensor
    permuted_shape = tuple(input.size(d) for d in dims)
    output = torch.empty(permuted_shape, dtype=input.dtype, device=input.device)
    ndim = input.ndim
    if ndim == 0:
        output = output.clone()
        return output
    
    # Compute inverse permutation
    inv_perm = [0] * ndim
    for i, d in enumerate(dims):
        inv_perm[d] = i
    inv_perm = torch.tensor(inv_perm, dtype=torch.int32, device=input.device)
    
    # Prepare metadata tensors
    input_strides = torch.tensor(input.stride(), dtype=torch.int32, device=input.device)
    output_shape = torch.tensor(output.shape, dtype=torch.int32, device=input.device)
    output_numel = output.numel()
    
    # Launch kernel
    BLOCK_SIZE = 1024
    grid = lambda meta: (triton.cdiv(output_numel, meta['BLOCK_SIZE']),)
    permute_copy_kernel[grid](
        input.data_ptr(),
        output.data_ptr(),
        input_strides.data_ptr(),
        output_shape.data_ptr(),
        inv_perm.data_ptr(),
        ndim,
        input.numel(),
        output_numel,
        BLOCK_SIZE=BLOCK_SIZE,
    )
    return output
