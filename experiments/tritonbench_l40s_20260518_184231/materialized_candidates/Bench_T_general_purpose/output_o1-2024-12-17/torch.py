import torch
import triton
import triton.language as tl

@triton.jit
def _permute_copy_kernel(
    input_ptr, output_ptr,
    dims_ptr, shape_ptr, rank,
    in_strides_ptr, out_strides_ptr,
    numel,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < numel
    
    # Convert global linear index to multi-dimensional indices in input
    # and use dims to map to correct output position
    linear_idx = offsets
    # Temporary array to store multi-dimensional indices
    idxs = tl.zeros([BLOCK_SIZE, 32], dtype=tl.int32)  # Hardcoded upper bound for rank
    # Compute multi-dimensional index for the original (non-permuted) shape
    for i in range(32):
        cond = i < rank
        size_i = tl.load(shape_ptr + i, mask=cond)
        remainder = linear_idx % size_i
        idxs = tl.where(cond[:, None], idxs, idxs)  # no-op for shape
        idxs[:, i] = remainder
        linear_idx = (linear_idx - remainder) // size_i

    # Compute input offset
    in_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    for i in range(32):
        cond = i < rank
        stride_i = tl.load(in_strides_ptr + i, mask=cond, other=0)
        in_offsets += stride_i * idxs[:, i]

    # Reorder indices according to dims, then compute output offset
    out_offsets = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    for i in range(32):
        cond = i < rank
        # dims[i] tells us which index from idxs to use
        d = tl.load(dims_ptr + i, mask=cond)
        stride_d = tl.load(out_strides_ptr + i, mask=cond, other=0)
        out_offsets += stride_d * idxs[:, d]
    
    values = tl.load(input_ptr + in_offsets, mask=mask, other=0.0)
    tl.store(output_ptr + out_offsets, values, mask=mask)


def permute_copy(input: torch.Tensor, dims) -> torch.Tensor:
    # Create output tensor with permuted shape
    out_shape = [input.shape[d] for d in dims]
    out = torch.empty(out_shape, dtype=input.dtype, device=input.device)
    
    # Prepare data for kernel launch
    rank = input.ndim
    numel = input.numel()
    
    # Convert shapes/strides/dims to torch.IntTensor for pointer access
    shape_t = torch.tensor(input.shape, dtype=torch.int32, device=input.device)
    dims_t = torch.tensor(dims, dtype=torch.int32, device=input.device)
    in_strides_t = torch.tensor(input.stride(), dtype=torch.int64, device=input.device)
    out_strides_t = torch.tensor(out.stride(), dtype=torch.int64, device=input.device)
    
    BLOCK_SIZE = 1024
    grid = ( (numel + BLOCK_SIZE - 1) // BLOCK_SIZE, )
    
    _permute_copy_kernel[grid](
        input.data_ptr(),
        out.data_ptr(),
        dims_t.data_ptr(),
        shape_t.data_ptr(),
        rank,
        in_strides_t.data_ptr(),
        out_strides_t.data_ptr(),
        numel,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return out
