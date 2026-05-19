import triton
import triton.language as tl
import torch

@triton.jit
def _min_reduce_kernel(
    input_ptr, output_min_ptr, output_idx_ptr,
    stride_in, stride_out_min, stride_out_idx,
    dim_size, BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Each program processes one row in the dimension being reduced
    offs = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Load initial values
    min_val = tl.load(input_ptr + pid * stride_in + 0, mask=offs < dim_size, other=float("inf"))
    min_idx = tl.zeros([BLOCK_SIZE], tl.int32)

    # Iterate over all elements in the reduction dimension
    for i in range(1, dim_size):
        val = tl.load(input_ptr + pid * stride_in + i, mask=offs < dim_size, other=float("inf"))
        cond = val < min_val
        min_val = tl.where(cond, val, min_val)
        min_idx = tl.where(cond, i, min_idx)

    # Store results
    if offs[0] < dim_size:  # only store once per row
        tl.store(output_min_ptr + pid * stride_out_min, min_val[0])
        tl.store(output_idx_ptr + pid * stride_out_idx, min_idx[0])


def min(input, dim, keepdim=False, *, out=None):
    # Validate dimension
    if dim < 0:
        dim += input.dim()
    assert 0 <= dim < input.dim(), "Dimension out of range"

    # Prepare output shape
    out_shape = list(input.shape)
    out_shape[dim] = 1 if keepdim else 0  # 0 will be removed later if not keepdim

    # Allocate or use 'out' if provided
    if out is not None:
        out_min, out_idx = out
    else:
        out_min = torch.empty_like(input.select(dim, 0), dtype=input.dtype)
        out_idx = torch.empty_like(input.select(dim, 0), dtype=torch.long)

    # Flatten input for row-wise reduce along 'dim' by permuting
    # so that 'dim' is merged into the last dimension of shape [rest, reduce_size]
    perm = list(range(input.dim()))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    input_ = input.permute(perm).contiguous()
    rows = input_.numel() // input_.size(-1)
    reduce_size = input_.size(-1)

    # Launch Triton kernel
    grid = (rows,)
    _min_reduce_kernel[grid](
        input_.data_ptr(),
        out_min.contiguous().data_ptr(),
        out_idx.contiguous().data_ptr(),
        input_.stride(-2) if input_.dim() > 1 else reduce_size,
        out_min.stride(0),
        out_idx.stride(0),
        reduce_size,
        BLOCK_SIZE=1
    )

    # Reshape/permute outputs back
    # Expand or squeeze dimension as needed
    out_min_shape = list(input.shape)
    out_min_shape[dim] = 1
    out_idx_shape = list(input.shape)
    out_idx_shape[dim] = 1

    out_min_ = out_min.view(*out_min_shape).permute(*perm).contiguous()
    out_idx_ = out_idx.view(*out_idx_shape).permute(*perm).contiguous()

    if not keepdim:
        out_min_ = out_min_.squeeze(dim)
        out_idx_ = out_idx_.squeeze(dim)

    if out is not None:
        out_min.copy_(out_min_)
        out_idx.copy_(out_idx_)
        return out
    else:
        return out_min_, out_idx_
