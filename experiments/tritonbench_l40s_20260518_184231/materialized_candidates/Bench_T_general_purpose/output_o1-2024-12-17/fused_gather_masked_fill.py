import torch
import triton
import triton.language as tl

@triton.jit
def _gather_masked_fill_kernel(
    input_ptr, index_ptr, mask_ptr, out_ptr,
    input_strides_ptr, index_strides_ptr, mask_strides_ptr, out_strides_ptr,
    input_shape_ptr, index_shape_ptr,
    dim, value,
    total_elements,
    rank: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Prevent out-of-bounds
    mask_oob = block_offsets >= total_elements

    # Convert 1D index in [0, total_elements) to multi-D index
    # for the 'index' and 'mask' (and thus 'out') shape
    # We'll store the multi-D index in idx_md
    idx_md = [tl.zeros([BLOCK_SIZE], dtype=tl.int32) for _ in range(rank)]
    tmp = block_offsets
    for d in range(rank - 1, -1, -1):
        shape_d = tl.load(index_shape_ptr + d)
        idx_md[d] = tmp % shape_d
        tmp = tmp // shape_d

    # Compute offset into index, mask, out
    # offset_x = sum(idx_md[d] * index_strides[d]) for d in 0..rank
    # offset_m = sum(idx_md[d] * mask_strides[d]) for d in 0..rank
    # offset_o = sum(idx_md[d] * out_strides[d])   for d in 0..rank
    offset_i = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    offset_m = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    offset_o = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    for d in range(rank):
        stride_i = tl.load(index_strides_ptr + d)
        stride_m = tl.load(mask_strides_ptr + d)
        stride_o = tl.load(out_strides_ptr + d)
        offset_i += idx_md[d].to(tl.int64) * stride_i
        offset_m += idx_md[d].to(tl.int64) * stride_m
        offset_o += idx_md[d].to(tl.int64) * stride_o

    # Gather the index from index tensor
    gather_idx = tl.load(index_ptr + offset_i, mask=~mask_oob, other=0).to(tl.int64)

    # Now compute the offset for reading from input by substituting
    # idx_md[dim] = gather_idx
    # offset_inp = sum(idx_md[d] * input_strides[d]), except dim uses gather_idx
    offset_inp = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    for d in range(rank):
        stride_val = tl.load(input_strides_ptr + d)
        sel = tl.where(d == dim, gather_idx, idx_md[d].to(tl.int64))
        offset_inp += sel * stride_val

    # Read from input and read mask
    inp_val = tl.load(input_ptr + offset_inp, mask=~mask_oob, other=0.0)
    m_val = tl.load(mask_ptr + offset_m, mask=~mask_oob, other=0)
    # Where mask is True, we set result to value
    res_val = tl.where(m_val.to(tl.bool), value, inp_val)

    # Write to out
    tl.store(out_ptr + offset_o, res_val, mask=~mask_oob)


def fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None):
    """
    fused_gather_masked_fill(input, dim, index, mask, value, *, sparse_grad=False, out=None) -> Tensor
    
    Performs a fused operation combining torch.gather and torch.Tensor.masked_fill.
    It first gathers values from 'input' along dimension 'dim' using 'index',
    and then replaces the gathered elements with 'value' where 'mask' is True.
    """
    # Check dimensionalities
    if input.dim() != index.dim():
        raise ValueError("Input and index must have the same number of dimensions.")
    if out is not None and out.shape != index.shape:
        raise ValueError("If 'out' is provided, it must have the same shape as 'index'.")

    # Shape checks
    output_shape = index.shape
    if out is None:
        out = torch.empty_like(index, dtype=input.dtype)

    # Broadcast check for mask
    # This will raise if mask is not broadcastable
    bmask = mask.broadcast_to(output_shape)

    # Prepare data pointers
    input_ptr = input.contiguous().data_ptr()
    index_ptr = index.contiguous().data_ptr()
    mask_ptr = bmask.contiguous().data_ptr()
    out_ptr = out.contiguous().data_ptr()

    # We'll gather the shape/strides info as int64
    # rank
    rank = input.dim()

    # Convert to lists of strides
    input_strides = list(input.contiguous().stride())
    index_strides = list(index.contiguous().stride())
    mask_strides = list(bmask.contiguous().stride())
    out_strides = list(out.contiguous().stride())

    # Convert shapes
    input_shape = list(input.contiguous().shape)
    index_shape = list(index.contiguous().shape)

    # Number of output elements
    total_elems = out.numel()

    # Allocate buffers on GPU for shape/strides
    # We'll store them as int64
    input_strides_t = torch.tensor(input_strides, dtype=torch.int64, device=input.device)
    index_strides_t = torch.tensor(index_strides, dtype=torch.int64, device=input.device)
    mask_strides_t = torch.tensor(mask_strides, dtype=torch.int64, device=input.device)
    out_strides_t = torch.tensor(out_strides, dtype=torch.int64, device=input.device)
    input_shape_t = torch.tensor(input_shape, dtype=torch.int32, device=input.device)
    index_shape_t = torch.tensor(index_shape, dtype=torch.int32, device=input.device)

    # Determine block size and grid
    BLOCK_SIZE = 1024
    grid = lambda meta: ((total_elems + BLOCK_SIZE - 1) // BLOCK_SIZE,)

    # Launch kernel
    _gather_masked_fill_kernel[grid](
        input_ptr, index_ptr, mask_ptr, out_ptr,
        input_strides_t, index_strides_t, mask_strides_t, out_strides_t,
        input_shape_t, index_shape_t,
        dim, float(value),
        total_elems,
        rank,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=4,
        num_stages=2,
    )

    return out
