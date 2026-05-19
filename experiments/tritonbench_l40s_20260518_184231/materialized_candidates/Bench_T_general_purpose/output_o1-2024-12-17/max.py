import triton
import triton.language as tl

@triton.jit
def _max_reduce_kernel(
    input_ptr, 
    output_vals_ptr, 
    output_inds_ptr, 
    n_elements, 
    stride, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    # Compute the offset for this program
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offset < n_elements

    # Load data
    data = tl.load(input_ptr + offset * stride, mask=mask, other=-float('inf'))

    # Keep track of partial maximum and indices
    max_vals = data
    max_inds = offset

    # Reduction in warp
    for i in [16, 8, 4, 2, 1]:
        temp_vals = tl.broadcast_to(max_vals, [BLOCK_SIZE])
        temp_inds = tl.broadcast_to(max_inds, [BLOCK_SIZE])
        shfl_vals = tl.shuffle_xor(temp_vals, i)
        shfl_inds = tl.shuffle_xor(temp_inds, i)
        cond = shfl_vals > max_vals
        eq_cond = (shfl_vals == max_vals) & (shfl_inds < max_inds)
        max_vals = tl.where(cond | eq_cond, shfl_vals, max_vals)
        max_inds = tl.where(cond | eq_cond, shfl_inds, max_inds)

    # Write out result of each block
    if tl.thread_id_x() == 0:
        tl.store(output_vals_ptr + pid, max_vals[0])
        tl.store(output_inds_ptr + pid, max_inds[0])

def max(input, dim, keepdim=False, *, out=None):
    """
    max(input, dim, keepdim=False, *, out=None) -> (Tensor, LongTensor)
    """
    import torch

    # Ensure dimension is in range
    dim = dim if dim >= 0 else dim + input.ndim
    assert 0 <= dim < input.ndim, "Dimension out of range"

    # If out is provided, it should be a tuple of two tensors
    if out is not None:
        assert isinstance(out, (tuple, list)) and len(out) == 2, \
            "out must be a tuple of (Tensor, Tensor)"
        out_vals, out_inds = out
    else:
        out_vals = None
        out_inds = None

    # Permute input so that reduction dimension is last
    # This simplifies the kernel to reduce along the last dimension
    perm = list(range(input.ndim))
    perm[dim], perm[-1] = perm[-1], perm[dim]
    inp_perm = input.permute(perm)
    shape = inp_perm.shape
    reduce_size = shape[-1]
    out_size = 1

    # Flatten all but the last dimension
    leading_dims = shape[:-1]
    num_rows = 1
    for sz in leading_dims:
        num_rows *= sz

    # Cast to contiguous float32 for safety
    inp_contig = inp_perm.contiguous().float()
    inp_ptr = inp_contig.data_ptr()

    # Allocate output
    if out_vals is None:
        out_vals_t = torch.empty((num_rows,), dtype=inp_contig.dtype, device=inp_contig.device)
    else:
        out_vals_t = out_vals.view(-1)

    if out_inds is None:
        out_inds_t = torch.empty((num_rows,), dtype=torch.long, device=inp_contig.device)
    else:
        out_inds_t = out_inds.view(-1)

    # Grid: one block per row
    grid = (num_rows,)

    # Launch kernel
    stride = 1
    BLOCK_SIZE = 1024
    triton.run(
        _max_reduce_kernel,
        grid=grid,
        num_warps=4,
        num_stages=1,
        args=[
            inp_ptr,
            out_vals_t.data_ptr(),
            out_inds_t.data_ptr(),
            reduce_size,
            stride
        ],
        constants={"BLOCK_SIZE": BLOCK_SIZE}
    )

    # Reshape results to leading_dims
    out_vals_reshaped = out_vals_t.reshape(leading_dims)
    out_inds_reshaped = out_inds_t.reshape(leading_dims)

    # Permute results back
    inv_perm = [0]*input.ndim
    for i, p in enumerate(perm):
        inv_perm[p] = i

    out_vals_final = out_vals_reshaped.permute(inv_perm)
    out_inds_final = out_inds_reshaped.permute(inv_perm)

    # If keepdim, expand the dims
    if keepdim:
        out_vals_final = out_vals_final.unsqueeze(dim)
        out_inds_final = out_inds_final.unsqueeze(dim)

    # If out was passed in, modify in-place
    if out is not None:
        out[0].copy_(out_vals_final)
        out[1].copy_(out_inds_final)
        return out[0], out[1]
    else:
        return out_vals_final, out_inds_final
