import triton
import triton.language as tl


@triton.jit
def _ifftshift_kernel(
    IN_PTR, OUT_PTR,
    n_elements,
    shifts_ptr, dims_ptr, num_shifts,
    shape_ptr, strides_ptr, ndim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    # Convert linear index to multi-dimensional index
    # Indices storage for each dimension
    idxs = [tl.zeros([BLOCK_SIZE], dtype=tl.int32) for _ in range(ndim)]
    remaining = offsets
    # shape/stride data
    shape_vals = [tl.load(shape_ptr + i) for i in range(ndim)]
    stride_vals = [tl.load(strides_ptr + i) for i in range(ndim)]

    for i in range(ndim):
        s = shape_vals[i]
        idxs[i] = remaining // s
        remaining = remaining % s

    # Shift required dims
    for i in range(num_shifts):
        d = tl.load(dims_ptr + i)
        shift_amount = tl.load(shifts_ptr + i)
        s = shape_vals[d]
        idxs[d] = (idxs[d] + shift_amount) % s

    # Compute new linear index
    linear_idx = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    partial_sum = tl.zeros([BLOCK_SIZE], dtype=tl.int32)
    for i in range(ndim):
        linear_idx = linear_idx + idxs[i] * stride_vals[i]

    # Load input data and store in output
    val = tl.load(IN_PTR + offsets, mask=mask, other=0.0)
    tl.store(OUT_PTR + linear_idx, val, mask=mask)


def ifftshift(input, dim=None):
    import torch

    # Handle dims
    if dim is None:
        dims = list(range(input.ndim))
    elif isinstance(dim, int):
        dims = [dim]
    else:
        dims = list(dim)

    # Prepare shifts for each dimension
    shape = input.shape
    shifts = []
    for d in dims:
        shifts.append(shape[d] // 2)

    # Flatten input if needed
    contig_input = input.contiguous()
    out = torch.empty_like(contig_input)
    n_elements = contig_input.numel()

    # Convert shape/strides to int32 for Triton
    shape_i32 = torch.tensor(contig_input.shape, dtype=torch.int32, device=contig_input.device)
    strides_i32 = torch.tensor(contig_input.stride(), dtype=torch.int32, device=contig_input.device)
    shifts_i32 = torch.tensor(shifts, dtype=torch.int32, device=contig_input.device)
    dims_i32 = torch.tensor(dims, dtype=torch.int32, device=contig_input.device)

    BLOCK_SIZE = 1024
    grid = ( (n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE, )

    _ifftshift_kernel[grid](
        contig_input, out,
        n_elements,
        shifts_i32, dims_i32, len(shifts),
        shape_i32, strides_i32, contig_input.ndim,
        BLOCK_SIZE=BLOCK_SIZE
    )

    return out.view(input.shape)
