import torch
import triton
import triton.language as tl


@triton.jit
def _ifftshift_kernel(
    input_ptr, output_ptr,
    shape_ptr, stride_in_ptr, stride_out_ptr, shift_ptr,
    rank, n_elements,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    offset_in = tl.zeros([BLOCK_SIZE], dtype=tl.int64)
    offset_out = tl.zeros([BLOCK_SIZE], dtype=tl.int64)

    remain = offsets
    i = 0
    while i < rank:
        dim_size = tl.load(shape_ptr + i)
        st_in = tl.load(stride_in_ptr + i)
        st_out = tl.load(stride_out_ptr + i)
        sh = tl.load(shift_ptr + i)

        d = remain % dim_size
        remain = remain // dim_size
        new_d = (d + sh) % dim_size

        offset_in += d * st_in
        offset_out += new_d * st_out
        i += 1

    val = tl.load(input_ptr + offset_in, mask=mask, other=0.0)
    tl.store(output_ptr + offset_out, val, mask=mask)


def ifftshift(input, dim=None):
    # Convert dim to a tuple of dimensions
    if dim is None:
        dims = range(input.ndim)
    elif isinstance(dim, int):
        dims = (dim,)
    else:
        dims = dim

    # Prepare output tensor
    output = torch.empty_like(input)

    # Build shift array:
    # For each dimension in dims, shift = -floor(size/2)
    # For others, shift = 0
    shift_values = []
    for i in range(input.ndim):
        if i in dims:
            shift_values.append(-(input.shape[i] // 2))
        else:
            shift_values.append(0)

    # Convert to torch tensors on the same device
    shape_arr = torch.tensor(list(input.shape), dtype=torch.int64, device=input.device)
    stride_in_arr = torch.tensor(list(input.stride()), dtype=torch.int64, device=input.device)
    stride_out_arr = torch.tensor(list(output.stride()), dtype=torch
