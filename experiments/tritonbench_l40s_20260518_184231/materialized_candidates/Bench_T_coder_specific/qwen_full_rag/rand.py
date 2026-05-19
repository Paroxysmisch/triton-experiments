import torch
import triton
import triton.language as tl


def rand(*size,
         generator=None,
         out=None,
         dtype=None,
         layout=None,
         device=None,
         requires_grad=False,
         pin_memory=None):
    if out is None:
        out = torch.empty(*size,
                          dtype=dtype,
                          device=device,
                          pin_memory=pin_memory)
    elif out.shape != size:
        raise ValueError("shape of out and size must be the same")
    if dtype is None:
        dtype = out.dtype
    if device is None:
        device = out.device

    n_dims = len(size)

    if n_dims > 3:
        raise ValueError("rand only supports up to 3D tensors")

    if n_dims == 3:
        n_rows, n_3d, n_cols = out.shape
        stride_row = out.stride(0)
        stride_3d = out.stride(1)
    elif n_dims == 2:
        n_rows, n_cols = out.shape
        n_3d = 1
        stride_row = out.stride(0)
        stride_3d = 1
    else:
        n_cols = out.shape[0]
        n_rows = 1
        n_3d = 1
        stride_row = 1
        stride_3d = 1

    full_block_size = triton.next_power_of_2(n_cols)
    philox_block_size = max(full_block_size // 4, 1)
    n_slices = full_block_size // philox_block_size
    num_warps = 4
    if philox_block_size >= 8192:
        num_warps = 32
    elif philox_block_size >= 4096:
        num_warps = 16
    elif philox_block_size >= 2048:
        num_warps = 8

    _rand_triton[(n_rows, n_3d)](out,
                                 stride_row,
                                 stride_3d,
                                 n_rows,
                                 n_3d,
                                 n_cols,
                                 n_slices=n_slices,
                                 num_warps=num_warps,
                                 block_size=philox_block_size)
    return out


@triton.jit
def _rand_triton(out_ptr,
                out_row_stride,
                out_3d_stride,
                n_rows,
                n_3d,
                n_cols,
                n_slices: tl.constexpr,
                block_size: tl.constexpr):
    """
    Generate a random float32 number in [0, 1) for each element in the output
    tensor.

    Args:
        out_ptr: The output tensor.
        out_row_stride: The stride between rows of the output tensor.
        out_3d_stride: The stride between 3D slices of the output tensor.
        n_rows: The number of rows in the output tensor.
        n_3d: The size of second dimension of the output tensor,
            if output tensor is 3D.
        n_cols: The number of columns in the output tensor.
        n_slices: The number of philox outputs to use.
    """
    tl.static_assert(n_slices > 0 and n_slices <= 4, "0 < n_slices <= 4")

    row_idx = tl.program_id(axis=0)
    three_d_idx = tl.program_id(axis=1)

    philox_offsets = tl.arange(0, block_size)
    seed = row_idx
    if three_d_idx > 0:
        seed ^= three_d_idx
    out1, out2, out3, out4 = tl.rand4x(seed, philox_offsets)

    output_row_start_ptr = (out_ptr + row_idx * out_row_stride +
                            three_d_idx * out_3d_stride)
    out1_offsets = philox_offsets
    tl.store(output_row_start_ptr + out1_offsets,
             out1,
             mask=out1_offsets < n_cols)
    if n_slices > 1:
        out2_offsets = tl.arange(block_size, block_size * 2)
        tl.store(output_row_start_ptr + out2_offsets,
                 out2,
                 mask=out2_offsets < n_cols)
    if n_slices > 2:
        out3_offsets = tl.arange(block_size * 2, block_size * 3)
        tl.store(output_row_start_ptr + out3_offsets,
                 out3,
                 mask=out3_offsets < n_cols)
    if n_slices > 3:
        out4_offsets = tl.arange(block_size * 3, block_size * 4)
        tl.store(output_row_start_ptr + out4_offsets,
                 out4,
                 mask=out4_offsets < n_cols)
