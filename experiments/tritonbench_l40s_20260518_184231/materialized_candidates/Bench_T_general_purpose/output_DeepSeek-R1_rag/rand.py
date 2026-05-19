import torch
import triton
import triton.language as tl
from typing import Optional, Union

@triton.jit
def _seeded_uniform_triton(
    out_ptr: torch.Tensor,
    seed_ptr: torch.Tensor,
    out_row_stride: int,
    out_3d_stride: int,
    seed_row_stride: int,
    n_rows: int,
    n_3d: int,
    n_cols: int,
    n_slices: tl.constexpr,
    block_size: tl.constexpr,
):
    tl.static_assert(n_slices > 0 and n_slices <= 4, "0 < n_slices <= 4")

    row_idx = tl.program_id(axis=0)
    three_d_idx = tl.program_id(axis=1)

    philox_offsets = tl.arange(0, block_size)
    seed = tl.load(seed_ptr + row_idx * seed_row_stride)
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

def rand(
    *size,
    generator: Optional[torch.Generator] = None,
    out: Optional[torch.Tensor] = None,
    dtype: Optional[torch.dtype] = None,
    layout=torch.strided,
    device: Optional[Union[torch.device, str]] = None,
    requires_grad: bool = False,
    pin_memory: bool = False,
) -> torch.Tensor:
    if layout != torch.strided:
        raise ValueError("Only strided layout is supported")

    output_dtype = dtype if dtype is not None else torch.get_default_dtype()
    needs_cast = output_dtype != torch.float32

    if out is not None:
        if out.shape != size:
            raise ValueError("shape of out and size must be the same")
        if out.layout != layout:
            raise ValueError("out tensor has incorrect layout")
        device = out.device if device is None else torch.device(device)
        tmp_out = out if out.dtype == torch.float32 else torch.empty(size, dtype=torch.float32, device=device)
    else:
        device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        tmp_out = torch.empty(size, dtype=torch.float32, layout=layout, device=device, pin_memory=pin_memory)

    n_dims = len(size)
    if n_dims == 3:
        n_rows, n_3d, n_cols = size
        stride_row = tmp_out.stride(0)
        stride_3d = tmp_out.stride(1)
    elif n_dims == 2:
        n_rows, n_cols = size
        n_3d = 1
        stride_row = tmp_out.stride(0)
        stride_3d = 1
    else:
        n_rows = 1
        n_3d = 1
        n_cols = size[0] if size else 0
        stride_row = 1
        stride_3d = 1

    if generator is None:
        generator = torch.default_generator

    seeds = torch.randint(0, 2**63, (n_rows,), generator=generator, dtype=torch.int64)
    seeds = seeds.to(device=device)

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

    grid = (n_rows, n_3d)
    _seeded_uniform_triton[grid](
        tmp_out,
        seeds,
        stride_row,
        stride_3d,
        seeds.stride(0),
        n_rows,
        n_3d,
        n_cols,
        n_slices=n_slices,
        num_warps=num_warps,
        block_size=philox_block_size,
    )

    if out is not None and out.dtype != torch.float32:
        out.copy_(tmp_out.to(dtype=output_dtype))
        result = out
    else:
        result = tmp_out.to(dtype=output_dtype) if needs_cast else tmp_out

    result.requires_grad = requires_grad
    return result
