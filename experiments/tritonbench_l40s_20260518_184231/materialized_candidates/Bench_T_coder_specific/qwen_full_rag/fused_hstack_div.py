import torch
import triton
import triton.language as tl
from ..utils.shape_utils import broadcast_shapes


def get_lastdim_numels(tensors):
    last_dim_numels = []
    for tensor in tensors:
        assert tensor.ndim >= 1, "Input tensor should have at least 1 dimension"
        last_dim_numels.append(tensor.size(-1))
    return last_dim_numels


def compute_output_shape(input_shapes, divisor_shape):
    assert len(input_shapes) > 0
    assert len(divisor_shape) == 1
    output_shape = list(input_shapes[0])
    output_shape[-1] = sum([s[-1] for s in input_shapes])
    output_shape = broadcast_shapes(output_shape, divisor_shape)
    return tuple(output_shape)


@triton.jit
def fused_hstack_div_kernel(
    inputs,
    divisor,
    output,
    input_strides,
    divisor_stride,
    output_stride,
    n_inputs,
    n_rows,
    n_cols_per_input,
    _unused_1,
    _unused_2,
    idx,
    num_warps,
):
    row_idx = tl.program_id(axis=0)
    col_block_idx = tl.program_id(axis=1)
    cols_per_warp = tl.num_programs(axis=1)
    col_offset = col_block_idx * cols_per_warp + tl.arange(0, num_warps)

    mask = col_offset < n_cols_per_input

    div_mask = col_offset < n_cols_per_input

    divisor_val = tl.load(divisor + col_offset * divisor_stride, div_mask)

    for i in range(n_inputs):
        curr_input_row = (
            inputs + row_idx * input_strides[i] + col_offset * n_rows
        )
        curr_output_row = (
            output + row_idx * output_stride + col_offset * n_rows
        )

        curr_input = tl.load(curr_input_row, mask=mask)
        curr_output = curr_input / divisor_val
        tl.store(curr_output_row, curr_output, mask=mask)


def fused_hstack_div(inputs, divisor, *, rounding_mode=None, out=None):
    assert isinstance(inputs, (list, tuple))
    n_tensors = len(inputs)

    if n_tensors == 0:
        raise RuntimeError("torch.hstack(): expected a non-empty TensorList")

    if n_tensors == 1:
        return torch.tensor(inputs[0]) / divisor

    if out is None:
        out = torch.empty_like(inputs[0])

    input_shapes = [list(_.shape) for _ in inputs]
    divisor_shape = list(divisor.shape)

    output_shape = compute_output_shape(input_shapes, divisor_shape)

    out.set_(torch.reshape(out, output_shape))

    lastdim_numels = get_lastdim_numels(inputs)

    div_lastdim_numel = divisor.size(divisor.ndim - 1)

    total_rows = 1
    for d in lastdim_numels[:-1]:
        total_rows *= d

    n_rows = total_rows
    n_cols_per_input = lastdim_numels[-1]
    div_n_cols = div_lastdim_numel

    inputs = torch.stack(inputs, 0).reshape((n_rows, n_cols_per_input))
    divisor = torch.reshape(divisor, (1, div_n_cols))

    N = 64
    num_warps = 8

    grid = (
        n_rows,
        triton.cdiv(n_cols_per_input, N),
    )

    fused_hstack_div_kernel[grid](
        inputs,
        divisor,
        out,
        lastdim_numels,
        div_lastdim_numel,
        1,
        n_tensors,
        lastdim_numels[-1],
        N,
        num_warps=num_warps,
        idx=0,
    )

    return out
