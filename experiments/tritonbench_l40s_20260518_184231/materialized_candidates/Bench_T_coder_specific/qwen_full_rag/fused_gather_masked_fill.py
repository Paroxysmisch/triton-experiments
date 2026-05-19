import torch
import triton
import triton.language as tl

@triton.jit
def fused_gather_masked_fill_kernel(X, I, O, mask, value, n_elements, CACHE_KEY_SIZE: tl.constexpr):
    idx = tl.arange(0, CACHE_KEY_SIZE)
    mask_vals = tl.load(mask + idx, mask=idx < n_elements, other=0)
    value_vals = tl.full((CACHE_KEY_SIZE,), value, dtype=X.dtype.element_ty)
    for i in range(n_elements):
        can_write = mask_vals[i % CACHE_KEY_SIZE] == 0
        x_val = tl.load(X + i, eviction_policy="evict_last")
        idx_to_store = i if can_write else i - 1
        tl.store(O + idx_to_store, x_val, mask=can_write)

def fused_gather_masked_fill(
    input: torch.Tensor,
    dim: int,
    index: torch.LongTensor,
    mask: torch.BoolTensor,
    value: float,
    *,
    sparse_grad: bool = False,
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    f_name = "fused_gather_masked_fill"

    assert (
        input.size(index.dim()) == index.numel()
    ), f"{f_name} expected `index.size({index.dim()}) ({index.size(index.dim())})` to equal `index.numel() ({index.numel()})`"
    assert input.ndim >= index.ndim, f"{f_name} only accepts `dim` such that `input.ndim >= index.ndim; got {input.ndim} < {index.ndim}`"
    assert (
        input.device == index.device
    ), f"{f_name} expected `input.device ({input.device})` to match `index.device ({index.device})`"
    assert input.is_contiguous(), f"{f_name} expected `input` to be contiguous but it isn't"
    assert index.is_contiguous(), f"{f_name} expected `index` to be contiguous but it isn't"
    assert mask.is_contiguous(), f"{f_name} expected `mask` to be contiguous but it isn't"

    input_shape_list = list(input.shape)
    input_strides_list = list(input.stride())
    dim_mod_input_ndim = dim % input.ndim
    inner_size = math.prod(input_shape_list[dim_mod_input_ndim:])
    outer_size = math.prod(input_shape_list[:dim_mod_input_ndim])
    n_dims_diff = index.ndim - input.ndim

    if n_dims_diff != 0:
        if n_dims_diff > 0:
            ones_list = [1 for _ in range(n_dims_diff)]
            input_shape_list = ones_list + input_shape_list
            input_strides_list = ones_list + input_strides_list
        else:
            input_shape_list = input_shape_list[-index.ndim:]
            input_strides_list = input_strides_list[-index.ndim:]

    assert all(
        size <= 1 for size in input_shape_list[index.ndim - 1 :]
    ), f"{f_name} expected `index.size({index.ndim}-1 : {index.ndim})` to contain only sizes 1 but got {input_shape_list[index.ndim - 1 :]}"

    index_flat = index.view(-1)
    input_view_shape = input_shape_list[: index.ndim - 1] + [inner_size]
    input_view = StridedBuffer(input, input_view_shape, input_strides_list)
    out_buffer = torch.empty_like(input_view, dtype=input.dtype)
    n_elements = index_flat.numel()

    grid_fn = lambda meta: (triton.cdiv(n_elements, meta["CACHE_KEY_SIZE"]),)

    fused_gather_masked_fill_kernel[grid_fn](input_view, index_flat, out_buffer, mask, value, n_elements, CACHE_KEY_SIZE=n_elements)

    out_buffer = out_buffer.view(input_shape_list)
    if out is None:
        return out_buffer
    else:
        assert (
            out.shape == out_buffer.shape
        ), f"{f_name} expected `out.shape ({out.shape})` to equal `out_buffer.shape ({out_buffer.shape})`"
        assert (
            out.strides == out_buffer.strides
        ), f"{f_name} expected `out.strides ({out.strides})` to equal `out_buffer.strides ({out_buffer.strides})`"
        out.copy_(out_buffer)
        return out
