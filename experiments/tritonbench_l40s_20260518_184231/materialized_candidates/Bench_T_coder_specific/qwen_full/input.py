import torch
import triton
import triton.language as tl

@triton.jit
def triton_mean(x, axis=None, keepdim=False, return_indices=False):
    if axis is None:
        axis = list(range(x.ndim))
    elif isinstance(axis, int):
        axis = [axis]
    else:
        axis = list(axis)

    x = triton.reduce.sum(x, axis, keepdim)
    return x

def mean(input, dim=None, keepdim=False, dtype=None, out=None):
    if dtype is not None:
        input = input.to(dtype)
    if out is not None:
        assert out.shape == broadcast_shapes(
            input.shape, [input.shape[i] if i in dim else 1 for i in range(input.ndim)]
        )
        assert out.dtype == input.dtype
    if isinstance(dim, (list, tuple)):
        dim = sorted(dim)
        in_shape = list(input.shape)
        for d in dim:
            assert d >= -input.ndim and d < input.ndim
        n_dims_to_keep = len(input.shape) - len(dim)
        out_shape = [1] * len(dim) + list(input.shape)
        out_shape = out_shape[:n_dims_to_keep] + [1] * len(dim) + out_shape[n_dims_to_keep:]
        out_shape = out_shape[:n_dims_to_keep] + list(
            input.shape[i] if i < n_dims_to_keep else 1 for i in range(len(input.shape))
        )
        out_strides = [0] * len(out_shape)
        out_strides[n_dims_to_keep - 1] = 1
        for i in range(n_dims_to_keep - 2, -1, -1):
            out_strides[i] = out_strides[i + 1] * out_shape[i + 1]
        out._ strides = tuple(out_strides)
        out._ shape = tuple(out_shape)
        ret = triton_mean(input, dim, keepdim)
        assert ret.shape == broadcast_shapes(
            input.shape, [input.shape[i] if i in dim else 1 for i in range(input.ndim)]
        )
        out[...] = ret
        return out
    else:
        assert dim >= -input.ndim and dim < input.ndim
        ret = triton_mean(input, dim, keepdim)
        return ret
