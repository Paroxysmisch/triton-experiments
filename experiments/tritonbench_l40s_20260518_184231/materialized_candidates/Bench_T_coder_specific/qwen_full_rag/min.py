import torch
import triton
import triton.language as tl

@triton.jit
def min_func(input, other):
    mask = input < other
    if mask.any():
        return tl.min(input)
    else:
        return other

@triton.jit
def min_with_ind_func(input, dim, keepdim, n_elements, last_dim_size):
    sorted_input, indices = tl.sort(input, dim=dim)
    min_val = tl.min(sorted_input)
    if not keepdim:
        input_strides = input.stride()
        input_shape = list(input.shape)
        input_shape[dim] = 1
        input = input.reshape(input_shape)
    min_indices = indices.min(dim=dim)
    if n_elements % last_dim_size != 0:
        neutral_min = float("inf")
        min_val = tl.where(min_indices < last_dim_size, min_val, neutral_min)
        min_indices = tl.where(min_indices < last_dim_size, min_indices, 0)
    return min_val, min_indices, input

@triton.jit
def min_with_ind_func_inplace(input, dim, keepdim, n_elements, last_dim_size):
    sorted_input, indices = tl.sort(input, dim=dim)
    min_val = tl.min(sorted_input)
    if not keepdim:
        input_strides = input.stride()
        input_shape = list(input.shape)
        input_shape[dim] = 1
        input = input.reshape(input_shape)
    min_indices = indices.min(dim=dim)
    if n_elements % last_dim_size != 0:
        neutral_min = float("inf")
        min_val = tl.where(min_indices < last_dim_size, min_val, neutral_min)
        min_indices = tl.where(min_indices < last_dim_size, min_indices, 0)
    return min_val, min_indices

def min(input, dim=None, keepdim=False, *, out=None):
    if dim is None:
        input = input.flatten()
        dim = 0
    else:
        if input.strides[dim] != 1:
            input = input.contiguous()
    n_elements = input.numel()
    last_dim_size = input.shape[dim]
    if out is None:
        min_result = torch.full([], dtype=input.dtype, device=input.device)
        min_ind_result = torch.full([], dtype=torch.int64, device=input.device)
    else:
        assert len(out) == 2
        min_result, min_ind_result = out
    if n_elements == 0:
        assert min_result.ndim == 0
        assert min_ind_result.ndim == 0
        return min_result, min_ind_result
    if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        keepdim = True
    if (
        input.dtype == torch.float32
        or input.dtype == torch.float64
        or input.dtype == torch.bfloat16
        or input.dtype == torch.float16
    ):
        return min_with_ind_func(input, dim, keepdim, n_elements, last_dim_size)
    else:
        min_fn = lambda other: min_func(input, other)
        return torch._C._foreach_reduce(min_fn, (input,), dim, keepdim, out)

def min_outofplace(input, dim=None, keepdim=False):
    if dim is None:
        input = input.flatten()
        dim = 0
    else:
        if input.strides[dim] != 1:
            input = input.contiguous()
    n_elements = input.numel()
    last_dim_size = input.shape[dim]
    min_result = torch.full([], dtype=input.dtype, device=input.device)
    min_ind_result = torch.full([], dtype=torch.int64, device=input.device)
    if n_elements == 0:
        assert min_result.ndim == 0
        assert min_ind_result.ndim == 0
        return min_result, min_ind_result
    if input.dtype in [torch.int8, torch.int16, torch.int32, torch.int64]:
        keepdim = True
    if (
        input.dtype == torch.float32
        or input.dtype == torch.float64
        or input.dtype == torch.bfloat16
        or input.dtype == torch.float16
    ):
        return min_with_ind_func_inplace(input, dim, keepdim, n_elements, last_dim_size)
    else:
        min_fn = lambda other: min_func(input, other)
        return torch._C._foreach_reduce_out_of_place(
            min_fn, (input,), dim, keepdim, (min_result, min_ind_result)
        )
