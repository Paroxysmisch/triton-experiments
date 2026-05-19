import torch
import triton
import triton.language as tl
from triton.language.libdevice import div_rn
from .utils import compile_func_with_device
from .heuristics import heuristics_pow_func_scalar_tensor

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    output_ptr,
    output_strides,
    input_ptr,
    input_strides,
    n_elements,
    scalar,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    data_ptrs = input_ptr + offsets * input_strides
    data = tl.load(data_ptrs, mask=mask)
    result = tl.pow(data, scalar)
    result_ptr = output_ptr + offsets * output_strides
    tl.store(result_ptr, result, mask=mask)

def pow_func_scalar_tensor_wrapper_rank_1(
    input: torch.Tensor, scalar: float, *, dtype=None
) -> torch.Tensor:
    output_shape = input.shape

    if dtype is None:
        dtype = input.dtype
    else:
        dtype = torch.dtype(dtype)

    output = torch.empty(output_shape, device=input.device, dtype=dtype)

    n_elements = output.numel()
    tile_size, num_warps = heuristics_pow_func_scalar_tensor(n_elements)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    pow_func_scalar_tensor_kernel_rank_1[grid](
        output,
        output.stride(0),
        input,
        input.stride(0),
        n_elements,
        scalar,
        BLOCK_SIZE=tile_size,
        num_warps=num_warps,
    )

    return output

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1_strided(
    output_ptr,
    output_strides,
    output_size,
    input_ptr,
    input_strides,
    input_size,
    scalar,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < input_size[0]
    data_ptrs = input_ptr + offsets * input_strides
    data = tl.load(data_ptrs, mask=mask)
    result = tl.pow(data, scalar)
    result_ptr = output_ptr + offsets * output_strides
    tl.store(result_ptr, result, mask=mask)

def pow_func_scalar_tensor_wrapper_rank_1_strided(
    input: torch.StridedBuffer, scalar: float, *, dtype=None
) -> torch.Tensor:
    output_shape = (input.size(0),)

    if dtype is None:
        dtype = input.dtype
    else:
        dtype = torch.dtype(dtype)

    output = torch.empty(output_shape, device="cuda", dtype=dtype)

    n_elements = output.numel()
    tile_size, num_warps = heuristics_pow_func_scalar_tensor(n_elements)

    grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

    pow_func_scalar_tensor_kernel_rank_1_strided[grid](
        output,
        output.stride(0),
        output_shape[0],
        input,
        input.stride(0),
        input_shape[0],
        scalar,
        BLOCK_SIZE=tile_size,
        num_warps=num_warps,
    )

    return output

def pow_func_scalar_tensor(
    input: torch.Tensor | torch.StridedBuffer, scalar: float, *, dtype=None
) -> torch.Tensor:
    if isinstance(input, torch.Tensor):
        return pow_func_scalar_tensor_wrapper_rank_1(input, scalar, dtype=dtype)
    elif isinstance(input, torch.StridedBuffer):
        return pow_func_scalar_tensor_wrapper_rank_1_strided(input, scalar, dtype=dtype)
    else:
        raise ValueError("Invalid input type")
