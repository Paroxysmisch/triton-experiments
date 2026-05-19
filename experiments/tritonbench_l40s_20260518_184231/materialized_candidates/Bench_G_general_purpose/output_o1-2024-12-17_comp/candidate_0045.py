import torch
import triton
import triton.language as tl

@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    input_ptr, 
    output_ptr,
    n_elements,
    scalar,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    start = pid * BLOCK_SIZE
    offsets = start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(input_ptr + offsets, mask=mask, other=0.0)
    y = tl.pow(x, scalar)
    tl.store(output_ptr + offsets, y, mask=mask)


def pow_func_scalar_tensor_wrapper_rank_1(input_data, scalar, output_data=None):
    if isinstance(input_data, torch.Tensor):
        input_ptr = input_data.data_ptr()
        n_elements = input_data.numel()
        dtype = input_data.dtype
        device = input_data.device
    else:
        input_ptr = input_data.ptr
        n_elements = input_data.size
        dtype = input_data.dtype
        device = input_data.device

    if output_data is None:
        if isinstance(input_data, torch.Tensor):
            output_data = torch.empty_like(input_data)
        else:
            output_data = torch.empty(n_elements, dtype=dtype, device=device)

    if isinstance(output_data, torch.Tensor):
        output_ptr = output_data.data_ptr()
    else:
        output_ptr = output_data.ptr

    BLOCK_SIZE = 1024
    num_warps = 4
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)

    pow_func_scalar_tensor_kernel_rank_1[grid](
        input_ptr,
        output_ptr,
        n_elements,
        scalar,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return output_data
