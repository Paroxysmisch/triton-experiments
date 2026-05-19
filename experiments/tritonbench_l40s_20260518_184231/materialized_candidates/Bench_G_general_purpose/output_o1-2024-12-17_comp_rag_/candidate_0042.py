import triton
import triton.language as tl
import torch

# -----------------------------------------------------------------------------------------------
# Kernel: pow_func_scalar_tensor_kernel_rank_1
# This kernel computes out[i] = pow(in[i], scalar) for a rank-1 tensor.
# -----------------------------------------------------------------------------------------------
@triton.jit
def pow_func_scalar_tensor_kernel_rank_1(
    in_ptr, 
    out_ptr, 
    n_elements, 
    scalar, 
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask, other=0.0)
    y = tl.pow(x, scalar)
    tl.store(out_ptr + offsets, y, mask=mask)


# -----------------------------------------------------------------------------------------------
# Wrapper: pow_func_scalar_tensor_wrapper_rank_1
# This wrapper sets up the Triton kernel launch for computing the power of a rank-1 tensor
# by a scalar value. It handles both torch.Tensor and hypothetical StridedBuffer objects.
# -----------------------------------------------------------------------------------------------
def pow_func_scalar_tensor_wrapper_rank_1(
    input_data,
    exponent: float,
    output_data=None,
    block_size: int = 1024,
    num_warps: int = 4
):
    # Handle input as either torch.Tensor or StridedBuffer (hypothetical)
    if isinstance(input_data, torch.Tensor):
        in_ptr = input_data.data_ptr()
        n_elements = input_data.numel()
        if output_data is None:
            output_data = torch.empty_like(input_data)
        out_ptr = output_data.data_ptr()
    else:
        # For a StridedBuffer-like object with .ptr and .numel() attributes
        in_ptr = input_data.ptr
        n_elements = input_data.numel()
        if output_data is None:
            # Assume we can instantiate a similar StridedBuffer for output
            output_data = input_data.empty_like()
        out_ptr = output_data.ptr

    grid = ( (n_elements + block_size - 1) // block_size, )
    
    pow_func_scalar_tensor_kernel_rank_1[grid](
        in_ptr, 
        out_ptr, 
        n_elements, 
        exponent,
        BLOCK_SIZE=block_size,
        num_warps=num_warps
    )

    return output_data
