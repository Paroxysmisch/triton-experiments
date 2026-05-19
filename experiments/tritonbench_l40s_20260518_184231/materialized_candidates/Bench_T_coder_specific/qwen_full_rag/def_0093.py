import torch
import triton
import triton.language as tl


@triton.jit
def softmax_log_kernel(
    input_ptr, output_ptr, stride_input_n, stride_output_n, stride_output_c,
    input_numel, input_inner, output_inner, NUM_WARPS: tl.constexpr,
    BLOCK_SIZE: tl.constexpr, LOG_SOFTMAX: tl.constexpr = False,
):
    """
    Args:
        input_ptr: pointer to the input data
        output_ptr: pointer to the output data
        stride_input_n: stride of input at n dimension
        stride_input_c: stride of input at feature dimension
        stride_output_n: stride of output at n dimension
        stride_output_c: stride of output at feature dimension
        input_numel: number of elements in input
        input_inner: inner loop size in input
        output_inner: inner loop size in output
        NUM_WARPS: number of warps
        BLOCK_SIZE: block size
    """
    pid = tl.program_id(axis=0)
    n_index = tl.program_id(axis=1)
    input_n_offset = n_index * stride_input_n
    output_n_offset = n_index * stride_output_n

    feature_block_ptr = tl.make_block_ptr(
        base=input_ptr + input_n_offset,
        shape=(input_inner,),
        strides=(stride_input_c, ),
        offsets=(pid * BLOCK_SIZE, ),
        block_shape=(BLOCK_SIZE, ),
        order=(0, ),
    )
    feature_ptrs = tl.block_ptr_to_memrefs(feature_block_ptr)
    feature_mask = (feature_block_ptr.off_v <= (input_inner - BLOCK_SIZE))
    x = tl.load(feature_ptrs, boundary_check=(0, ), eviction_policy="evict_last")
    if LOG_SOFTMAX:
        c = tl.max(x, axis=0)
        x = tl.where(x == c, x - c, x)
        x = tl.where(feature_mask, x, float("-inf"))
        exp_x = tl.exp(x)
        y = tl.sum(exp_x, axis=0)
        y = tl.where(feature_mask, y, 1.0)
        z = tl.log(y)
        x = x - z
    else:
        x = tl.where(feature_mask, x, float("-inf"))
        exp_x = tl.exp(x)
        y = tl.sum(exp_x, axis=0)
        y = tl.where(feature_mask, y, 1e-12)
        x = x / tl.log(y)

    output_feature_ptr = tl.make_block_ptr(
        base=output_ptr + output_n_offset,
        shape=(output_inner, ),
        strides=(stride_output_c, ),
        offsets=(pid * BLOCK_SIZE, ),
        block_shape=(BLOCK_SIZE, ),
        order=(0, ),
    )
    output_ptrs = tl.block_ptr_to_memrefs(output_feature_ptr)
    tl.store(output_ptrs, x, boundary_check=(0, ))


def softmax_log(
    input: torch.Tensor,
    dim: int = -1,
    dtype: torch.dtype = None,
    eps: float = 1e-12,
    log_softmax: bool = False,
) -> torch.Tensor:
    """Similar to PyTorch's softmax, but uses a stable implementation.

    Args:
        input (torch.Tensor): Input tensor.
        dim (int): Dimension to apply softmax on.
        dtype (torch.dtype, optional): Desired data type of the output tensor.
            If specified, casts the input tensor to this data type before computation.
            This is useful for preventing data type overflows.
        eps (float, optional): Minimum value that the result can take. Defaults to 1e-12.
        log_softmax (bool, optional): If True, returns the log of softmax instead of softmax.
            Defaults to False.

    Returns:
        torch.Tensor: Output tensor.
    """
    if dtype is not None:
        input = input.to(dtype)

    if input.is_floating_point() is False:
        input = input.to(torch.get_default_dtype())

    shape = list(input.shape)
    dim = dim % input.ndim
    input_numel = input.numel()
    input_strides = list(input.stride())
    input_inner = shape.pop(dim)
    input_outer = math.prod(shape)
    stride_input_n = input_strides[dim]
    stride_input_c = input_strides[dim - 1] if (dim - 1 >= 0) else 1
    stride_output_n = input_strides[dim]
    stride_output_c = input_strides[dim - 1] if (dim - 1 >= 0) else 1

    output = torch.empty_like(input, dtype=torch.float32, memory_format=torch.contiguous)
    output_inner = output.size(output.dim() - 1)
    output_outer = output.numel() // output_inner

    BLOCK_SIZE = 1024
    num_warps = 4
    grid = lambda meta: (
        triton.cdiv(input_inner, meta["BLOCK_SIZE"]),
        output_outer,
    )

    softmax_log_kernel[grid](
        input,
        output,
        stride_input_n,
        stride_output_n,
        stride_output_c,
        input_numel,
        input_inner,
        output_inner,
        NUM_WARPS=num_warps,
        BLOCK_SIZE=BLOCK_SIZE,
        LOG_SOFTMAX=log_softmax,
    )

    if log_softmax is False:
        min_val = torch.full_like(output, fill_value=eps)
        output = torch.maximum(output, min_val)

    return output
