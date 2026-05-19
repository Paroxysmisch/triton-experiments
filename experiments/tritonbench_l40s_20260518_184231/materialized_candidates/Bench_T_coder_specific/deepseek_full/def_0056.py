import torch
import triton
import triton.language as tl

@triton.jit
def fused_fractional_max_pool2d_with_relu(
    input: torch.Tensor,
    kernel_size,
    output_size=None,
    output_ratio=None,
    return_indices=False,
):
    input_ = tl.where(input > 0, input, 0)
    if output_size is not None:
        output_size_arg = output_size
    elif output_ratio is not None:
        output_size_arg = tuple(
            int(dim * ratio) for dim, ratio in zip(input.shape[2:], output_ratio)
        )
    else:
        output_size_arg = None
    if triton.version.full_version >= "2.1.0":
        if return_indices:
            return tl.nn.fractional_max_pool2d(input_, kernel_size, output_size=output_size_arg, return_indices=return_indices)
        else:
            return tl.nn.fractional_max_pool2d(input_, kernel_size, output_size=output_size_arg)
    else:
        if return_indices:
            return tl.nn.fractional_max_pool2d(input_, kernel_size, output_size=output_size_arg, return_indices=return_indices, _allow_tf32=False)
        else:
            return tl.nn.fractional_max_pool2d(input_, kernel_size, output_size=output_size_arg, _allow_tf32=False)
