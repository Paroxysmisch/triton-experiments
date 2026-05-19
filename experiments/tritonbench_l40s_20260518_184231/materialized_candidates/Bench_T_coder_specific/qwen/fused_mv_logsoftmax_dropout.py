import torch
import triton
import triton.language as tl

# Define the Triton kernel
fused_mv_logsoftmax_dropout_kernel = fused_mv_logsoftmax_dropout_kernel

# Triton kernel configuration
kernel_config = triton.Config(
    num_stages=2,
    num_warps=4,
    block_size=[32, 1, 1]
)

@torch.jit.script
def fused_mv_logsoftmax_dropout(input, vec, p=0.5, training=True, inplace=False, dim=0, *, out=None):
    assert input.dim() == 2 and vec.dim() == 1, "Input and vector dimensions must be 2 and 1 respectively"
    assert input.size(-1) == vec.size(0), "Number of columns in input must match size of vector"

    if out is None:
        out = torch.empty_like(input)

    n, m = input.shape
    stride_A_row = input.stride(0)
    stride_A_col = input.stride(1)
    stride_B = vec.stride(0)
    stride_C = out.stride(0)
    stride_D = out.stride(1)

    fused_mv_logsoftmax_dropout_kernel[
        grid=(n // kernel_config.block_size[0],),
        block=kernel_config.block_size,
        config=kernel_config,
        num_warps=kernel_config.num_warps
    ](
        input.data_ptr(), vec.data_ptr(), out.data_ptr(), out.data_ptr(),
        n, m, stride_A_row, stride_A_col, stride_B, stride_C, stride_D,
        p, training, dim
    )

    return out
