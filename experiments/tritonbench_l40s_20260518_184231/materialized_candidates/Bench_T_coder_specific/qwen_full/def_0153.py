import torch
import triton
import triton.language as tl
from ..utils.shape_utils import volume

# Triton kernel for 2D adaptive average pooling
@triton.jit
def adaptive_avg_pool2d_kernel(
    X,
    Y,
    stride_s_0_x,
    stride_s_1_x,
    stride_s_0_y,
    stride_s_1_y,
    S_0,
    S_1,
    N,
    C,
    HS,
    WS,
    BLOCK_SIZE: tl.constexpr,
):
    pid_nc = tl.program_id(0)
    pid_c = pid_nc % C
    pid_n = pid_nc // C
    # compute offset
    offset_n = pid_n * N * C + pid_c
    offset_y = (
        offset_n * S_0 * S_1
        + tl.arange(0, BLOCK_SIZE)[:, None] * S_1
        + tl.arange(0, BLOCK_SIZE)[None, :]
    )
    offset_x_block = (
        offset_n * stride_s_0_x * stride_s_1_x
        + tl.arange(0, BLOCK_SIZE)[:, None] * stride_s_0_x * WS
        + tl.arange(0, BLOCK_SIZE)[None, :] * stride_s_1_x
    )

    # compute x block ptr
    x_block_ptr = X + offset_x_block
    # compute y ptr
    y_ptr = Y + offset_y

    # do not compute for out of boundary
    x_mask = (pid_n < N) & (pid_c < C)

    # for average pooling, we accumulate in fp32
    accum = tl.zeros((BLOCK_SIZE, BLOCK_SIZE), dtype=tl.float32)

    for i in range(tl.cdiv(HS, BLOCK_SIZE)):
        for j in range(tl.cdiv(WS, BLOCK_SIZE)):
            # compute mask
            mask = (
                (i * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[:, None] < HS)
                & (j * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)[None, :] < WS)
                & x_mask
            )
            # boundary check on feature map
            x = tl.load(x_block_ptr + i * BLOCK_SIZE * stride_s_0_x * stride_s_1_x + j * BLOCK_SIZE * stride_s_0_x * stride_s_1_x, mask=mask, other=0.0)
            accum += x

            # move to the next src block
            x_block_ptr += BLOCK_SIZE * stride_s_0_x * stride_s_1_x

        # write back
        if x_mask:
            tl.store(y_ptr, accum / (HS * WS), mask=(tl.arange(0, BLOCK_SIZE)[:, None] < S_0) & (tl.arange(0, BLOCK_SIZE)[None, :] < S_1))
            y_ptr += S_0 * S_1

# Wrapper function for calling the Triton kernel
def adaptive_avg_pool2d(output_size) -> Tensor:
    def wrapper(x):
        assert x.dim() == 4 or x.dim() == 3, "only accept 3d or 4d tensor"
        if x.dim() == 3:
            x = x.unsqueeze(0)
        N, C, H, W = x.shape
        if isinstance(output_size, int):
            output_size = (output_size, output_size)
        assert isinstance(output_size, tuple) and len(output_size) == 2, "invalid output size"
        s0, s1 = output_size
        assert s0 is not None and s1 is not None, "square matrix is not supported"
        y = torch.empty((N, C, s0, s1), dtype=x.dtype, device=x.device)
        # check if kernel exists, if not, compile
        kernel, num_warps = get_kernel("adaptive_avg_pool2d", x.dtype, BLOCK_SIZE=s0)
        # launch kernel
        grid = (N * C,)
        adaptive_avg_pool2d_kernel[grid](
            x,
            y,
            stride_s_0_x=H,
            stride_s_1_x=W,
            stride_s_0_y=s0,
            stride_s_1_y=s1,
            S_0=s0,
            S_1=s1,
            N=N,
            C=C,
            HS=H,
            WS=W,
            BLOCK_SIZE=s0,
            num_warps=num_warps,
        )
        return y.squeeze(0) if N == 1 and C == 1 else y
    return wrapper
