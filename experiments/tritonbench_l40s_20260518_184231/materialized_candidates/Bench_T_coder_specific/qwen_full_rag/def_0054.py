import paddle
import triton
import triton.language as tl
from ...utils import get_kernel_meta


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 32}),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 64}),
        triton.Config({"BLOCK_ROW_SIZE": 128, "BLOCK_COL_SIZE": 128}),
        triton.Config({"BLOCK_ROW_SIZE": 256, "BLOCK_COL_SIZE": 256}),
        triton.Config({"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 64}),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 128}),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 256}),
        triton.Config({"BLOCK_ROW_SIZE": 128, "BLOCK_COL_SIZE": 256}),
        triton.Config({"BLOCK_ROW_SIZE": 256, "BLOCK_COL_SIZE": 512}),
        triton.Config({"BLOCK_ROW_SIZE": 512, "BLOCK_COL_SIZE": 512}),
        triton.Config({"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 32}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 64}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 128, "BLOCK_COL_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 256, "BLOCK_COL_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 64}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 128}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 128, "BLOCK_COL_SIZE": 256}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 256, "BLOCK_COL_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 512, "BLOCK_COL_SIZE": 512}, num_warps=4),
        triton.Config({"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 32}, num_warps=8),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 64}, num_warps=8),
        triton.Config({"BLOCK_ROW_SIZE": 128, "BLOCK_COL_SIZE": 128}, num_warps=8),
        triton.Config({"BLOCK_ROW_SIZE": 256, "BLOCK_COL_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_ROW_SIZE": 32, "BLOCK_COL_SIZE": 64}, num_warps=8),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 128}, num_warps=8),
        triton.Config({"BLOCK_ROW_SIZE": 64, "BLOCK_COL_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_ROW_SIZE": 128, "BLOCK_COL_SIZE": 256}, num_warps=8),
        triton.Config({"BLOCK_ROW_SIZE": 256, "BLOCK_COL_SIZE": 512}, num_warps=8),
        triton.Config({"BLOCK_ROW_SIZE": 512, "BLOCK_COL_SIZE": 512}, num_warps=8),
    ],
    key=["TRT_INPUT_C", "TRT_INPUT_H", "TRT_INPUT_W", "TRT_FILTER_HW"],
    reset_to_zero=["output"],
)
@triton.jit
def _gelu_conv2d_forward_triton(
    input,
    filter,
    bias,
    output,
    TRT_INPUT_N: tl.constexpr,
    TRT_INPUT_C: tl.constexpr,
    TRT_INPUT_H: tl.constexpr,
    TRT_INPUT_W: tl.constexpr,
    TRT_FILTER_IN_C: tl.constexpr,
    TRT_FILTER_OUT_C: tl.constexpr,
    TRT_FILTER_HW: tl.constexpr,
    STRIDE_INPUT_N: tl.constexpr,
    STRIDE_INPUT_C: tl.constexpr,
    STRIDE_INPUT_H: tl.constexpr,
    STRIDE_INPUT_W: tl.constexpr,
    STRIDE_FILTER_X: tl.constexpr,
    STRIDE_FILTER_Y: tl.constexpr,
    STRIDE_FILTER_IN_C: tl.constexpr,
    STRIDE_FILTER_OUT_C: tl.constexpr,
    STRIDE_BIAS: tl.constexpr,
    PADDING_H: tl.constexpr,
    PADDING_W: tl.constexpr,
    DILATION_H: tl.constexpr,
    DILATION_W: tl.constexpr,
    GROUPS: tl.constexpr,
    BLOCK_ROW_SIZE: tl.constexpr,
    BLOCK_COL_SIZE: tl.constexpr,
    BIAS: tl.constexpr,
    APPROXIMATE: tl.constexpr,
):
    # Get program ids for x and y axes
    pid_x = tl.program_id(axis=0)
    pid_y = tl.program_id(axis=1)
    # Calculate row and column offsets
    row_offs = pid_x * BLOCK_ROW_SIZE + tl.arange(0, BLOCK_ROW_SIZE)
    col_offs = pid_y * BLOCK_COL_SIZE + tl.arange(0, BLOCK_COL_SIZE)
    # Create masks for rows and columns
    row_mask = row_offs[:, None] < TRT_INPUT_H
    col_mask = col_offs[None, :] < TRT_INPUT_W

    # Compute positions and masks for input and weights
    input_pos = (
        (row_offs[:, None] * DILATION_H + PADDING_H) * TRT_INPUT_W * TRT_INPUT_C
        + (col_offs[None, :] * DILATION_W + PADDING_W) * TRT_INPUT_C
    )

    row_weights_off = tl.arange(0, BLOCK_ROW_SIZE)
    row_weight_mask = row_weights_off[:, None] < TRT_FILTER_HW
    col_weights_off = tl.arange(0, BLOCK_COL_SIZE)
    col_weight_mask = col_weights_off[None, :] < TRT_FILTER_HW

    # Load input patches
    patch_input = tl.load(
        input + (pid_y * BLOCK_COL_SIZE + col_offs) * STRIDE_INPUT_W
        + (pid_x * BLOCK_ROW_SIZE + row_offs) * STRIDE_INPUT_H,
        mask=row_mask & col_mask,
        other=0.0,
    ).to(tl.float32)

    # Slide over the remaining dimensions
    for _ in range(TRT_FILTER_HW):
        w_h_off = ((pid_y * BLOCK_COL_SIZE + col_offs) * DILATION_W + PADDING_W) * (
            TRT_INPUT_C * TRT_FILTER_HW
        ) + (
            (pid_x * BLOCK_ROW_SIZE + row_offs) * DILATION_H * TRT_FILTER_HW
            + (pid_y * BLOCK_COL_SIZE + col_offs) * TRT_FILTER_HW
            + tl.arange(0, TRT_FILTER_HW)
        )

        weight = tl.load(
            filter + w_h_off,
            mask=row_weight_mask & col_weight_mask,
        ).to(tl.float32)

        conv_out = tl.sum(weight * patch_input, axis=1)[None, :]
        patch_input = tl.load(
            input
            + conv_out
            + (pid_y * BLOCK_COL_SIZE + col_offs) * STRIDE_INPUT_W
            + (pid_x * BLOCK_ROW_SIZE + row_offs) * STRIDE_INPUT_H,
            mask=(row_mask & col_mask) & (w_h_off < TRT_FILTER_HW),
            other=0.0,
        ).to(tl.float32)

    # Finalize the convolution and add bias if present
    final_conv = (
        tl.sum(
            tl.load(
                filter
                + (pid_y * BLOCK_COL_SIZE + col_offs)
                * DILATION_W
                * TRT_FILTER_HW
                + (pid_x * BLOCK_ROW_SIZE + row_offs)
                * DILATION_H
                * TRT_FILTER_HW,
                mask=row_mask & col_mask,
            ).to(tl.float32)
            * patch_input,
            axis=1,
        )[None, :]
        + bias
    )

    tanh_out = tl.tanh(
        0.79788456 * final_conv * (1 + 0.044715 * final_conv * final_conv))
    gelu_out = 0.5 * final_conv * (1 + tanh_out)

    # Store the output
    tl.store(
        output
        + (pid_y * BLOCK_COL_SIZE + col_offs) * STRIDE_INPUT_W
        + (pid_x * BLOCK_ROW_SIZE + row_offs) * STRIDE_INPUT_H,
        gelu_out,
        mask=row_mask & col_mask,
    )


def gelu_conv2d_forward_triton(
    input, weight, bias=None, stride=1, padding=0, dilation=1, groups=1, approximate
