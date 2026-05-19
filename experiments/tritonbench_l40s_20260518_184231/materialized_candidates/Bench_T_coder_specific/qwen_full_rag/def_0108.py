import torch
import triton
import triton.language as tl
from torch import Tensor
from torchvision.ops import grid_sample, affine_grid


@triton.jit
def grid_sample_with_affine_kernel(
        input_pointer: tl.tensor,
        theta_pointer: tl.tensor,
        output_pointer: tl.tensor,
        N: tl.constexpr,
        C: tl.constexpr,
        H_IN: tl.constexpr,
        W_IN: tl.constexpr,
        H_OUT: tl.constexpr,
        W_OUT: tl.constexpr,
        INTERPOLATION_MODE: tl.constexpr,
        PADDING_MODE: tl.constexpr,
        ALIGN_CORNERS: tl.constexpr,
        BLOCK_SIZE: tl.constexpr):
    # Get program ids
    program_id_n = tl.program_id(axis=0)
    program_id_h = tl.program_id(axis=1)
    program_id_w = tl.program_id(axis=2)

    # Calculate start indices
    start_index_n = program_id_n
    start_index_h = program_id_h * BLOCK_SIZE
    start_index_w = program_id_w * BLOCK_SIZE

    # Calculate offsets
    offsets_n = start_index_n + tl.arange(0, 1)
    offsets_h = start_index_h + tl.arange(0, BLOCK_SIZE)
    offsets_w = start_index_w + tl.arange(0, BLOCK_SIZE)

    # Create masks
    n_mask = offsets_n < N
    h_mask = offsets_h < H_OUT
    w_mask = offsets_w < W_OUT

    # Load theta values
    offset_theta = offsets_n[:, None] * 6 + \
                   offsets_h[None, :] * 2 + \
                   tl.arange(0, BLOCK_SIZE)[None, :]

    theta_00 = tl.load(theta_pointer + offset_theta,
                       mask=n_mask[:, None] & h_mask[None, :])
    theta_01 = tl.load(theta_pointer + offset_theta + 1,
                       mask=n_mask[:, None] & h_mask[None, :])
    theta_02 = tl.load(theta_pointer + offset_theta + 2,
                       mask=n_mask[:, None] & h_mask[None, :])
    theta_10 = tl.load(theta_pointer + offset_theta + 3,
                       mask=n_mask[:, None] & h_mask[None, :])
    theta_11 = tl.load(theta_pointer + offset_theta + 4,
                       mask=n_mask[:, None] & h_mask[None, :])
    theta_12 = tl.load(theta_pointer + offset_theta + 5,
                       mask=n_mask[:, None] & h_mask[None, :])

    # Unpack offsets
    i = (offsets_h[:, None]).to(tl.float32)
    j = (offsets_w[None, :]).to(tl.float32)
    ones = tl.full((BLOCK_SIZE, BLOCK_SIZE), 1, dtype=tl.float32)

    # Apply affine transform
    if ALIGN_CORNERS:
        x = (j) * (W_IN - 1) / (W_OUT - 1)
        y = (i) * (H_IN - 1) / (H_OUT - 1)
    else:
        x = ((j + 0.5)) * (W_IN - 1) / (W_OUT - 1) - 0.5
        y = ((i + 0.5)) * (H_IN - 1) / (H_OUT - 1) - 0.5

    dx = theta_00 * x + theta_01 * y + theta_02
    dy = theta_10 * x + theta_11 * y + theta_12

    w_mask &= (dx >= 0) & (dy >= 0)
    e_mask = w_mask & ((dx < W_IN - 1) & (dy < H_IN - 1))
    f_mask = w_mask & ((dx < W_IN) & (dy < H_IN))

    # Sampling
    offset_input_float = (offsets_n[:, None]) * H_IN * W_IN + \
                         (dy[None, :].to(tl.int64)) * W_IN + \
                         (dx[None, :].to(tl.int64))

    offset_input_round = (offsets_n[:, None]) * H_IN * W_IN + \
                         ((dy[None, :]).to(tl.int64) * W_IN +
                          (dx[None, :]).to(tl.int64))

    if PADDING_MODE == "zeros":
        input_slice = tl.zeros([1, 1], dtype=input_pointer.dtype.element_ty)
    else:
        input_slice = tl.load(
            input_pointer + offset_input_float,
            mask=e_mask.to(tl.int1),
            other=((PADDING_MODE == "border") * (
                    tl.load(
                        input_pointer + offset_input_round,
                        mask=f_mask.to(tl.int1),
                        other=(PADDING_MODE == "reflection") * (
                            tl.load(
                                input_pointer +
                                offset_input_round % (H_IN * W_IN -
                                                     W_IN) + W_IN,
                                mask=f_mask.to(tl.int1),
                                other=tl.load(
                                    input_pointer +
                                    offset_input_round % (H_IN * W_IN) -
                                    W_IN,
                                    mask=f_mask.to(tl.int1),
                                    other=tl.load(
                                        input_pointer +
                                        (offset_input_round %
                                         (H_IN * W_IN)) % W_IN +
                                        (H_IN - 1) * W_IN,
                                        mask=f_mask.to(tl.int1),
                                        other=tl.load(
                                            input_pointer +
                                            (offset_input_round %
                                             (H_IN * W_IN)) % W_IN,
                                            mask=f_mask.to(tl.int1),
                                            other=tl.load(
                                                input_pointer +
                                                (offset_input_round %
                                                 (H_IN * W_IN)) % W_IN +
                                                (H_IN - 1) * W_IN,
                                                mask=f_mask.to(tl.int1)),
                                        ),
                                    ),
                                ),
                            )
                        )))).to(input_pointer.dtype.element_ty)

    # Interpolate
    if INTERPOLATION_MODE == "linear":
        input_slice_up = tl.load(
            input_pointer + offset_input_float + W_IN,
            mask=f_mask.to(tl.int1),
            other=((PADDING_MODE == "border") * (
                    tl.load(
                        input_pointer + offset_input_round + W_IN,
                        mask=f_mask.to(tl.int1),
                        other=(PADDING_MODE == "reflection") * (
                            tl.load(
                                input_pointer +
                                offset_input_round % (H_IN * W_IN -
                                                     W_IN) + 2 * W_IN,
                                mask=f_mask.to(tl.int64),
                                other=tl.load(
                                    input_pointer +
                                    offset_input_round % (H_IN * W_IN) -
                                    W_IN + 2 * W_IN,
                                    mask=f_mask.to(tl.int64),
                                    other=tl.load(
                                        input_pointer +
                                        (offset_input_round %
                                         (H_IN * W_IN)) % W_IN +
                                        (H_IN - 1) * W_IN + 2 * W_IN,
                                        mask=f_mask.to(tl.int64),
                                        other=tl.load(
                                            input_pointer +
                                            (offset_input_round %
                                             (H_IN * W_IN)) % W_IN + 2 * W_IN,
                                            mask=f_mask.to(tl.int64)),
                                        ),
                                    ),
                                ),
                            )
                        )))).to(input_pointer.dtype.element_ty)
        input_slice_down = tl.load(
            input_pointer + offset_input_float - W_IN,
            mask=f_mask.to(tl.int1),
            other=((PADDING_MODE == "border") * (
                    tl.load(
                        input_pointer + offset_input_round - W_IN,
                        mask=f_mask.to(tl.int1),
                        other=(PADDING_MODE == "reflection") * (
                            tl.load(
                                input_pointer +
                                offset_input_round % (H_IN * W_IN -
                                                     W_IN) - W_IN,
                                mask=f_mask.to(tl.int64),
                                other=tl.load(
                                    input_pointer +
                                    offset_input_round % (H_IN * W_IN) -
                                    2 * W_IN,
                                    mask=f_mask.to(tl.int64),
                                    other=tl.load(
                                        input_pointer +
                                        (offset_input_round %
                                         (H_IN * W_IN)) % W_IN -
                                        W_IN,
                                        mask=f_mask.to(tl.int64),
                                        other=tl.load(
                                            input_pointer +
                                            (offset_input_round %
                                             (H_IN * W_IN)) % W_IN -
                                            W_IN,
                                            mask=f_mask.to(tl.int64)),
                                        ),
                                    ),
                                ),
                            )
                        )))).to(input_pointer.dtype.element_ty)
        output = (1 - (dy[:, None] - tl.floor(dy[:, None])) *
                  (1 / H_IN)) * input_slice + \
                 (dy[:, None] - tl.floor(dy[:, None])) * (1 / H_IN) * input_slice_up - \
                 (1 - (dx[None, :] - tl.floor(dx[None, :])) *
                  (1 / W_IN)) * input_slice - \
                 (dx[None, :] - tl.floor(dx[None, :])) * (1 / W_IN) * input_slice_left - \
                 (1 - (dy[:, None] - tl.floor(dy[:, None])) *
                  (1 / H_IN)) * input_slice_down + \
                 (dy[:, None] - tl.floor(dy[:, None])) * (1 / H_IN) * input_slice_down_right
    elif INTERPOLATION_MODE == "bicubic":
        input_slice_2r = tl.load(
