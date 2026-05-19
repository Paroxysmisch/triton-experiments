import torch
import triton
import triton.language as tl
from triton.ops.conv2d import (
    _output_shape,
    _parse_padding,
    _im2col_dilated_stride,
    _get_config,
    _to_triton_dtype,
)


@triton.jit
def conv2d_relu_fwd(
    y_ptr, x_ptr, w_ptr, b_ptr, y_pad_ptr, x_pad_ptr, M, N, K, ngroups, rK, rN, s_h,
    s_w, p_h, p_w, d_h, d_w, stride_am, stride_ak, x_offs, y_offs, PMOD: tl.constexpr,
    NMOD: tl.constexpr, KMOD: tl.constexpr, G equal 1, SPLIT_K: tl.constexpr,
    SPLIT_N: tl.constexpr,
):
    """Kernel for computing Out = activation(Conv2d(Input, Weight))
    Input has shape (batch, in_height, in_width, in_channels)
    Weight has shape (num_filters, filter_height, filter_width, in_channels)
    Output has shape (batch, out_height, out_width, out_channels)
    """
    # partial row indices
    pid = tl.program_id(axis=0)
    pid_sk = tl.program_id(axis=1)
    i_m = pid // (N // SPLIT_N)
    i_n = pid % (N // SPLIT_N)
    i_k = pid_sk
    num_pid_n = tl.cdiv(N, SPLIT_N)
    num_pid_k = K // tl.num_programs(axis=1)
    offs_k = i_k * tl.num_programs(axis=2) + tl.arange(0, tl.num_programs(axis=2))

    # do matrix mul
    rc_block_start_x = i_n * (rN // SPLIT_N)
    offs_n = i_n * (rN // SPLIT_N) + tl.arange(0, rN // SPLIT_N)
    offs_k = i_k * tl.num_programs(axis=2) + tl.arange(0, tl.num_programs(axis=2))

    accumulator = tl.zeros((SPLIT_N, tl.num_programs(axis=2)), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, tl.num_programs(axis=1) * tl.num_programs(axis=2))):
        selected_weight = tl.load(w_ptr + (offs_k[:, None] * stride_ak +
                                           (k * tl.num_programs(axis=1) *
                                            tl.num_programs(axis=2) + offs_n[None, :])),
                                  mask=(offs_k[:, None] < K - k * tl.num_programs(
                                      axis=1) * tl.num_programs(axis=2)) &
                                  (offs_n[None, :] < N), other=0.0)

        # Fetching blocks of the input, currently set to fetch 128 rows at a time.
        # Note that cross-row-padding is achieved via masking (see above).
        selected_input = tl.load(x_pad_ptr + (i_m * stride_am +
                                              offs_n[None, :] * d_h * s_h +
                                              offs_k[:, None] * stride_ak),
                                 mask=(offs_n[None, :] * d_h * s_h <
                                       M * s_h) & (offs_k[:, None] < K -
                                                 k * tl.num_programs(
                                                     axis=1) * tl.num_programs(
                                                         axis=2)),
                                 other=0.0)

        if b_ptr is not None:
            broadcasted_bias = tl.load(b_ptr + offs_k,
                                       mask=offs_k < K - k * tl.num_programs(
                                           axis=1) * tl.num_programs(axis=2),
                                       other=0.0)
            accumulator += broadcasted_bias[:, None]

        offs_rn = ((rc_block_start_x + offs_n) % rN) * N
        rc_selected_input = tl.reshape(selected_input,
                                       (tl.num_programs(axis=0), -1)).to(
                                           selected_weight.dtype)
        accumulator += tl.dot(rc_selected_input, selected_weight)

        rc_block_start_x += rN // SPLIT_N

    accumulator = accumulator.to(y_ptr.dtype.element_ty)

    # apply activation
    accumulator = tl.maximum(accumulator, 0)

    # rematerialize masks to save registers
    m = (offs_n[None, :] * d_h * s_h < M * s_h) & (offs_k[:, None] < K -
                                                   k * tl.num_programs(
                                                       axis=1) * tl.num_programs(
                                                           axis=2))
    n = (offs_n < N)
    # write back result
    if PMOD == "none":
        tl.store(y_ptr + (i_m * N + offs_n.to(tl.int32)), accumulator, mask=n)
    elif PMOD == "row":
        y_ptrs = y_ptr + (pid * N + offs_n.to(tl.int32))
        tl.atomic_add(y_ptrs, accumulator, mask=m)
    else:
        assert PMOD == "col"
        y_ptrs = y_ptr + (offs_n.to(tl.int32) * M + i_m)
        tl.atomic_add(y_ptrs, accumulator, mask=m)


def relu_conv2d(
    input: torch.Tensor, weight: torch.Tensor, bias: Optional[
        torch.Tensor] = None, stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0, dilation: Union[int, Tuple[
        int, int]] = 1, groups: int = 1, inplace: bool = False
) -> torch.Tensor:
    stride_am, stride_ak = weight.stride(-2)
    in_batch, in_height, in_width, in_channels = input.shape
    f_count, f_height, f_width, f_in_channels = weight.shape
    out_count, out_height, out_width, out_channels = (
        in_batch,
        in_height//f_height,
        in_width//f_width,
        f_count,
    )

    assert f_in_channels == in_channels, "AtrousConv2d only supports in_channels == in_group_channels"
    assert in_channels % groups == 0
    assert f_count % groups == 0

    # handle default stride and padding
    if isinstance(stride, int):
        stride = (stride, stride)
    if isinstance(dilation, int):
        dilation = (dilation, dilation)
    if isinstance(padding, str):
        padding = padding.upper()
        if padding == "VALID":
            padding = (0, 0)
        elif padding == "SAME":
            required_out_size = (in_height - 1)//stride[0] + 1
            padding = max((f_height - 1)*dilation[0] + 1 - stride[0],
                          0) + max((f_width - 1)*dilation[1] + 1 - stride[1],
                                   0), 2)
            padding = (padding // 2, padding - padding // 2)
        else:
            raise ValueError("Invalid padding string " + padding)
    else:
        if len(padding) == 2:
            padding = list(padding)
            padding.insert(0, 0)
            padding.append(0)
        assert len(padding) == 4, "Invalid padding dimensions" + str(padding)

    out = torch.empty(out_count, out_height, out_width, out_channels,
                      device=input.device, dtype=input.dtype)

    # early return for empty conv
    if f_height == 0 or f_width == 0:
        return out

    # compute padding value
    p_h = (f_height - 1) * dilation[0]
    p_w = (f_width - 1) * dilation[1]

    # run kernel
    grid = lambda META: (META["SPLIT_M"], META["SPLIT_K"],
                         triton.cdiv(f_count, META["SPLIT_K"] * META["SPLIT_N"]))
    _, rN, rK = _im2col_dilated_stride(
        in_height, in_width, f_height, f_width, d_h=dilation[0], d_w=dilation[1],
        pad_h=padding[0], pad_w=padding[2], stride_h=stride[0], stride_w=stride[1])
    with torch.cuda.device(input.device.index):
        conv2d_relu_fwd[grid](out, input, weight, bias, out, _pad(input, [padding[0], padding[2]], [0, 0]),
                              out_count, out_height, out_width, in_channels, groups, rK, rN, stride[0], stride[1],
                              p_h, p_w, dilation[0], dilation[1], stride_am, stride_ak, out_height //
                              META["SPLIT_M"], out, PMOD="none", NMOD=rN, KMOD=rK, G equal 1, SPLIT_K=triton.cdiv(
                                  f_count, META["SPLIT_N"] * META["SPLIT_K"]), SPLIT_N=rN)

    return out
