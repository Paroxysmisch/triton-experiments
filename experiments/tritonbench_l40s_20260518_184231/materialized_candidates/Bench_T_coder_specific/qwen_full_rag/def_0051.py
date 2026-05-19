import torch
import triton
import triton.language as tl
from torch import Tensor
from typing import Optional

@triton.jit
def _cos_and_avg_pool1d_triton(
    inp,
    out,
    ws,
    s_iw,
    s_ow,
    half_ws,
    scale,
    M: tl.constexpr,
    N: tl.constexpr,
    MAX_N: tl.constexpr,
):
    mid = tl.program_id(0) * M + tl.arange(0, M)
    iws = mid[:, None] * s_iw + tl.arange(0, N)[None, :] + half_ws
    mask = iws < MAX_N
    x = tl.load(inp + iws, mask=mask, other=0.0)
    x = tl.cos(x)
    x = tl.sum(x, axis=1) / scale
    ows = mid * s_ow
    tl.store(out + ows, x, mask=M > ows)

def cos_avg_pool1d(
    x: Tensor,
    ws: int,
    stride: Optional[int] = None,
    padding: int = 0,
    ceil_mode: bool = False,
    count_include_pad: bool = True,
) -> Tensor:
    """Applies a avg pool1d after cos to the input.

    Args:
        x (Tensor): the input tensor
        ws (int): the size of the window to take a max over
        stride (Optional[int]): the stride of the window. Default value is ws
        padding (int): implicit zero padding to be added on both sides
        ceil_mode (bool): if true, use ceil instead of floor to compute the output shape
        count_include_pad (bool): if true, include padding while averaging

    Returns:
        Tensor: the output tensor
    """
    if stride is None:
        stride = ws

    if count_include_pad:
        op_ws = ws
    else:
        op_ws = ws - 2 * padding

    m = x.ndim
    x = x.unsqueeze(0) if x.ndim < 3 else x
    x = x.flatten(0, 1)
    n = x.shape[0]
    out = torch.empty(n, dtype=x.dtype, device=x.device)
    start = -padding if padding else 0
    end = n * stride - start + ws
    s_out = (end + stride - 1) // stride
    s_iw = x.stride(0)
    s_ow = out.stride(0)
    half_ws = ws // 2
    MAX_N = triton.next_power_of_2(ws)
    if MAX_N > 32768:
        raise ValueError("ws must be <= 32768")
    NT = triton.cdiv(s_out, 128)
    assert NT > 0
    scale = float(op_ws) if count_include_pad else float(ws)
    _cos_and_avg_pool1d_triton[(NT, 128)](
        x,
        out,
        ws,
        s_iw,
        s_ow,
        half_ws,
        scale,
        128,
        s_out,
        n,
        MAX_N,
    )
    out = out.reshape(-1, *x.shape[1:])
    return out.squeeze(0) if m < 3 else out
