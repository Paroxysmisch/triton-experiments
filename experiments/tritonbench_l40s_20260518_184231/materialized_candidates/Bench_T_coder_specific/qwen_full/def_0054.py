import torch
import triton
import triton.language as tl
from triton.language.libdevice import erf, exp, pow, tanh

@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def gelu_forward(x, approximate="none"):
    if approximate == "none":
        return x * 0.5 * (1.0 + erf(x / 1.41421356237))
    else:
        return 0.5 * x * (1.0 + tanh(x * 0.79788456 * (1.0 + 0.044715 * pow(x.to(tl.float32), 2.0)))))

@triton.jit
def conv2d_gemm(
    input, weight, bias, stride, padding, dilation, groups, approximate
):
    N, C, H, W = input.shape
    K, _, R, C = weight.shape
    output = tl.zeros([N, K, H, W], dtype=tl.float32)

    padding = (0, 0, 0, 0) if isinstance(padding, int) else padding
    padding = ((0, 0), (0, 0)) + tuple(
        p if isinstance(p, (tuple, list)) else (p, p)
        for p in padding
    )

    stride = (1, 1) if isinstance(stride, int) else stride
    stride = (*((1,) * (input.ndim - len(stride)),) + tuple(stride))

    dilation = (1, 1) if isinstance(dilation, int) else dilation
    dilation = (*((1,) * (weight.ndim - len(dilation)),) + tuple(dilation))

    input = tl.nn.pad(
        input, [(0, 0), (0, 0)] + list(padding), mode="constant", value=0
    )

    pid_n = tl.program_id(0)
    pid_k = tl.program_id(1)
    pid_h = tl.program_id(2)
    pid_w = tl.program_id(3)

    # bias
    if bias is not None:
        bias = tl.load(bias + pid_k)

    # k/h/w
    for _ in range(0, N):
        # N/C
        for c in range(0, C, 16):
            # C/R/C
            for rr in range(0, R, 8):
                for cc in range(0, C, 16):
                    # N/C/R/C
                    offset_c = c + cc
                    offset_r = rr + pid_h * 8
                    offset_pid = (
                        (offset_c // 16) * R * C
                        + offset_r * C
                        + (offset_pid_k * groups + pid_k) * C
                        + cc
                    )
                    A = tl.load(
                        input + offset_n * stride[0] * stride[2] * stride[3] +
                        offset_c * stride[1] * stride[2] * stride[3] +
                        (offset_h * stride[2] + offset_r) * stride[3] + cc,
                        mask=(offset_c < C)
                        & (offset_r < H)
                        & (offset_c // 16 == pid_k),
                        other=0,
                    )

                    for _ in range(8 // 16):
                        # K/C/R/C
                        offset_k = pid_k * (C // groups) + (cc // 16)
                        B = tl.load(
                            weight + offset_k * R * C + offset_pid,
                            mask=(offset_k < K)
                            & (offset_r < R)
                            & (offset_c // 16 == pid_k),
                            other=0,
                        )
                        output += tl.dot(A, B, allow_tf32=False)
                    output += bias
    output = gelu_forward(output, approximate)
    return output

def gelu_conv2d(
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    stride: Union[int, Tuple[int, int]] = 1,
    padding: Union[int, Tuple[int, int], str] = 0,
    dilation: Union[int, Tuple[int, int]] = 1,
    groups: int = 1,
    approximate: str = "none",
    out: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    if input.dtype == torch.float16:
        input = input.to(torch.bfloat16)
    out = conv2d_gemm(
        input,
        weight,
        bias,
        stride,
        padding,
        dilation,
        groups,
        approximate,
    )
    return out
