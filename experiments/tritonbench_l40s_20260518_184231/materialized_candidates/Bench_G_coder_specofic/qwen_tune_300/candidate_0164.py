import torch
import triton
import triton.language as tl
from torch import Tensor
from torch.autograd.function import Function

_kAlpha = math.sqrt(2.0 / math.pi)

@triton.heuristics(
    {
        "EVEN_M": lambda args: args["M"] % args["BLOCK_M"] == 0,
        "EVEN_N": lambda args: args["N"] % args["BLOCK_N"] == 0,
        "EVEN_K": lambda args: args["K"] % args["BLOCK_K"] == 0,
        "IS_TANH": lambda args: args["ACTIVATION"] == "tanh",
        "IS_RELU": lambda args: args["ACTIVATION"] == "relu",
        "IS_GELU": lambda args: args["ACTIVATION"] == "gelu",
        "IS_FAST_GELU": lambda args: args["ACTIVATION"] == "fast_gelu",
    }
)
@triton.jit
def kernel_fma(
    A,  # shape: (M, K)
    B,  # shape: (K, N)
    C,  # shape: (M, N)
    bias,  # shape: (N,)
    activation,  # Optional[Tensor]
    # stride
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    # meta
    M: tl.constexpr,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr = 32,
    GROUP_M: tl.constexpr = 8,
    SAVE_ACT_INPUT: tl.constexpr = False,
    ACTIVATION: tl.constexpr = None,
    EVEN_M: tl.constexpr = False,
    EVEN_N: tl.constexpr = False,
    EVEN_K: tl.constexpr = False,
    IS_TANH: tl.constexpr = False,
    IS_RELU: tl.constexpr = False,
    IS_GELU: tl.constexpr = False,
    IS_FAST_GELU: tl.constexpr = False,
):
    # split k to reduce shared memory usage
    # for large N, we can also split N
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    if pid_m * BLOCK_M >= M or pid_n * BLOCK_N >= N:
        return

    offs_am = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_bn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (
        offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak
    )  # (BLOCK_M, BLOCK_K)
    b_ptrs = B + (
        offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn
    )  # (BLOCK_K, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        if EVEN_K:
            a = tl.load(a_ptrs)
            b = tl.load(b_ptrs)
        else:
            a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
            b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    if SAVE_ACT_INPUT:
        act_in = tl.where(c_mask, accumulator, 0.0)
        tl.store(c_ptrs, act_in)

    if bias is not None:
        bias_ptrs = bias + offs_cn
        if not EVEN_N:
            bias = tl.load(bias_ptrs, mask=offs_cn < N - pid_n * BLOCK_N, other=0.0)
        else:
            bias = tl.load(bias_ptrs)
        accumulator += bias[None, :]

    if ACTIVATION:
        if IS_RELU:
            accumulator = tl.where(accumulator >= 0, accumulator, 0.0)
        elif IS_TANH:
            accumulator = 2 * tl.sigmoid(2 * accumulator) - 1
        elif IS_GELU:
            # tanh(x) = 2 * sigmoid(2x) - 1
            tanh_2ax = 2 * tl.sigmoid(2 * accumulator) - 1
            # gelu(x) = 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3))
            accumulator = 0.5 * accumulator * (1 + tanh_2ax * (1 + 0.044715 * accumulator * accumulator))
        elif IS_FAST_GELU:
            accumulator = accumulator * 0.5 * (1 + tanh(_kAlpha * (accumulator + 0.044715 * accumulator * accumulator * accumulator))
    c = accumulator.to(tl.float16)

    tl.store(c_ptrs, c, mask=c_mask)


class LinearLayer(Function):
    @staticmethod
    def forward(
        ctx,
        x: Tensor,
        weight: Tensor,
        bias: Tensor,
        activation: str = None,
        save_act_input: bool = False,
    ):
        assert weight.shape[1] == x.shape[-1]
        assert weight.shape[0] == bias.shape[0]
        M, K = weight.shape
        N = bias.shape[-1]
        assert x.shape[-1] == K
        x_ = x.reshape(-1, M, K)
        o = torch.empty((x_.shape[0], M, N), dtype=x.dtype, device=x.device)
        bias_ = bias.reshape(1, N)
        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),)
        kernel_fma[grid](
            x_,
            weight,
            o,
            bias_,
            None if activation is None else torch.empty(1, dtype=x.dtype, device=x.device),
            x_.stride(0),
            x_.stride(1),
            weight.stride(0),
            weight.stride(1),
            o.stride(1),
            o.stride(2),
            M=M,
            N=N,
            K=K,
            ACTIVATION=activation,
            SAVE_ACT_INPUT=save_act_input,
        )
        o = o.reshape_as(x)
        ctx.save_for_backward(x, weight, bias)
        ctx.activation = activation
        return o


def linear_layer(
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    activation: str = None,
    save_act_input: bool = False,
) -> Tensor:
    assert activation is None or save_act_input
    return LinearLayer.apply(x, weight, bias, activation, save_act_input)
