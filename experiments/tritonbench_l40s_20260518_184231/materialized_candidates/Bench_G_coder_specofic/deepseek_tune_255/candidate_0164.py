import torch
import triton
import triton.language as tl
from torch.autograd.function import FunctionCtx

@triton.jit
def tanh(x):
    return 2 * tl.sigmoid(2 * x) - 1

@triton.jit
def relu(x):
    return tl.where(x >= 0, x, 0.0)

@triton.jit
def gelu(x):
    return x * 0.5 * (1.0 + tl.libdevice.erf(x / 1.41421356237))

@triton.jit
def fast_gelu(x):
    pi = 3.1415926535897932
    a = (0.79788456 * x + 0.03567732 * x * x * x) * (1.0 + 0.044715 * x * x)
    b = x * 0.5 * (1.0 + tanh(a * (1.0 / 4.0) * (1.0 + 0.79788456 * x + 0.0331 * x * x)))
    return b

@triton.jit
def kernel_fma(
    A,
    B,
    C,
    bias,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    MUL_ROUTED: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BIAS: tl.constexpr,
    fp8_w: tl.constexpr = False,
    fp8_a: tl.constexpr = False,
    fp8_act: tl.constexpr = False,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = A + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    if BIAS:
        bias_ptrs = bias + offs_bn
        accumulator += tl.load(bias_ptrs).to(tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_SIZE_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_SIZE_K, other=0.0)
        if fp8_w:
            b = unpack8to16(b)
        if fp8_a:
            a = unpack8to16(a)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    if MUL_ROUTED:
        accumulator = mul_and_add(accumulator)
    if ACTIVATION == "tanh":
        accumulator = tanh(accumulator)
    elif ACTIVATION == "relu":
        accumulator = relu(accumulator)
    elif ACTIVATION == "gelu":
        accumulator = gelu(accumulator)
    elif ACTIVATION == "fast_gelu":
        accumulator = fast_gelu(accumulator)
    if fp8_act:
        accumulator = pack16to8(accumulator, 0)
    c_ptrs = C + stride_cm * offs_am[:, None] + stride_cn * offs_bn[None, :]
    c_mask = (offs_am[:, None] < M) & (offs_bn[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=c_mask)

class LinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx: FunctionCtx,
        A: torch.Tensor,
        B: torch.Tensor,
        bias: Optional[torch.Tensor],
        pre_act: Optional[bool],
        activation: Optional[str],
        save_inputs_in_fwd: Optional[bool],
    ):
        assert A.shape[-1] == B.shape[1], "Incompatible dimensions"
        M, K = A.shape
        _, N = B.shape
        C = torch.empty((M, N), device=A.device, dtype=A.dtype)
        fp8_w = getattr(B, "fp8_w", False)
        fp8_a = getattr(A, "fp8_w", False)
        fp8_act = getattr(B, "fp8_act", False) or getattr(A, "fp8_act", False)
        BLOCK_SIZE_M = 128
        BLOCK_SIZE_N = 128
        BLOCK_SIZE_K = 32
        grid = lambda META: (
            triton.cdiv(M, META["BLOCK_SIZE_M"]) * triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )
        kernel_fma[grid](
            A,
            B,
            C,
            bias,
            M,
            N,
            K,
            A.stride(0),
            A.stride(1),
            B.stride(0),
            B.stride(1),
            C.stride(0),
            C.stride(1),
            BLOCK_SIZE_M,
            BLOCK_SIZE_N,
            BLOCK_SIZE_K,
            GROUP_SIZE_M=4,
            MUL_ROUTED=False,
            ACTIVATION=activation if not pre_act else None,
            BIAS=bias is not None,
            fp8_w=fp8_w,
            fp8_a=fp8_a,
            fp8_act=fp8_act,
        )
        ctx.save_for_backward(A, B, bias)
        ctx.activation = activation
        ctx.pre_act = pre_act
        ctx.fp8_w = fp8_w
        ctx.fp8_a = fp8_a
        ctx.fp8_act = fp
