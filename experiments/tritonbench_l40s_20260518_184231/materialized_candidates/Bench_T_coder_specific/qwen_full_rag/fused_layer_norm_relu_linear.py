import torch
import triton
import triton.language as tl
from torch import Tensor
from triton.runtime.jit import get_cuda_stream

@triton.jit
def _layer_norm_relu_linear_fwd_fused(
    C,  # Pointers to matrices
    A,
    B,
    W,
    Z,
    mean,  # Intermediate buffers
    rstd,
    row_scale,
    col_off,
    M,  # Matrix dimensions
    N,
    K,
    s_a_m,
    s_a_n,
    s_c_m,
    s_c_k,
    s_z_m,
    s_z_k,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    ACTIVATION: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = A + (offs_am[:, None] * s_a_m + offs_k[None, :] * s_a_n)
    b_ptrs = B + (offs_k[:, None] * s_a_n + offs_bn[None, :] * s_a_m)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * s_a_n
        b_ptrs += BLOCK_K * s_a_m

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = C + s_c_m * offs_cm[:, None] + s_c_k * offs_k[None, :]
    z_ptrs = Z + s_z_m * offs_cm[:, None] + s_z_k * offs_k[None, :]

    for k in range(0, tl.cdiv(K, BLOCK_K)):
        mask = (offs_k[None, :] < K - k * BLOCK_K) & (
            (offs_cn[None, :] + col_off) < N
        )
        w = tl.load(W + offs_k, mask=offs_k < K - k * BLOCK_K, other=0.0)
        bias = tl.load(B + offs_k[:, None], mask=mask, other=0.0)
        act_offs = s_c_k * offs_k[None, :] + s_c_m * offs_cm[:, None]
        act_ptrs = C + act_offs
        act = tl.load(act_ptrs, mask=mask)
        if ACTIVATION == "relu":
            act = tl.where(act >= 0, act, 0)
        elif ACTIVATION == "gelu":
            l = 0.5 * (1.0 + tl.math.erf(0.7071067811 * act))
            act = act * l
        elif ACTIVATION == "glu":
            h = act[:, 0 : (N // 2)]
            g = act[:, (N // 2) : N]
            act = h * tl.sigmoid(g)
        ln_mean = tl.sum(act, axis=1) / N
        ln_var = tl.sum((act - ln_mean[:, None]) * (act - ln_mean[:, None]), axis=1) / N
        inv_ln_std = 1 / tl.sqrt(ln_var + 1e-5)
        act = (act - ln_mean[:, None]) * inv_ln_std[:, None]
        act = act * (1 + w) + bias
        tl.store(c_ptrs, act, mask=mask)
        z = act
        z = z.to(C.dtype.element_ty)
        tl.store(z_ptrs, z, mask=mask)
        c_ptrs += BLOCK_K * s_c_k
        z_ptrs += BLOCK_K * s_z_k
        offs_k += BLOCK_K


def fused_layer_norm_relu_linear(
    x: Tensor,
    weight: Tensor,
    bias: Tensor = None,
    normalized_shape=None,
    eps=1e-5,
    elementwise_affine=True,
    activation="relu",
):
    ftype = x.dtype
    assert x.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert x.is_contiguous()
    assert weight.is_contiguous()
    if bias is not None:
        assert bias.is_contiguous()
    assert x.shape[-1] == weight.shape[0]
    assert x.ndim >= 2
    if normalized_shape is not None:
        assert isinstance(normalized_shape, int)
        assert normalized_shape <= x.shape[-1]
        shape = x.shape
        x = x.reshape([-1, shape[-1]])
        M, N = x.shape
        K = weight.shape[0]
        col_off = 0
    else:
        assert isinstance(normalized_shape, type(None))
        M, N = x.shape
        K = weight.shape[0]
        col_off = N // 2
    out = torch.empty((M, K), dtype=x.dtype, device=x.device)
    z = torch.empty((M, K), dtype=torch.float32, device=x.device)
    mean = torch.empty((M,), dtype=torch.float32, device=x.device)
    rstd = torch.empty((M,), dtype=torch.float32, device=x.device)
    row_scale = torch.empty((M, K), dtype=torch.float32, device=x.device)
    x_arg = x.view(-1, x.shape[-1])
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_M"]) * triton.cdiv(N, META["BLOCK_N"]),
    )
    _layer_norm_relu_linear_fwd_fused[grid](
        out,
        x_arg,
        weight,
        z,
        mean,
        rstd,
        row_scale,
        col_off,
        M,
        N,
        K,
        x_arg.stride(0),
        x_arg.stride(1),
        out.stride(0),
        out.stride(1),
        z.stride(0),
        z.stride(1),
        ACTIVATION=activation,
    )
    out = out.reshape(shape[:-1] + (K,))
    return out
