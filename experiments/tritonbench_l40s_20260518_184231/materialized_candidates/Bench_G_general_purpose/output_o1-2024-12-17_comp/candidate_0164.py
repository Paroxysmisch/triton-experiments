import torch
import triton
import triton.language as tl

# --------------------------------------------------------------------------------
# Activation functions in Triton

@triton.jit
def _relu(x):
    return tl.where(x > 0, x, 0.0)

@triton.jit
def _tanh(x):
    # Approximate tanh using elementwise operations
    # For numerical stability, clamp input to [-10, 10]
    x = tl.max(tl.min(x, 10.0), -10.0)
    e2x = tl.exp(2 * x)
    return (e2x - 1) / (e2x + 1)

@triton.jit
def _gelu(x):
    return 0.5 * x * (1.0 + tl.erf(x / 1.41421356237))

@triton.jit
def _fast_gelu(x):
    return 0.5 * x * (1.0 + tl.tanh(0.7978845608 * x + 0.0356774081 * x * x * x))

@triton.jit
def _apply_activation(x, act_id):
    # 0: None, 1: ReLU, 2: Tanh, 3: GELU, 4: FastGELU
    if act_id == 1:
        x = _relu(x)
    if act_id == 2:
        x = _tanh(x)
    if act_id == 3:
        x = _gelu(x)
    if act_id == 4:
        x = _fast_gelu(x)
    return x

# --------------------------------------------------------------------------------
# Matrix multiplication kernel with possible bias and activation

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=4),
        triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "BLOCK_K": 32}, num_warps=8),
    ],
    key=["M", "N", "K"]
)
@triton.jit
def kernel_fma(
    A_ptr, B_ptr, C_ptr,
    Bias_ptr, act_id,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    mask_m = offs_m < M
    mask_n = offs_n < N

    # Create a pointer for storing partial sums
    accum = tl.zeros([BLOCK_M, BLOCK_N], dtype=tl.float32)

    # k-loop
    # We move in steps of BLOCK_K
    for k_offs in range(0, K, BLOCK_K):
        offs_k = tl.arange(0, BLOCK_K)
        a_ptrs = A_ptr + (offs_m[:, None] * stride_am + (k_offs + offs_k[None, :]) * stride_ak)
        b_ptrs = B_ptr + ((k_offs + offs_k[:, None]) * stride_bk + offs_n[None, :] * stride_bn)
        a_mask = (mask_m[:, None]) & (k_offs + offs_k[None, :] < K)
        b_mask = (k_offs + offs_k[:, None] < K) & (mask_n[None, :])
        a_vals = tl.load(a_ptrs, mask=a_mask, other=0.0)
        b_vals = tl.load(b_ptrs, mask=b_mask, other=0.0)
        accum += tl.dot(a_vals, b_vals)

    # Optionally add bias
    if Bias_ptr != 0:
        bias_vals = tl.load(Bias_ptr + offs_n * stride_bias, mask=mask_n, other=0.0)
        accum += bias_vals[None, :]

    # Apply activation
    accum = _apply_activation(accum, act_id)

    # Write back
    c_ptrs = C_ptr + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accum, mask=mask_m[:, None] & mask_n[None, :])

# --------------------------------------------------------------------------------
# PyTorch autograd Function

class LinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, A, B, bias, activation):
        M, K = A.shape
        Kb, N = B.shape
        assert K == Kb, "Incompatible dimensions for matrix multiplication"

        out = torch.empty((M, N), device=A.device, dtype=A.dtype)
        bias_ptr = bias.data_ptr() if bias is not None else 0
        stride_bias = 1 if bias is not None else 0

        act_id = 0
        if activation == 'relu':
            act_id = 1
        elif activation == 'tanh':
            act_id = 2
        elif activation == 'gelu':
            act_id = 3
        elif activation == 'fastgelu':
            act_id = 4

        grid = lambda META: (triton.cdiv(M, META["BLOCK_M"]), triton.cdiv(N, META["BLOCK_N"]))

        kernel_fma[grid](
            A, B, out,
            bias_ptr, act_id,
            M, N, K,
            A.stride(0), A.stride(1),
            B.stride(0), B.stride(1),
            out.stride(0), out.stride(1),
            stride_bias,
        )
        return out

# --------------------------------------------------------------------------------
# User-facing function

def linear_layer(A, B, bias=None, activation=None):
    return LinearLayer.apply(A, B, bias, activation)
