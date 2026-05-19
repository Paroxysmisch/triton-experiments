import torch
import triton
import triton.language as tl

# ------------------------------------------------------------------------------
# Optional activation functions in Triton
# ------------------------------------------------------------------------------
@triton.jit
def tanh_activation(x):
    # Approximation of tanh
    # tanh(x) = (e^(2x) - 1) / (e^(2x) + 1)
    exp2x = tl.exp(x * 2.0)
    return (exp2x - 1.0) / (exp2x + 1.0)

@triton.jit
def relu_activation(x):
    return tl.where(x >= 0, x, 0)

@triton.jit
def gelu_activation(x):
    # Approximate GELU
    # 0.5 * x * (1.0 + erf(x / sqrt(2)))
    # Here we use an approximation with tl.erf
    return 0.5 * x * (1.0 + tl.erf(x * 0.7071067811865475))

@triton.jit
def fast_gelu_activation(x):
    # FastGelu approximation
    # 0.5 * x * (1.0 + tanh(0.7978845608 * (x + 0.044715 * x^3)))
    return 0.5 * x * (1.0 + tl.tanh(0.7978845608 * (x + 0.044715 * x * x * x)))

# ------------------------------------------------------------------------------
# Dispatch function to select activation. We treat this as a compile-time choice.
# ------------------------------------------------------------------------------
@triton.jit
def apply_activation(x, activation_id: tl.constexpr):
    # ID mapping:
    # 0 -> no activation
    # 1 -> tanh
    # 2 -> relu
    # 3 -> gelu
    # 4 -> fast_gelu
    if activation_id == 1:
        return tanh_activation(x)
    if activation_id == 2:
        return relu_activation(x)
    if activation_id == 3:
        return gelu_activation(x)
    if activation_id == 4:
        return fast_gelu_activation(x)
    return x

# ------------------------------------------------------------------------------
# Triton kernel: Matrix multiplication + bias + optional activation
# C = A * B + bias (row-wise) --> optionally apply activation
# ------------------------------------------------------------------------------
@triton.autotune(
    configs=[
        triton.Config({}, num_warps=4, num_stages=2),
        triton.Config({}, num_warps=8, num_stages=3),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def kernel_fma(
    A, B, C, Bias, Out,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bias,
    activation_id: tl.constexpr,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr
):
    pid = tl.program_id(0)
    
    # Compute the start indices for this program ID
    num_pid_m = (M + BLOCK_M - 1) // BLOCK_M
    pid_m = pid // ((N + BLOCK_N - 1) // BLOCK_N)
    pid_n = pid % ((N + BLOCK_N - 1) // BLOCK_N)

    start_m = pid_m * BLOCK_M
    start_n = pid_n * BLOCK_N

    # Create range for sub-block
    rm = start_m + tl.arange(0, BLOCK_M)
    rn = start_n + tl.arange(0, BLOCK_N)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # Pointer arithmetic for A and B
    a_ptrs = A + (rm[:, None] * stride_am + tl.arange(0, BLOCK_K)[None, :] * stride_ak)
    b_ptrs = B + (tl.arange(0, BLOCK_K)[:, None] * stride_bk + rn[None, :] * stride_bn)

    # Accumulate over K in steps of BLOCK_K
    for k_block_id in range(0, K, BLOCK_K):
        # Load A and B
        a = tl.where((rm[:, None] < M) & (k_block_id + tl.arange(0, BLOCK_K)[None, :] < K),
                     tl.load(a_ptrs), 0.0)
        b = tl.where((k_block_id + tl.arange(0, BLOCK_K)[:, None] < K) & (rn[None, :] < N),
                     tl.load(b_ptrs), 0.0)
        # Matmul accumulate
        acc += tl.dot(a, b)
        # Advance pointers
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    # Add bias if provided (row-wise bias)
    if Bias != tl.zeros(1, dtype=tl.int1):
        bias_vals = tl.load(Bias + rm * stride_bias, mask=(rm < M), other=0.0)
        acc = acc + bias_vals[:, None]

    # Apply optional activation
    acc = apply_activation(acc, activation_id)

    # Write back to Out (in fp16 or bf16)
    c_ptrs = C + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    out_ptrs = Out + (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask = (rm[:, None] < M) & (rn[None, :] < N)

    # Store pre-activation results in C if the user wants to keep them
    tl.store(c_ptrs, acc.to(tl.float16), mask=mask)

    # Copy to Out (same as C here, unless partial usage is desired)
    tl.store(out_ptrs, acc.to(tl.float16), mask=mask)

# ------------------------------------------------------------------------------
# PyTorch Custom Function
# ------------------------------------------------------------------------------
class LinearLayer(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inp, weight, bias=None, activation_id=0, store_pre_activation=False):
        """
        inp:       [M, K]
        weight:    [K, N]
        bias:      [N] or None
        activation_id: 0=no activation, 1=tanh, 2=relu, 3=gelu, 4=fast_gelu
        store_pre_activation: if True, saves pre-activation for backward
        """
        M, K = inp.shape
        K2, N = weight.shape
        assert K == K2, "Input matrix dimensions must match weight matrix"
        device = inp.device
        
        # Allocate output
        out = torch.empty((M, N), dtype=inp.dtype, device=device)
        # Optionally store pre-activation
        c_store = torch.empty((M, N), dtype=inp.dtype, device=device) if store_pre_activation else torch.zeros((1,), device=device)

        # Bias pointer: if None, pass a dummy pointer
        bias_data = bias if bias is not None else torch.zeros((1,), device=device)

        # Strides
        stride_am = inp.stride(0)
        stride_ak = inp.stride(1)
        stride_bk = weight.stride(0)
        stride_bn = weight.stride(1)
        stride_cm = out.stride(0)
        stride_cn = out.stride(1)
        stride_bias = bias_data.stride(0)

        grid = lambda META: (
            (M + META['BLOCK_M'] - 1) // META['BLOCK_M']
            * (N + META['BLOCK_N'] - 1) // META['BLOCK_N'],
        )

        kernel_fma[grid](
            inp, weight, c_store, bias_data, out,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            stride_bias,
            activation_id,
            BLOCK_M=128, BLOCK_N=64, BLOCK_K=32
        )

        ctx.save_for_backward(inp, weight, bias, out if store_pre_activation else None)
        ctx.activation_id = activation_id
        ctx.store_pre_activation = store_pre_activation
        return out

    @staticmethod
    def backward(ctx, grad_output):
        # This example does not implement backward pass for brevity.
        # You could implement the backward logic here by re-running the forward
        # with appropriate partial derivatives or using separate Triton kernels.
        return None, None, None, None, None

# ------------------------------------------------------------------------------
# User-facing wrapper
# ------------------------------------------------------------------------------
def linear_layer(inp, weight, bias=None, activation=None, store_pre_activation=False):
    """
    inp: 2D Tensor [M, K], weight: 2D Tensor [K, N], bias: 1D Tensor [N], optional
    activation: One of 'none', 'tanh', 'relu', 'gelu', 'fast_gelu'
    store_pre_activation: If True, pre-activation is stored (for potential backward usage).
    """
    activations_map = {
        None: 0,
        'none': 0,
        'tanh': 1,
        '
