import torch
import triton
import triton.language as tl

# Define activation functions in Triton
@triton.jit
def relu(x):
    return tl.where(x >= 0, x, 0.0)

@triton.jit
def gelu(x):
    cdf = 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))
    return x * cdf

@triton.jit
def fast_gelu(x):
    return 0.5 * x * (1.0 + tl.tanh(tl.sqrt(2 / 3.141592653589793) * (x + 0.044715 * x * x * x)))

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32}, num_warps=4),
    ],
    key=['M', 'N', 'K']
)
@triton.jit
def kernel_fma(
    a_ptr, b_ptr, c_ptr, bias_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    stride_bm: tl.constexpr,
    ACTIVATION: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_in_group = GROUP_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = a_ptr + offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak
    b_ptrs = b_ptr + offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_m[:, None] < M) & (offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k) & (offs_n[None, :] < N), other=0.0)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    if bias_ptr is not None:
        bias_ptrs = bias_ptr + offs_m * stride_bm
        bias = tl.load(bias_ptrs, mask=offs_m < M, other=0.0)
        acc += bias[:, None]

    if ACTIVATION == "relu":
        acc = relu(acc)
    elif ACTIVATION == "tanh":
        acc = tl.tanh(acc)
    elif ACTIVATION == "gelu":
        acc = gelu(acc)
    elif ACTIVATION == "fast_gelu":
        acc = fast_gelu(acc)

    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    c_mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, acc.to(tl.float16), mask=c_mask)

class LinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, activation):
        M, K = x.shape
        N, _ = weight.shape
        
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)
        
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        
        kernel_fma[grid](
            x, weight, output, bias,
            M, N, K,
            x.stride(0), x.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            bias.stride(0) if bias is not None else 0,
            ACTIVATION=activation,
            GROUP_M=8,
            BLOCK_M=64,  # Defaults, autotune will override
            BLOCK_N=64,
            BLOCK_K=32
        )
        
        ctx.save_for_backward(x, weight, bias)
        ctx.activation = activation
        return output

    @staticmethod
    def backward(ctx, grad_output):
        # Backward pass implementation would require custom kernels or PyTorch autograd
        x, weight, bias = ctx.saved_tensors
        activation = ctx.activation
        
        grad_input = grad_weight = grad_bias = None
        
        if ctx.needs_input_grad[0]:
            grad_input = grad_output @ weight.T
        if ctx.needs_input_grad[1]:
            grad_weight = x.T @ grad_output
        if bias is not None and ctx.needs_input_grad[2]:
            grad_bias = grad_output.sum(0)
        
        # Handle activation derivative if needed
        return grad_input, grad_weight, grad_bias, None

def linear_layer(x, weight, bias=None, activation=None, save_preact=False):
    if activation not in [None, "relu", "tanh", "gelu", "fast_gelu"]:
        raise ValueError("Unsupported activation")
    
    if save_preact:
        preact = x @ weight
        if bias is not None:
            preact += bias.unsqueeze(0)
        output = LinearLayer.apply(x, weight, bias, activation)
        return output, preact
    else:
        return LinearLayer.apply(x, weight, bias, activation)
