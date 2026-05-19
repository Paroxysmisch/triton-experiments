import torch
import triton
import triton.language as tl

# Activation Functions
@triton.jit
def tanh_activation(x):
    return tl.tanh(x)

@triton.jit
def relu_activation(x):
    return tl.maximum(x, 0.0)

@triton.jit
def gelu_activation(x):
    return x * 0.5 * (1.0 + tl.erf(x / tl.sqrt(2.0)))

@triton.jit
def fast_gelu_activation(x):
    return 0.5 * x * (1.0 + tl.tanh(x * 0.7978845608 * (1.0 + 0.044715 * x * x)))

@triton.jit
def no_activation(x):
    return x

# Main Triton Kernel
@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 128, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 256, 'BLOCK_K': 64}, num_stages=3, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def kernel_fma(
    # Pointers to matrices
    A, B, C,
    # Matrix dimensions
    M, N, K,
    # Stride values
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Optional parameters
    bias_ptr,
    activation: tl.constexpr,
    # Meta-parameters
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    a_ptrs = A + (offs_m[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = B + (offs_k[:, None] * stride_bk + offs_n[None, :] * stride_bn)
    
    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, K, BLOCK_K):
        a = tl.load(a_ptrs, mask=(offs_k[None, :] < K - k), other=0.0)
        b = tl.load(b_ptrs, mask=(offs_k[:, None] < K - k), other=0.0)
        accumulator += tl.dot(a, b)
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk
    
    # Add bias if present
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_n, mask=offs_n < N, other=0.0)
        accumulator += bias[None, :]
    
    # Apply activation
    if activation == 1:
        accumulator = relu_activation(accumulator)
    elif activation == 2:
        accumulator = tanh_activation(accumulator)
    elif activation == 3:
        accumulator = gelu_activation(accumulator)
    elif activation == 4:
        accumulator = fast_gelu_activation(accumulator)
    
    # Write back result
    c_ptrs = C + (offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn)
    tl.store(c_ptrs, accumulator, mask=(offs_m[:, None] < M) & (offs_n[None, :] < N))

# PyTorch Wrapper
class LinearLayer(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input, weight, bias, activation):
        # Check dimensions
        assert input.dim() == 2, "Input must be 2D"
        M, K = input.shape
        N, K_w = weight.shape
        assert K == K_w, "Input and weight dimensions mismatch"
        
        output = torch.empty((M, N), device=input.device, dtype=input.dtype)
        
        # Map activation to integer code
        activation_code = 0
        if activation == 'relu': activation_code = 1
        elif activation == 'tanh': activation_code = 2
        elif activation == 'gelu': activation_code = 3
        elif activation == 'fast_gelu': activation_code = 4
        
        grid = lambda meta: (triton.cdiv(M, meta['BLOCK_M']) * triton.cdiv(N, meta['BLOCK_N']),)
        
        kernel_fma[grid](
            input, weight, output,
            M, N, K,
            input.stride(0), input.stride(1),
            weight.stride(0), weight.stride(1),
            output.stride(0), output.stride(1),
            bias,
            activation_code,
        )
        
        ctx.save_for_backward(input, weight, bias)
        ctx.activation = activation
        return output

def linear_layer(input, weight, bias=None, activation='none'):
    activation = activation.lower()
    assert activation in ['none', 'relu', 'tanh', 'gelu', 'fast_gelu']
    
    # Convert to contiguous tensors
    input = input.contiguous()
    weight = weight.contiguous().T  # Triton uses column-major
    if bias is not None:
        bias = bias.contiguous()
    
    return LinearLayer.apply(input, weight, bias, activation)

# Example usage
batch_size, in_features, out_features = 512, 1024, 2048
input = torch.randn(batch_size, in_features, device='cuda')
weight = torch.randn(out_features, in_features, device='cuda')
bias = torch.randn(out_features, device='cuda')

# With ReLU activation
output = linear_layer(input, weight, bias, activation='relu')

# With GELU activation
output = linear_layer(input, weight, bias, activation='gelu')
