import triton
import triton.language as tl
import torch

@triton.jit
def kernel_fma(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, bias_ptr,
    # Matrix dimensions
    M, N, K,
    # The stride variables represent how much to increase the ptr by when moving by 1 element in a particular dimension
    stride_am, stride_ak,  # A matrix strides
    stride_bk, stride_bn,  # B matrix strides 
    stride_cm, stride_cn,  # C matrix strides
    # Meta-parameters
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
    GROUP_SIZE_M: tl.constexpr,
    # Optional activation function
    ACTIVATION: tl.constexpr
):
    """Kernel for computing C = activation(A @ B + bias)"""
    
    # Program ID
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = GROUP_SIZE_M * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * GROUP_SIZE_M
    group_size_m = min(num_pid_m - first_pid_m, GROUP_SIZE_M)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    # Block pointers
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Iterate to compute matrix multiplication
    for k in range(0, K, BLOCK_SIZE_K):
        # Load the matrices from memory
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        # Compute matrix multiplication
        acc += tl.dot(a, b)
        # Advance the ptrs to the next K block
        a_ptrs += BLOCK_SIZE_K * stride_ak
        b_ptrs += BLOCK_SIZE_K * stride_bk
    
    # Add bias if provided
    if bias_ptr is not None:
        bias = tl.load(bias_ptr + offs_bn)
        acc += bias[None, :]
    
    # Apply activation function
    if ACTIVATION == "relu":
        acc = tl.maximum(acc, 0)
    elif ACTIVATION == "gelu":
        acc = acc * 0.5 * (1 + tl.math.erf(acc / 1.4142135623730951))
    elif ACTIVATION == "tanh":
        acc = tl.math.tanh(acc)
        
    # Store the result
    offs_cm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_cn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)

# Wrapper function for the kernel
def linear_layer(x, weight, bias=None, activation=None):
    """
    Compute linear layer: activation(x @ weight.T + bias)
    """
    batch_size, in_features = x.shape
    out_features, _ = weight.shape
    
    # Allocate output
    y = torch.empty((batch_size, out_features), device=x.device, dtype=x.dtype)
    
    # Launch kernel
    grid = lambda META: (
        triton.cdiv(batch_size, META['BLOCK_SIZE_M']) * \
        triton.cdiv(out_features, META['BLOCK_SIZE_N']),
    )
    
    kernel_fma[grid](
        x, weight, y, bias,
        batch_size, out_features, in_features,
        x.stride(0), x.stride(1),
        weight.stride(0), weight.stride(1),
        y.stride(0), y.stride(1),
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=256,
        BLOCK_SIZE_K=32,
        GROUP_SIZE_M=8,
        ACTIVATION=activation,
    )
    
    return y

# Example usage
class LinearLayer(torch.nn.Module):
    def __init__(self, in_features, out_features, activation=None, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.activation = activation
        
        self.weight = torch.nn.Parameter(torch.empty(out_features, in_features))
        if bias:
            self.bias = torch.nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter('bias', None)
        
        self.reset_parameters()
        
    def reset_parameters(self):
        torch.nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = torch.nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            torch.nn.init.uniform_(self.bias, -bound, bound)
            
    def forward(self, x):
        return linear_layer(x, self.weight, self.bias, self.activation)
