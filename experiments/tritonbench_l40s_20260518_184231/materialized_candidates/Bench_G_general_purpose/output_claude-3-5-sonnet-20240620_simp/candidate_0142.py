import triton
import triton.language as tl
import torch

@triton.jit
def _geglu_tanh_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_cm, stride_cn,
    M, N,
    BLOCK_SIZE: tl.constexpr
):
    # Position of elements
    pid = tl.program_id(0)
    
    # Number of elements per block
    n_elements = BLOCK_SIZE
    
    # Offset calculations
    offs_m = pid * n_elements + tl.arange(0, n_elements)
    offs_n = tl.arange(0, N)
    
    # Create mask for bounds checking
    mask = offs_m < M
    
    # Load input tensors
    a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=mask[:, None])
    b = tl.load(b_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn, mask=mask[:, None])
    
    # Compute GEGLU with tanh approximation
    # GELU(x) ≈ 0.5x * (1 + tanh(sqrt(2/π) * (x + 0.044715x^3)))
    x = b
    x3 = x * x * x
    inner = 0.797885 * (x + 0.044715 * x3)  # sqrt(2/π) ≈ 0.797885
    gelu = 0.5 * x * (1.0 + tl.tanh(inner))
    
    # Multiply with gate (a)
    output = a * gelu
    
    # Store result
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn, output, mask=mask[:, None])

@triton.jit
def _geglu_tanh_backward_kernel(
    grad_output_ptr, a_ptr, b_ptr,
    grad_a_ptr, grad_b_ptr,
    stride_gom, stride_gon,
    stride_am, stride_an,
    stride_bm, stride_bn,
    stride_gam, stride_gan,
    stride_gbm, stride_gbn,
    M, N,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    n_elements = BLOCK_SIZE
    
    offs_m = pid * n_elements + tl.arange(0, n_elements)
    offs_n = tl.arange(0, N)
    
    mask = offs_m < M
    
    # Load inputs
    grad_output = tl.load(grad_output_ptr + offs_m[:, None] * stride_gom + offs_n[None, :] * stride_gon, mask=mask[:, None])
    a = tl.load(a_ptr + offs_m[:, None] * stride_am + offs_n[None, :] * stride_an, mask=mask[:, None])
    b = tl.load(b_ptr + offs_m[:, None] * stride_bm + offs_n[None, :] * stride_bn, mask=mask[:, None])
    
    # GELU derivative computation
    x = b
    x2 = x * x
    x3 = x2 * x
    inner = 0.797885 * (x + 0.044715 * x3)
    tanh_inner = tl.tanh(inner)
    
    # d(GELU)/dx = 0.5 * (1 + tanh(inner) + x * (1 - tanh^2(inner)) * (0.797885 * (1 + 0.134145x^2)))
    gelu_grad = 0.5 * (1.0 + tanh_inner + 
                       x * (1.0 - tanh_inner * tanh_inner) * 
                       (0.797885 * (1.0 + 0.134145 * x2)))
    
    # Compute gradients
    grad_a = grad_output * (0.5 * x * (1.0 + tanh_inner))
    grad_b = grad_output * a * gelu_grad
    
    # Store gradients
    tl.store(grad_a_ptr + offs_m[:, None] * stride_gam + offs_n[None, :] * stride_gan, grad_a, mask=mask[:, None])
    tl.store(grad_b_ptr + offs_m[:, None] * stride_gbm + offs_n[None, :] * stride_gbn, grad_b, mask=mask[:, None])

def calculate_settings(M, N):
    BLOCK_SIZE = 128
    num_warps = 4
    return BLOCK_SIZE, num_warps

def geglu_forward(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda
    assert a.shape == b.shape
    assert a.dtype == b.dtype
    
    M, N = a.shape
    BLOCK_SIZE, num_warps = calculate_settings(M, N)
    
    # Compute number of blocks needed
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)
    
    # Output tensor
    c = torch.empty_like(a)
    
    # Launch kernel
    _geglu_tanh_forward_kernel[grid](
        a_ptr=a, b_ptr=b, c_ptr=c,
        stride_am=a.stride(0), stride_an=a.stride(1),
        stride_bm=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        M=M, N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return c

def geglu_backward(grad_output: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    assert grad_output.is_cuda and a.is_cuda and b.is_cuda
    assert grad_output.shape == a.shape == b.shape
    assert grad_output.dtype == a.dtype == b.dtype
    
    M, N = a.shape
    BLOCK_SIZE, num_warps = calculate_settings(M, N)
    
    # Compute number of blocks needed
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE']),)
    
    # Output gradients
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    
    # Launch kernel
    _geglu_tanh_backward_kernel[grid](
        grad_output_ptr=grad_output, 
        a_ptr=a, b_ptr=b,
        grad_a_ptr=grad_a, grad_b_ptr=grad_b,
        stride_gom=grad_output.stride(0), stride_gon=grad_output.stride(1),
        stride_am=a.stride(0), stride_an=a.stride(1),
        stride_bm=b.stride(0), stride_bn=b.stride(1),
        stride_gam=grad_a.stride(0), stride_gan=grad_a.stride(1),
        stride_gbm=grad_b.stride(0), stride_gbn=grad_b.stride(1),
        M=M, N=N,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )
    
    return grad_a, grad_b
