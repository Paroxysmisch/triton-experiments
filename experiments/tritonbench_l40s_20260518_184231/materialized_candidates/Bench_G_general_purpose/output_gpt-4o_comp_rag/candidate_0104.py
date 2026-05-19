import triton
import triton.language as tl
import torch

# Function to precompute sine and cosine values for position-dependent transformations
def get_freq_multi_tokens(theta, length):
    positions = torch.arange(length, dtype=torch.float32)
    div_term = torch.exp(torch.arange(0, length, 2).float() * -(torch.log(torch.tensor(10000.0)) / length))
    freqs = torch.zeros((length, 2), dtype=torch.float32)
    freqs[:, 0] = torch.sin(positions * div_term)
    freqs[:, 1] = torch.cos(positions * div_term)
    return freqs

# Triton kernel
@triton.jit
def rbe_triton(X, stride_xm, stride_xk, Z, stride_zm, stride_zk, freqs, BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(axis=0)
    offs_m = pid * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    mask_m = offs_m < stride_xm
    mask_k = offs_k < stride_xk
    
    X_real = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :] * 2, mask=mask_m[:, None] & mask_k[None, :])
    X_imag = tl.load(X + offs_m[:, None] * stride_xm + offs_k[None, :] * 2 + 1, mask=mask_m[:, None] & mask_k[None, :])
    
    freq_real = tl.load(freqs + offs_k[None, :] * 2, mask=mask_k[None, :])
    freq_imag = tl.load(freqs + offs_k[None, :] * 2 + 1, mask=mask_k[None, :])
    
    out_real = X_real * freq_real - X_imag * freq_imag
    out_imag = X_real * freq_imag + X_imag * freq_real
    
    Z_real = Z + offs_m[:, None] * stride_zm + offs_k[None, :] * 2
    Z_imag = Z + offs_m[:, None] * stride_zm + offs_k[None, :] * 2 + 1
    
    tl.store(Z_real, out_real, mask=mask_m[:, None] & mask_k[None, :])
    tl.store(Z_imag, out_imag, mask=mask_m[:, None] & mask_k[None, :])

# Wrapper function
def rbe_triton_wrapper(x, out):
    batch, M, K = x.shape
    BLOCK_SIZE_M = 2
    BLOCK_SIZE_K = 1024
    
    freqs = get_freq_multi_tokens(10000, K).cuda()
    
    grid = lambda meta: (triton.cdiv(M, meta['BLOCK_SIZE_M']),)
    
    rbe_triton[grid](
        x, x.stride(1), x.stride(2),
        out, out.stride(1), out.stride(2),
        freqs,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K
    )

# Example usage
batch = 1
M = 4
K = 1024
x = torch.randn((batch, M, K), dtype=torch.float32, device='cuda')
out = torch.zeros_like(x, device='cuda')

rbe_triton_wrapper(x, out)
print(out)
