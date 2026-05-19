import torch
import triton
import triton.language as tl
import math

@triton.jit
def _fused_bmm_dropout_gelu_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr, mask_ptr,
    # Matrix dimensions
    B, N, M, P,
    # Strides
    stride_ab, stride_an, stride_am,
    stride_bb, stride_bm, stride_bp,
    stride_cb, stride_cn, stride_cp,
    # Dropout probability and scale
    p, scale,
    # Whether using tanh approximation
    approximate: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    # Matrix multiplication indices
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(N, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(P, BLOCK_SIZE_N)
    num_pid_b = B
    
    # Batch, row, and column indices
    bid = pid // (num_pid_m * num_pid_n)
    pid_mn = pid % (num_pid_m * num_pid_n)
    pid_m = pid_mn // num_pid_n
    pid_n = pid_mn % num_pid_n

    # Block pointers
    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    
    a_ptrs = a_ptr + bid * stride_ab + offs_am[:, None] * stride_an + offs_k[None, :] * stride_am
    b_ptrs = b_ptr + bid * stride_bb + offs_k[:, None] * stride_bm + offs_bn[None, :] * stride_bp
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    
    # Matrix multiplication
    for k in range(0, tl.cdiv(M, BLOCK_SIZE_K)):
        a = tl.load(a_ptrs)
        b = tl.load(b_ptrs)
        acc += tl.dot(a, b)
        a_ptrs += BLOCK_SIZE_K * stride_am
        b_ptrs += BLOCK_SIZE_K * stride_bm
    
    # Apply dropout mask and scale
    mask = tl.load(mask_ptr + bid * N * P + offs_am[:, None] * P + offs_bn[None, :])
    acc = acc * mask * scale
    
    # GELU activation
    if approximate:
        # Tanh approximation
        acc = 0.5 * acc * (1.0 + tl.tanh(math.sqrt(2.0 / math.pi) * (acc + 0.044715 * acc * acc * acc)))
    else:
        # Standard GELU
        acc = 0.5 * acc * (1.0 + tl.erf(acc / math.sqrt(2.0)))
    
    # Store result
    c_ptrs = c_ptr + bid * stride_cb + offs_am[:, None] * stride_cn + offs_bn[None, :] * stride_cp
    tl.store(c_ptrs, acc)

def fused_bmm_dropout_gelu(input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None):
    # Input validation
    assert input1.dim() == 3 and input2.dim() == 3, "Input tensors must be 3D"
    B, N, M = input1.shape
    B2, M2, P = input2.shape
    assert B == B2 and M == M2, "Incompatible batch and inner dimensions"
    assert 0 <= p <= 1, "Dropout probability must be between 0 and 1"
    
    # Handle output tensor
    if out is None:
        out = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)
    elif out.shape != (B, N, P):
        raise RuntimeError(f"Expected output shape {(B, N, P)}, got {out.shape}")
    
    # Create dropout mask
    if training:
        mask = torch.empty((B, N, P), device=input1.device, dtype=input1.dtype)
        mask.bernoulli_(1 - p)
        scale = 1.0 / (1.0 - p)
    else:
        mask = torch.ones((B, N, P), device=input1.device, dtype=input1.dtype)
        scale = 1.0
    
    # Grid and block sizes
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 16
    grid = (B * triton.cdiv(N, BLOCK_SIZE_M) * triton.cdiv(P, BLOCK_SIZE_N),)
    
    # Launch kernel
    _fused_bmm_dropout_gelu_kernel[grid](
        input1, input2, out, mask,
        B, N, M, P,
        input1.stride(0), input1.stride(1), input1.stride(2),
        input2.stride(0), input2.stride(1), input2.stride(2),
        out.stride(0), out.stride(1), out.stride(2),
        p, scale,
        approximate == 'tanh',
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K
    )
    
    return out
