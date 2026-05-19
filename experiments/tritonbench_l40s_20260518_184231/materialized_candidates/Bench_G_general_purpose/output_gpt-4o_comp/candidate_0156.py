import triton
import triton.language as tl
import torch

# Kernel for the softmax operation
@triton.jit
def _softmax(
    X, OUT, stride_xm, stride_xn, stride_ym, stride_yn,
    N, LOG, CAUSAL, MASK_TYPE, DEPTH, IS_FP16,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset_m = pid * BLOCK_SIZE
    offsets_n = tl.arange(0, BLOCK_SIZE)
    offsets_m = offset_m + tl.arange(0, BLOCK_SIZE)
    
    # Load input
    X = tl.load(X + offsets_m[:, None] * stride_xm + offsets_n[None, :] * stride_xn, mask=offsets_m[:, None] < N, other=-float('inf'))
    
    # Compute max for numerical stability
    max_val = tl.max(X, axis=1)
    X = X - max_val[:, None]
    
    # Exponentiate
    exp_X = tl.exp(X)
    
    # Apply masking if necessary
    if CAUSAL:
        mask = offsets_n[None, :] <= offsets_m[:, None]
        exp_X = tl.where(mask, exp_X, 0.0)
    
    # Sum and normalize
    sum_exp_X = tl.sum(exp_X, axis=1)
    softmax_X = exp_X / sum_exp_X[:, None]
    
    # Logarithmic softmax
    if LOG:
        softmax_X = tl.log(softmax_X)
    
    # Store result
    tl.store(OUT + offsets_m[:, None] * stride_ym + offsets_n[None, :] * stride_yn, softmax_X, mask=offsets_m[:, None] < N)

# Host function for the softmax operation
def softmax(X, log=False, causal=False, mask_type=None):
    assert X.ndim == 3
    M, N, K = X.shape
    BLOCK_SIZE = 128  # Example block size, tune this for your hardware
    DEPTH = triton.next_power_of_2(K)
    IS_FP16 = X.dtype == torch.float16
    
    OUT = torch.empty_like(X)
    
    grid = lambda META: (M * N + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE']
    
    _softmax[grid](
        X, OUT, X.stride(0), X.stride(1), OUT.stride(0), OUT.stride(1),
        N, log, causal, mask_type, DEPTH, IS_FP16,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return OUT

# Kernel for the backward pass of the softmax operation
@triton.jit
def _softmax_backward(
    D_OUT, OUT, D_X, stride_dom, stride_don, stride_ym, stride_yn, stride_dxom, stride_dxon,
    N, LOG, CAUSAL, MASK_TYPE, DEPTH, IS_FP16,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    offset_m = pid * BLOCK_SIZE
    offsets_n = tl.arange(0, BLOCK_SIZE)
    offsets_m = offset_m + tl.arange(0, BLOCK_SIZE)
    
    # Load gradients and outputs
    D_OUT = tl.load(D_OUT + offsets_m[:, None] * stride_dom + offsets_n[None, :] * stride_don, mask=offsets_m[:, None] < N, other=0.0)
    OUT = tl.load(OUT + offsets_m[:, None] * stride_ym + offsets_n[None, :] * stride_yn, mask=offsets_m[:, None] < N, other=0.0)
    
    # Compute gradient of softmax
    D_X = OUT * (D_OUT - tl.sum(D_OUT * OUT, axis=1)[:, None])
    
    # Store result
    tl.store(D_X + offsets_m[:, None] * stride_dxom + offsets_n[None, :] * stride_dxon, D_X, mask=offsets_m[:, None] < N)

# Host function for the backward pass of the softmax operation
def softmax_backward(D_OUT, OUT, log=False, causal=False, mask_type=None):
    assert D_OUT.ndim == 3 and OUT.ndim == 3
    M, N, K = D_OUT.shape
    BLOCK_SIZE = 128  # Example block size, tune this for your hardware
    DEPTH = triton.next_power_of_2(K)
    IS_FP16 = D_OUT.dtype == torch.float16
    
    D_X = torch.empty_like(D_OUT)
    
    grid = lambda META: (M * N + META['BLOCK_SIZE'] - 1) // META['BLOCK_SIZE']
    
    _softmax_backward[grid](
        D_OUT, OUT, D_X, D_OUT.stride(0), D_OUT.stride(1), OUT.stride(0), OUT.stride(1), D_X.stride(0), D_X.stride(1),
        N, log, causal, mask_type, DEPTH, IS_FP16,
        BLOCK_SIZE=BLOCK_SIZE
    )
    
    return D_X
