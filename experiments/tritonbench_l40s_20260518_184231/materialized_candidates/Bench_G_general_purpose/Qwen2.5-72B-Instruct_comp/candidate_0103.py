import triton
import triton.language as tl

@triton.jit
def _rms_layernorm_forward(
    X,  # input tensor
    W,  # weight tensor
    Y,  # output tensor
    RMS,  # row-wise RMS
    stride,  # stride of the input tensor
    N,  # number of elements in each row
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x_ptrs = X + pid * stride + offsets
    x = tl.load(x_ptrs, mask=mask)
    x = x.to(tl.float32)
    
    # Compute the row-wise variance
    x_mean = tl.sum(x, axis=0) / N
    x_centered = x - x_mean
    x_var = tl.sum(x_centered * x_centered, axis=0) / N
    inv_std = 1.0 / tl.sqrt(x_var + 1e-6)
    
    # Normalize the input data
    y = (x - x_mean) * inv_std
    w = tl.load(W + offsets, mask=mask)
    y = y * w
    
    # Store the output
    y_ptrs = Y + pid * stride + offsets
    tl.store(y_ptrs, y, mask=mask)
    
    # Store the row-wise RMS
    rms_ptrs = RMS + pid
    tl.store(rms_ptrs, inv_std)

@triton.jit
def _rms_layernorm_backward(
    dY,  # gradient of the output
    X,  # input tensor
    W,  # weight tensor
    RMS,  # row-wise RMS
    dX,  # gradient of the input
    dW,  # gradient of the weight
    stride,  # stride of the input tensor
    N,  # number of elements in each row
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x_ptrs = X + pid * stride + offsets
    dY_ptrs = dY + pid * stride + offsets
    dX_ptrs = dX + pid * stride + offsets
    w_ptrs = W + offsets
    dW_ptrs = dW + offsets
    rms_ptrs = RMS + pid
    
    x = tl.load(x_ptrs, mask=mask).to(tl.float32)
    dY = tl.load(dY_ptrs, mask=mask).to(tl.float32)
    w = tl.load(w_ptrs, mask=mask).to(tl.float32)
    inv_std = tl.load(rms_ptrs).to(tl.float32)
    
    # Compute the gradient with respect to the input
    dX = dY * w * inv_std
    tl.store(dX_ptrs, dX, mask=mask)
    
    # Compute the gradient with respect to the weight
    x_centered = (x - tl.sum(x, axis=0) / N) * inv_std
    dW = tl.sum(dY * x_centered, axis=0)
    tl.atomic_add(dW_ptrs, dW)

@triton.jit
def _gemma_rms_layernorm_forward(
    X,  # input tensor
    W,  # weight tensor
    Y,  # output tensor
    RMS,  # row-wise RMS
    stride,  # stride of the input tensor
    N,  # number of elements in each row
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < N
    x_ptrs = X + pid * stride + offsets
    x = tl.load(x_ptrs, mask=mask)
    x = x.to(tl.float32)
    
    # Compute the row-wise variance
    x_mean = tl.sum(x, axis=0) / N
    x_centered = x - x_mean
    x_var = tl.sum(x_centered * x_centered, axis=0) / N
    inv_std = 1.0 / tl.sqrt(x_var + 1e-6)
    
    # Normalize the input data
    y = (x - x_mean) * inv_std
    w = tl.load(W + offsets, mask=mask)
    y = y * (w + 1.0)
    
    # Store the output
    y_ptrs = Y + pid * stride + offsets
    tl.store(y_ptrs, y, mask=mask)
    
    # Store the row-wise RMS
    rms_ptrs = RMS + pid
    tl.store(rms_ptrs, inv_std)

import torch
import triton
import triton.language as tl

class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, X, W):
        N = X.shape[-1]
        stride = X.stride(-1)
        BLOCK_SIZE = 1024
        num_warps = 4
        
        Y = torch.empty_like(X)
        RMS = torch.empty((X.shape[0],), device=X.device, dtype=torch.float32)
        
        _rms_layernorm_forward[(X.shape[0],)](X, W, Y, RMS, stride, N, BLOCK_SIZE, num_warps=num_warps)
        
        ctx.save_for_backward(X, W, RMS)
        return Y
    
    @staticmethod
    def backward(ctx, dY):
        X, W, RMS = ctx.saved_tensors
        N = X.shape[-1]
        stride = X.stride(-1)
        BLOCK_SIZE = 1024
        num_warps = 4
        
        dX = torch.empty_like(X)
        dW = torch.empty_like(W)
        
        _rms_layernorm_backward[(X.shape[0],)](dY, X, W, RMS, dX, dW, stride, N, BLOCK_SIZE, num_warps=num_warps)
        
        return dX, dW

def fast_rms_layernorm(X, W):
    return Fast_RMS_Layernorm.apply(X, W)
