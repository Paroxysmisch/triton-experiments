import triton
import triton.language as tl
import torch
import math

@triton.jit
def _rms_layernorm_forward(
    X, Y, W, Rstd,
    stride_x, stride_y,
    N, eps,
    BLOCK_SIZE: tl.constexpr,
    IS_EVEN: tl.constexpr,
    GEMMA: tl.constexpr  # New flag for Gemma variant
):
    row = tl.program_id(0)
    x_ptrs = X + row * stride_x + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < N
    
    # Load input data
    x = tl.load(x_ptrs, mask=mask if not IS_EVEN else None, other=0.0).to(tl.float32)
    
    # Compute variance
    x_sq = x * x
    var = tl.sum(x_sq, axis=0) / N
    rstd = 1.0 / tl.sqrt(var + eps)
    tl.store(Rstd + row, rstd)
    
    # Normalize and scale
    x_norm = x * rstd
    w_ptrs = W + tl.arange(0, BLOCK_SIZE)
    w = tl.load(w_ptrs, mask=mask if not IS_EVEN else None).to(tl.float32)
    
    if GEMMA:
        w += 1.0  # Gemma variant adds 1.0 to weights
    
    y = x_norm * w
    y_ptrs = Y + row * stride_y + tl.arange(0, BLOCK_SIZE)
    tl.store(y_ptrs, y, mask=mask if not IS_EVEN else None)

@triton.jit
def _rms_layernorm_backward(
    DY, X, W, DX, DW_partial,
    Rstd, stride_dy, stride_x, stride_dx,
    M, N, eps,
    BLOCK_SIZE: tl.constexpr,
    IS_EVEN: tl.constexpr,
    GEMMA: tl.constexpr
):
    pid = tl.program_id(0)
    cols = tl.arange(0, BLOCK_SIZE)
    mask = cols < N
    
    # Initialize partial sums for DW
    dw = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    
    rows_per_program = tl.cdiv(M, tl.num_programs(0))
    start_row = pid * rows_per_program
    end_row = min((pid + 1) * rows_per_program, M)
    
    for row in range(start_row, end_row):
        # Load data
        x_ptrs = X + row * stride_x + cols
        dy_ptrs = DY + row * stride_dy + cols
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        dy = tl.load(dy_ptrs, mask=mask, other=0.0)
        rstd = tl.load(Rstd + row)
        
        # Compute normalized x
        x_norm = x * rstd
        w = tl.load(W + cols, mask=mask, other=0.0)
        if GEMMA:
            w += 1.0
        
        # Compute gradients
        wdy = w * dy
        dw += dy * x_norm
        
        # Compute dx components
        c1 = tl.sum(x_norm * wdy, axis=0) / N
        dx = (wdy - x_norm * c1) * rstd
        
        # Store dx
        dx_ptrs = DX + row * stride_dx + cols
        tl.store(dx_ptrs, dx, mask=mask)
    
    # Store partial DW sums
    dw_ptrs = DW_partial + pid * N + cols
    tl.store(dw_ptrs, dw, mask=mask)

class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps, gemma=False):
        # Shape checks
        M, N = x.shape
        assert weight.shape == (N,)
        
        # Allocate outputs
        y = torch.empty_like(x)
        rstd = torch.empty(M, dtype=torch.float32, device=x.device)
        
        # Kernel configuration
        BLOCK_SIZE = triton.next_power_of_2(N)
        IS_EVEN = N % BLOCK_SIZE == 0
        
        # Choose kernel variant
        kernel = _rms_layernorm_forward
        if gemma:
            kernel = _gemma_rms_layernorm_forward  # Assumes similar kernel with GEMMA=True
        
        # Launch kernel
        grid = (M,)
        kernel[grid](
            x, y, weight, rstd,
            x.stride(0), y.stride(0),
            N, eps,
            BLOCK_SIZE, IS_EVEN,
            GEMMA=gemma,
            num_warps=calculate_settings(N)[1]
        )
        
        ctx.save_for_backward(x, weight, rstd)
        ctx.eps = eps
        ctx.gemma = gemma
        return y
    
    @staticmethod
    def backward(ctx, dy):
        x, weight, rstd = ctx.saved_tensors
        M, N = x.shape
        
        # Allocate gradients
        dx = torch.empty_like(x)
        dw = torch.empty_like(weight)
        
        # Kernel configuration
        BLOCK_SIZE = triton.next_power_of_2(N)
        IS_EVEN = N % BLOCK_SIZE == 0
        sm_count = torch.cuda.get_device_properties(x.device).multi_processor_count
        
        # Temporary storage for DW partial sums
        dw_partial = torch.empty((sm_count, N), device=x.device, dtype=torch.float32)
        
        # Launch backward kernel
        grid = (sm_count,)
        _rms_layernorm_backward[grid](
            dy, x, weight, dx, dw_partial,
            rstd, dy.stride(0), x.stride(0), dx.stride(0),
            M, N, ctx.eps,
            BLOCK_SIZE, IS_EVEN, ctx.gemma,
            num_warps=calculate_settings(N)[1]
        )
        
        # Sum partial DW gradients
        dw = dw_partial.sum(0).to(weight.dtype)
        return dx, dw, None, None

def fast_rms_layernorm(x, weight, eps=1e-6, gemma=False):
    M, N = x.shape
    if N > 65536 // x.element_size():
        raise ValueError("Feature dimension too large for kernel")
    
    BLOCK_SIZE, num_warps = calculate_settings(N)
    return Fast_RMS_Layernorm.apply(x, weight, eps, gemma)

def calculate_settings(N):
    if N < 512: return 128, 4
    if N < 2048: return 512, 8
    return 1024, 16

x = torch.randn(1024, 512, device='cuda')
weight = torch.randn(512, device='cuda')

# Standard forward
y = fast_rms_layernorm(x, weight)

# Gemma variant forward
y_gemma = fast_rms_layernorm(x, weight, gemma=True)
