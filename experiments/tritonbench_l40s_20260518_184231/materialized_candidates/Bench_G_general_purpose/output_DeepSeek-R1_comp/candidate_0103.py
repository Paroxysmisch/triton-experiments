import torch
import triton
import triton.language as tl

@triton.jit
def _rms_layernorm_forward(
    x_ptr, w_ptr, o_ptr, var_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    x_ptrs = x_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_cols

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0) / n_cols
    inv_var = 1.0 / tl.sqrt(sum_sq + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE) % n_cols, mask=mask, other=0.0)
    y = x * inv_var * w
    tl.store(o_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE), y.to(x_ptr.dtype.element_ty), mask=mask)
    tl.store(var_ptr + row, inv_var)

@triton.jit
def _rms_layernorm_backward_dx(
    grad_o_ptr, x_ptr, w_ptr, inv_var_ptr,
    grad_x_ptr,
    n_cols,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    x_ptrs = x_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_cols

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE) % n_cols, mask=mask, other=0.0)
    grad_o = tl.load(grad_o_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE), mask=mask, other=0.0).to(tl.float32)
    inv_var = tl.load(inv_var_ptr + row)

    term1 = grad_o * w * inv_var
    dot_product = tl.sum(grad_o * w * x) * (inv_var ** 3) / n_cols
    term2 = x * dot_product
    grad_x = term1 - term2
    tl.store(grad_x_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE), grad_x.to(grad_x_ptr.dtype.element_ty), mask=mask)

@triton.jit
def _rms_layernorm_backward_dw(
    grad_o_ptr, x_ptr, inv_var_ptr,
    grad_w_ptr,
    n_rows, n_cols,
    BLOCK_SIZE: tl.constexpr
):
    col = tl.program_id(0)
    offs = col + tl.arange(0, BLOCK_SIZE) * n_cols
    mask = col + tl.arange(0, BLOCK_SIZE) * n_cols < n_rows * n_cols

    acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    for row in range(0, n_rows):
        x = tl.load(x_ptr + row * n_cols + col, mask=col < n_cols)
        grad_o = tl.load(grad_o_ptr + row * n_cols + col, mask=col < n_cols)
        inv_var = tl.load(inv_var_ptr + row)
        acc += grad_o.to(tl.float32) * x.to(tl.float32) * inv_var

    tl.store(grad_w_ptr + col, acc.to(grad_w_ptr.dtype.element_ty), mask=col < n_cols)

@triton.jit
def _gemma_rms_layernorm_forward(
    x_ptr, w_ptr, o_ptr, var_ptr,
    n_cols, eps,
    BLOCK_SIZE: tl.constexpr
):
    row = tl.program_id(0)
    x_ptrs = x_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE)
    mask = tl.arange(0, BLOCK_SIZE) < n_cols

    x = tl.load(x_ptrs, mask=mask, other=0.0).to(tl.float32)
    sum_sq = tl.sum(x * x, axis=0) / n_cols
    inv_var = 1.0 / tl.sqrt(sum_sq + eps)

    w = tl.load(w_ptr + tl.arange(0, BLOCK_SIZE) % n_cols, mask=mask, other=0.0)
    y = x * inv_var * (1.0 + w)
    tl.store(o_ptr + row * n_cols + tl.arange(0, BLOCK_SIZE), y.to(x_ptr.dtype.element_ty), mask=mask)
    tl.store(var_ptr + row, inv_var)

def calculate_settings(n_cols):
    block_size = 1
    while block_size <= n_cols and block_size < 1024:
        block_size <<= 1
    block_size >>= 1
    block_size = max(block_size, 16)
    num_warps = 4 if block_size < 256 else 8
    return block_size, num_warps

class Fast_RMS_Layernorm(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, eps=1e-6, use_gemma=False):
        n_rows, n_cols = x.shape
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        y = torch.empty_like(x)
        inv_var = torch.empty(n_rows, device=x.device, dtype=torch.float32)
        
        if use_gemma:
            _gemma_rms_layernorm_forward[(n_rows,)](x, weight, y, inv_var, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        else:
            _rms_layernorm_forward[(n_rows,)](x, weight, y, inv_var, n_cols, eps, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        
        ctx.save_for_backward(x, weight, inv_var)
        ctx.n_cols = n_cols
        ctx.use_gemma = use_gemma
        return y

    @staticmethod
    def backward(ctx, grad_output):
        x, weight, inv_var = ctx.saved_tensors
        n_cols = ctx.n_cols
        n_rows = x.size(0)
        
        grad_x = torch.empty_like(x)
        grad_w = torch.zeros_like(weight)
        
        BLOCK_SIZE, num_warps = calculate_settings(n_cols)
        _rms_layernorm_backward_dx[(n_rows,)](grad_output, x, weight, inv_var, grad_x, n_cols, BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps)
        
        BLOCK_SIZE_DW = 128
        grid_dw = (triton.cdiv(n_cols, BLOCK_SIZE_DW),)
        _rms_layernorm_backward_dw[grid_dw](grad_output, x, inv_var, grad_w, n_rows, n_cols, BLOCK_SIZE=BLOCK_SIZE_DW)
        
        return grad_x, grad_w, None, None

def fast_rms_layernorm(x, weight, eps=1e-6, use_gemma=False):
    return Fast_RMS_Layernorm.apply(x, weight, eps, use_gemma)
