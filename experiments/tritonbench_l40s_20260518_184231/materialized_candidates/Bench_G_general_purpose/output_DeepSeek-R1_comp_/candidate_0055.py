import torch
import triton
import triton.language as tl

@triton.jit
def silu(x):
    return x * tl.sigmoid(x)

@triton.jit
def _swiglu_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_cm, stride_ck,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    rm = pid
    for rk in range(0, N, BLOCK_SIZE):
        rk_offs = rk + tl.arange(0, BLOCK_SIZE)
        mask = rk_offs < N
        a = tl.load(a_ptr + rm * stride_am + rk_offs * stride_ak, mask=mask, other=0.0)
        b = tl.load(b_ptr + rm * stride_bm + rk_offs * stride_bk, mask=mask, other=0.0)
        silu_a = silu(a)
        c = silu_a * b
        tl.store(c_ptr + rm * stride_cm + rk_offs * stride_ck, c, mask=mask)

@triton.jit
def _swiglu_backward_kernel(
    grad_output_ptr, a_ptr, b_ptr,
    grad_a_ptr, grad_b_ptr,
    M, N,
    stride_gom, stride_gok,
    stride_am, stride_ak,
    stride_bm, stride_bk,
    stride_gam, stride_gak,
    stride_gbm, stride_gbk,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    if pid >= M:
        return
    rm = pid
    for rk in range(0, N, BLOCK_SIZE):
        rk_offs = rk + tl.arange(0, BLOCK_SIZE)
        mask = rk_offs < N
        go = tl.load(grad_output_ptr + rm * stride_gom + rk_offs * stride_gok, mask=mask, other=0.0)
        a = tl.load(a_ptr + rm * stride_am + rk_offs * stride_ak, mask=mask, other=0.0)
        b = tl.load(b_ptr + rm * stride_bm + rk_offs * stride_bk, mask=mask, other=0.0)
        s = tl.sigmoid(a)
        silu_a = a * s
        grad_a = go * b * (s + a * s * (1 - s))
        grad_b = go * silu_a
        tl.store(grad_a_ptr + rm * stride_gam + rk_offs * stride_gak, grad_a, mask=mask)
        tl.store(grad_b_ptr + rm * stride_gbm + rk_offs * stride_gbk, grad_b, mask=mask)

def calculate_settings(n):
    MAX_FUSED_SIZE = 4096
    max_block = min(n, MAX_FUSED_SIZE)
    block_size = 1
    while block_size <= max_block // 2:
        block_size *= 2
    if torch.cuda.is_available() and torch.version.hip is not None:
        warp_size = 64
    else:
        warp_size = 32
    num_warps = max(block_size // warp_size, 1)
    return block_size, num_warps

def swiglu_forward(a: torch.Tensor, b: torch.Tensor):
    assert a.shape == b.shape, "a and b must have the same shape"
    a = a.contiguous()
    b = b.contiguous()
    M, N = a.shape
    c = torch.empty_like(a)
    BLOCK_SIZE, num_warps = calculate_settings(N)
    grid = (M,)
    _swiglu_forward_kernel[grid](
        a, b, c,
        M, N,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return c

def swiglu_backward(grad_output: torch.Tensor, a: torch.Tensor, b: torch.Tensor):
    assert grad_output.shape == a.shape == b.shape, "All inputs must have the same shape"
    a = a.contiguous()
    b = b.contiguous()
    grad_output = grad_output.contiguous()
    M, N = a.shape
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)
    BLOCK_SIZE, num_warps = calculate_settings(N)
    grid = (M,)
    _swiglu_backward_kernel[grid](
        grad_output, a, b,
        grad_a, grad_b,
        M, N,
        grad_output.stride(0), grad_output.stride(1),
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        grad_a.stride(0), grad_a.stride(1),
        grad_b.stride(0), grad_b.stride(1),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return grad_a, grad_b
