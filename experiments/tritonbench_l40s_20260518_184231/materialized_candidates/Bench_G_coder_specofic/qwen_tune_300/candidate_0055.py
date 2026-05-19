import triton
import triton.language as tl
import torch
from .linalg import _dot

TRITON_22 = version.parse(triton.__version__) >= version.parse('2.2.0')

@triton.jit
def swiglu_tie_w_kernel(x_ptr, w_ptr, y_ptr, N, C, C2, stride_x_n, stride_x_c, stride_w_n, stride_w_c, stride_y_n, stride_y_c,
                        BLOCK_SIZE: tl.constexpr, tie_w: tl.constexpr, w2_cache: tl.constexpr):
    pid = tl.program_id(axis=0)
    X = tl.arange(0, BLOCK_SIZE)
    mask = X < N
    x = tl.load(x_ptr + pid * stride_x_n + X * stride_x_c, mask, other=0.0)
    w_cache = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for i in range(C2):
        w2 = tl.load(w_ptr + i * stride_w_c + X * stride_w_c, mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
        if w2_cache:
            w_cache[X] = w2
        if tie_w:
            w1_ptr = w_ptr + C2 * stride_w_c + i * 2 * stride_w_c
        else:
            w1_ptr = w_ptr + i * stride_w_c
        w1 = tl.load(w1_ptr + X * stride_w_c, mask, other=0.0, eviction_policy='evict_first').to(tl.float32)
        y = w1 * x
        if not w2_cache:
            y += tl.sigmoid(w2) * x
        else:
            y += tl.sigmoid(w_cache[X]) * x
        tl.store(y_ptr + pid * stride_y_n + i * stride_y_c + X * stride_y_c, y, mask=mask)

@triton.jit
def swiglu_untie_w_kernel(x_ptr, w_ptr, y_ptr, N, C, C2, stride_x_n, stride_x_c, stride_w_n, stride_w_c, stride_y_n, stride_y_c,
                          BLOCK_SIZE: tl.constexpr, w2_cache: tl.constexpr):
    pid = tl.program_id(axis=0)
    X = tl.arange(0, BLOCK_SIZE)
    mask = X < N
    x = tl.load(x_ptr + pid * stride_x_n + X * stride_x_c, mask, other=0.0)
    w_cache = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for i in range(C2):
        if i % 2 == 0:
            w = tl.load(w_ptr + i * stride_w_c + X * stride_w_c, mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
            if w2_cache:
                w_cache[X] = w
        else:
            w2 = tl.load(w_ptr + i * stride_w_c + X * stride_w_c, mask, other=0.0, eviction_policy='evict_last').to(tl.float32)
            w = w * tl.sigmoid(w2)
            if w2_cache:
                w_cache[X] = w
            else:
                w = w * tl.sigmoid(w_cache[X % BLOCK_SIZE])
        tl.store(y_ptr + pid * stride_y_n + i * stride_y_c + X * stride_y_c, w * x, mask=mask)

def swiglu_forward(a, b, tie_w=True, w2_cache=True):
    if a.stride(-1) != 1:
        a = a.contiguous()
    if b.stride(-1) != 1:
        b = b.contiguous()
    a_ = a.unsqueeze(0) if a.ndim == 1 else a
    b_ = b.unsqueeze(0) if b.ndim == 1 else b
    assert a_.shape[-2:] == b_.shape[-2:], 'Incompatible dimensions between a and b'
    b2 = b_ * tl.sigmoid(b_)
    if tie_w:
        assert b_.stride(-1) == 1, 'Tie_w only supports cached access to b'
    N, C = b_.shape[-2:]
    C2 = C // 2
    y = torch.empty((N, C2), device=b.device, dtype=torch.float32)
    BLOCK_SIZE, num_warps = calculate_settings(C2)
    if tie_w:
        swiglu_tie_w_kernel[(N, )](a_, b_, y, N, C, C2, a_.stride(0), a_.stride(1), b_.stride(0), b_.stride(1), y.stride(0),
                                    y.stride(1), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, tie_w=True, w2_cache=w2_cache)
    else:
        swiglu_untie_w_kernel[(N, )](a_, b_, y, N, C, C2, a_.stride(0), a_.stride(1), b_.stride(0), b_.stride(1), y.stride(0),
                                     y.stride(1), BLOCK_SIZE=BLOCK_SIZE, num_warps=num_warps, w2_cache=w2_cache)
    return y.squeeze(0) if a.ndim == 1 else y

@triton.jit
def swiglu_tie_w_backward_kernel(x_ptr, w_ptr, dy_ptr, dx_ptr, dw_ptr, N, C, C2, stride_x_n, stride_x_c, stride_w_n, stride_w_c,
                                 stride_dy_n, stride_dy_c, stride_dx_n, stride_dx_c, stride_dw_n, stride_dw_c, BLOCK_SIZE: tl.constexpr,
                                 tie_w: tl.constexpr, w2_cache: tl.constexpr):
    pid = tl.program_id(axis=0)
    X = tl.arange(0, BLOCK_SIZE)
    mask = X < N
    dy = tl.load(dy_ptr + pid * stride_dy_n + X * stride_dy_c, mask, other=0.0)
    if tie_w:
        w2 = tl.load(w_ptr + C2 * stride_w_c + X * stride_w_c, mask, other=0.0, eviction_policy='evict_last')
    x = tl.load(x_ptr + pid * stride_x_n + X * stride_x_c, mask, other=0.0)
    tl.store(dx_ptr + pid * stride_dx_n + X * stride_dx_c, dy * x, mask=mask)
    wgrad = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for i in range(C2):
        if tie_w:
            w1 = tl.load(w_ptr + i * stride_w_c + X * stride_w_c, mask, other=0.0, eviction_policy='evict_first')
        else:
            w1_ptr = w_ptr + i * stride_w_c
            w1 = tl.load(w1_ptr + X * stride_w_c, mask, other=0.0, eviction_policy='evict_first')
        w2 = tl.load(w_ptr + i * stride_w_c + X * stride_w_c + stride_w_c, mask, other=0.0, eviction_policy='evict_last')
        if w2_cache:
            w2 = tl.sigmoid(w2)
        else:
            w2 = tl.sigmoid(w2_cache[X])
        wgrad += dy * (w1 * (1 - w2) * x + tl.sigmoid(w2) * x)
    tl.store(dw_ptr + pid * stride_dw_n + X * stride_dw_c, wgrad, mask=mask)

@triton.jit
def swiglu_untie_w_backward_kernel(x_ptr, w_ptr, dy_ptr, dx_ptr, dw_ptr, N, C, C2, stride_x_n, stride_x_c, stride_w_n, stride_w_c,
                                   stride_dy_n, stride_dy_c, stride_dx_n, stride_dx_c, stride_dw_n, stride_dw_c,
                                   BLOCK_SIZE: tl.constexpr, w2_cache: tl.constexpr):
    pid = tl.program_id(axis=0)
    X = tl.arange(0, BLOCK_SIZE)
    mask = X < N
    dy = tl.load(dy_ptr + pid * stride_dy_n + X * stride_dy_c, mask, other=0.0)
    wgrad = tl.zeros((BLOCK_SIZE, ), dtype=tl.float32)
    for i in range(C2):
        w1 = tl.load(w_ptr + i * stride_w_c + X * stride_w_c, mask, other=0.0, eviction_policy='evict_first')
        if i % 2 == 0:
            w2 = tl.load(w_ptr + i * stride_w_c + X * stride_w_c + stride_w_c, mask, other=0.0, eviction_policy='evict_last')
            if w2_cache:
                w2 = tl.sigmoid(w2)
            else:
                w2 = tl.sigmoid(w2_cache[X])
            wgrad += dy * (w1 * (1 - w2) * x + tl.sigmoid(w2) * x)
        else:
            w = tl.load(w_ptr + i * stride_w_c + X * stride_w_c, mask, other=0.0, eviction_policy='evict_first')
            tl.store(dx_ptr + pid * stride_dx_n + i * stride_dx_c + X * stride_dx_c, dy * w * x)
            wgrad += dy * w * x
    tl.store(dw_ptr + pid *
