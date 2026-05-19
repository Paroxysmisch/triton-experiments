import triton
import triton.language as tl

@triton.jit
def _geglu_tanh_forward_kernel(
    a_ptr, b_ptr, c_ptr,
    a_size, b_size, c_size,
    stride_a, stride_b, stride_c,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < a_size

    a_offsets = offsets * stride_a
    b_offsets = offsets * stride_b
    c_offsets = offsets * stride_c

    a = tl.load(a_ptr + a_offsets, mask=mask)
    b = tl.load(b_ptr + b_offsets, mask=mask)

    # Compute the GEGLU activation
    tanh_approx = tl.tanh(0.7978845608 * (b + 0.044715 * tl.pow(b, 3)))
    c = a * 0.5 * (1.0 + tanh_approx)

    tl.store(c_ptr + c_offsets, c, mask=mask)

@triton.jit
def _geglu_tanh_backward_kernel(
    grad_c_ptr, a_ptr, b_ptr, grad_a_ptr, grad_b_ptr,
    a_size, b_size, c_size,
    stride_grad_c, stride_a, stride_b, stride_grad_a, stride_grad_b,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < a_size

    a_offsets = offsets * stride_a
    b_offsets = offsets * stride_b
    grad_c_offsets = offsets * stride_grad_c
    grad_a_offsets = offsets * stride_grad_a
    grad_b_offsets = offsets * stride_grad_b

    a = tl.load(a_ptr + a_offsets, mask=mask)
    b = tl.load(b_ptr + b_offsets, mask=mask)
    grad_c = tl.load(grad_c_ptr + grad_c_offsets, mask=mask)

    # Compute the GEGLU activation
    tanh_approx = tl.tanh(0.7978845608 * (b + 0.044715 * tl.pow(b, 3)))
    c = a * 0.5 * (1.0 + tanh_approx)

    # Compute gradients
    grad_tanh_approx = 0.7978845608 * (1.0 - tl.pow(tanh_approx, 2)) * (1.0 + 0.134145 * tl.pow(b, 2))
    grad_b = grad_c * a * 0.5 * grad_tanh_approx
    grad_a = grad_c * 0.5 * (1.0 + tanh_approx)

    tl.store(grad_a_ptr + grad_a_offsets, grad_a, mask=mask)
    tl.store(grad_b_ptr + grad_b_offsets, grad_b, mask=mask)

### Python Wrapper Code

import torch
import triton
import triton.language as tl

def calculate_settings(size):
    BLOCK_SIZE = 128
    num_warps = 4
    return BLOCK_SIZE, num_warps

def geglu_forward(a, b):
    a = a.contiguous()
    b = b.contiguous()
    assert a.shape == b.shape, "Input tensors must have the same shape"
    c = torch.empty_like(a)

    BLOCK_SIZE, num_warps = calculate_settings(a.numel())
    grid = (triton.cdiv(a.numel(), BLOCK_SIZE),)

    _geglu_tanh_forward_kernel[grid](
        a, b, c,
        a.numel(), b.numel(), c.numel(),
        a.stride(0), b.stride(0), c.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return c

def geglu_backward(grad_c, a, b):
    grad_c = grad_c.contiguous()
    a = a.contiguous()
    b = b.contiguous()
    assert a.shape == b.shape == grad_c.shape, "Input tensors must have the same shape"
    grad_a = torch.empty_like(a)
    grad_b = torch.empty_like(b)

    BLOCK_SIZE, num_warps = calculate_settings(a.numel())
    grid = (triton.cdiv(a.numel(), BLOCK_SIZE),)

    _geglu_tanh_backward_kernel[grid](
        grad_c, a, b, grad_a, grad_b,
        a.numel(), b.numel(), grad_c.numel(),
        grad_c.stride(0), a.stride(0), b.stride(0), grad_a.stride(0), grad_b.stride(0),
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps
    )

    return grad_a, grad_b
