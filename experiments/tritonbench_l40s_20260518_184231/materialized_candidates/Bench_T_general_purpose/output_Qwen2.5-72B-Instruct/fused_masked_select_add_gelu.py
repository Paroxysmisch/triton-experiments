import triton
import triton.language as tl

@triton.jit
def fused_masked_select_add_gelu_kernel(
    X_ptr,  # Pointer to input tensor
    M_ptr,  # Pointer to mask tensor
    O_ptr,  # Pointer to other tensor or scalar
    Y_ptr,  # Pointer to output tensor
    X_strides,  # Strides for input tensor
    M_strides,  # Strides for mask tensor
    O_strides,  # Strides for other tensor
    Y_strides,  # Strides for output tensor
    X_shape,  # Shape of input tensor
    M_shape,  # Shape of mask tensor
    O_shape,  # Shape of other tensor
    Y_shape,  # Shape of output tensor
    alpha,  # Scalar to scale the other tensor
    approximate,  # Approximation method for GELU
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X_shape[0]

    x_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + offsets
    m_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + offsets
    o_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + offsets
    y_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + offsets

    x = tl.load(X_ptr + x_offsets, mask=mask, other=0.0)
    m = tl.load(M_ptr + m_offsets, mask=mask, other=0.0)
    o = tl.load(O_ptr + o_offsets, mask=mask, other=0.0)

    selected = tl.where(m, x, 0.0)
    scaled_other = alpha * o
    result = selected + scaled_other

    if approximate == 'tanh':
        result = 0.5 * result * (1 + tl.tanh(tl.sqrt(2 / 3.141592653589793) * (result + 0.044715 * result * result * result)))
    else:
        result = 0.5 * result * (1 + tl.math.erf(result / tl.sqrt(2.0)))

    tl.store(Y_ptr + y_offsets, result, mask=mask)

import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_SIZE': 128}, num_warps=4),
        triton.Config({'BLOCK_SIZE': 256}, num_warps=8),
    ],
    key=['X_shape', 'M_shape', 'O_shape', 'Y_shape'],
)
@triton.jit
def fused_masked_select_add_gelu_kernel(
    X_ptr,  # Pointer to input tensor
    M_ptr,  # Pointer to mask tensor
    O_ptr,  # Pointer to other tensor or scalar
    Y_ptr,  # Pointer to output tensor
    X_strides,  # Strides for input tensor
    M_strides,  # Strides for mask tensor
    O_strides,  # Strides for other tensor
    Y_strides,  # Strides for output tensor
    X_shape,  # Shape of input tensor
    M_shape,  # Shape of mask tensor
    O_shape,  # Shape of other tensor
    Y_shape,  # Shape of output tensor
    alpha,  # Scalar to scale the other tensor
    approximate,  # Approximation method for GELU
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE

    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < X_shape[0]

    x_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + offsets
    m_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + offsets
    o_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + offsets
    y_offsets = tl.zeros((BLOCK_SIZE,), dtype=tl.int32) + offsets

    x = tl.load(X_ptr + x_offsets, mask=mask, other=0.0)
    m = tl.load(M_ptr + m_offsets, mask=mask, other=0.0)
    o = tl.load(O_ptr + o_offsets, mask=mask, other=0.0)

    selected = tl.where(m, x, 0.0)
    scaled_other = alpha * o
    result = selected + scaled_other

    if approximate == 'tanh':
        result = 0.5 * result * (1 + tl.tanh(tl.sqrt(2 / 3.141592653589793) * (result + 0.044715 * result * result * result)))
    else:
        result = 0.5 * result * (1 + tl.math.erf(result / tl.sqrt(2.0)))

    tl.store(Y_ptr + y_offsets, result, mask=mask)

def fused_masked_select_add_gelu(input, mask, other, *, alpha=1, approximate='none', out=None):
    if out is None:
        out = torch.empty_like(input)

    grid = (triton.cdiv(input.numel(), 128),)
    fused_masked_select_add_gelu_kernel[grid](
        input, mask, other, out,
        input.stride(0), mask.stride(0), other.stride(0), out.stride(0),
        input.numel(), mask.numel(), other.numel(), out.numel(),
        alpha, approximate
    )
    return out
