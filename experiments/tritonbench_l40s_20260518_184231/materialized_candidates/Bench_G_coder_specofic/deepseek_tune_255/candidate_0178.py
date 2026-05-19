import triton
import triton.language as tl
import torch

@triton.jit
def diag_ssm_forward_kernel(
    x_ptr,
    s_ptr,
    y_ptr,
    lambda_ptr,
    length_ptr,
    output_ptr,
    batch_size,
    dim,
    T,
    BLOCK_SIZE: tl.constexpr,
):
    # Kernel function for forward pass of SSM
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + arange
    mask = arange < dim * batch_size
    x_strided = tl.load(x_ptr + offsets, mask=mask, other=0)

    s_strided = tl.load(s_ptr + offsets, mask=mask, other=0)
    lambda_strided = tl.load(lambda_ptr + offsets, mask=mask, other=0)
    length = tl.load(length_ptr)
    output = tl.load(output_ptr)

    for t in range(T):
        if output:
            s_strided = s_strided * lambda_strided + x_strided
            tl.store(y_ptr + offsets, s_strided, mask=mask)
        else:
            tl.store(y_ptr + offsets, s_strided, mask=mask)
            s_strided = s_strided * lambda_strided + x_strided
        x_ptr += dim * batch_size
        x_strided = tl.load(x_ptr + offsets, mask=mask, other=0)
        y_ptr += dim * batch_size

        if t < length - 1:
            s_ptr += dim * batch_size
            s_strided = tl.load(s_ptr + offsets, mask=mask, other=0)

@triton.jit
def diag_ssm_forward_kernel_complex(
    x_ptr,
    s_ptr,
    y_ptr,
    lambda_ptr,
    length_ptr,
    output_ptr,
    batch_size,
    dim,
    T,
    BLOCK_SIZE: tl.constexpr,
):
    # Kernel function for forward pass of SSM with complex numbers
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + arange
    mask = arange < dim * batch_size

    x_real_strided = tl.load(x_ptr + offsets, mask=mask, other=0)
    x_imag_strided = tl.load(x_ptr + offsets + dim * batch_size, mask=mask, other=0)

    s_real_strided = tl.load(s_ptr + offsets, mask=mask, other=0)
    s_imag_strided = tl.load(s_ptr + offsets + dim * batch_size, mask=mask, other=0)

    lambda_real_strided = tl.load(lambda_ptr + offsets, mask=mask, other=0)
    lambda_imag_strided = tl.load(
        lambda_ptr + offsets + dim * batch_size, mask=mask, other=0
    )

    length = tl.load(length_ptr)
    output = tl.load(output_ptr)

    for t in range(T):
        if output:
            s_real_next = (
                s_real_strided * lambda_real_strided - s_imag_strided * lambda_imag_strided
            ) + x_real_strided
            s_imag_next = (
                s_real_strided * lambda_imag_strided + s_imag_strided * lambda_real_strided
            ) + x_imag_strided
        else:
            s_real_next = s_real_strided
            s_imag_next = s_imag_strided
            tl.store(y_ptr + offsets, s_real_strided, mask=mask)
            tl.store(y_ptr + offsets + dim * batch_size, s_imag_strided, mask=mask)
            s_real_strided = (
                s_real_strided * lambda_real_strided - s_imag_strided * lambda_imag_strided
            ) + x_real_strided
            s_imag_strided = (
                s_real_strided * lambda_imag_strided + s_imag_strided * lambda_real_strided
            ) + x_imag_strided
        x_ptr += 2 * dim * batch_size
        x_real_strided = tl.load(x_ptr + offsets, mask=mask, other=0)
        x_imag_strided = tl.load(x_ptr + offsets + dim * batch_size, mask=mask, other=0)
        y_ptr += 2 * dim * batch_size

        if t < length - 1:
            s_ptr += 2 * dim * batch_size
            s_real_strided = tl.load(s_ptr + offsets, mask=mask, other=0)
            s_imag_strided = tl.load(s_ptr + offsets + dim * batch_size, mask=mask, other=0)

@triton.jit
def diag_ssm_backward_kernel(
    x_ptr,
    s_ptr,
    y_ptr,
    lambda_ptr,
    length_ptr,
    grad_y_ptr,
    grad_s_ptr,
    grad_x_ptr,
    grad_lambda_ptr,
    batch_size,
    dim,
    T,
    BLOCK_SIZE: tl.constexpr,
):
    # Kernel function for backward pass of SSM
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    arange = tl.arange(0, BLOCK_SIZE)
    offsets = block_start + arange
    mask = arange < dim * batch_size
    grad_y_strided = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
    grad_s_strided = tl.load(grad_s_ptr + offsets, mask=mask, other=0)
    grad_lambda_strided = tl.load(grad_lambda_ptr + offsets, mask=mask, other=0)
    length = tl.load(length_ptr)

    for t in range(T - 1, -1, -1):
        grad_s_ptr -= dim * batch_size
        grad_s_strided = tl.load(grad_s_ptr + offsets, mask=mask, other=0)
        grad_lambda_ptr -= dim * batch_size
        grad_lambda_strided = tl.load(grad_lambda_ptr + offsets, mask=mask, other=0)
        grad_s_strided += grad_y_strided
        tl.store(grad_s_ptr + offsets, grad_s_strided, mask=mask)

        if t < length - 1:
            grad_y_ptr -= dim * batch_size
            grad_y_strided = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
            grad_s_ptr -= dim * batch_size
            grad_s_strided = tl.load(grad_s_ptr + offsets, mask=mask, other=0)
            grad_lambda_ptr -= dim * batch_size
            grad_lambda_strided = tl.load(grad_lambda_ptr + offsets, mask=mask, other=0)

        grad_s_strided *= grad_lambda_strided
        tl.store(grad_lambda_ptr + offsets, grad_lambda_strided, mask=mask)

@triton.jit
def diag_ssm_backward_kernel_complex(
    x_ptr
