import triton
import triton.language as tl

@triton.jit
def diag_ssm_forward_kernel(
    s_ptr, x_ptr, y_ptr, Lambda_ptr,
    batch_size, length, dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * dim

    s = tl.load(s_ptr + offsets, mask=mask)
    Lambda = tl.load(Lambda_ptr + offsets, mask=mask)

    for t in range(length):
        x = tl.load(x_ptr + t * batch_size * dim + offsets, mask=mask)
        s = s * Lambda + x
        tl.store(y_ptr + t * batch_size * dim + offsets, s, mask=mask)

    tl.store(s_ptr + offsets, s, mask=mask)

@triton.jit
def diag_ssm_forward_kernel_complex(
    s_ptr, x_ptr, y_ptr, Lambda_ptr,
    batch_size, length, dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * dim

    s_real = tl.load(s_ptr + offsets, mask=mask)
    s_imag = tl.load(s_ptr + offsets + batch_size * dim, mask=mask)
    Lambda_real = tl.load(Lambda_ptr + offsets, mask=mask)
    Lambda_imag = tl.load(Lambda_ptr + offsets + batch_size * dim, mask=mask)

    for t in range(length):
        x_real = tl.load(x_ptr + t * batch_size * dim + offsets, mask=mask)
        x_imag = tl.load(x_ptr + t * batch_size * dim + offsets + batch_size * dim, mask=mask)

        s_real, s_imag = (
            s_real * Lambda_real - s_imag * Lambda_imag + x_real,
            s_real * Lambda_imag + s_imag * Lambda_real + x_imag
        )

        tl.store(y_ptr + t * batch_size * dim + offsets, s_real, mask=mask)
        tl.store(y_ptr + t * batch_size * dim + offsets + batch_size * dim, s_imag, mask=mask)

    tl.store(s_ptr + offsets, s_real, mask=mask)
    tl.store(s_ptr + offsets + batch_size * dim, s_imag, mask=mask)

@triton.jit
def diag_ssm_backward_kernel(
    grad_y_ptr, grad_s_ptr, grad_x_ptr, grad_lambda_ptr, s_ptr, x_ptr, Lambda_ptr,
    batch_size, length, dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * dim

    grad_s = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    Lambda = tl.load(Lambda_ptr + offsets, mask=mask)

    for t in range(length - 1, -1, -1):
        grad_y = tl.load(grad_y_ptr + t * batch_size * dim + offsets, mask=mask)
        x = tl.load(x_ptr + t * batch_size * dim + offsets, mask=mask)
        s = tl.load(s_ptr + t * batch_size * dim + offsets, mask=mask)

        grad_s = grad_s * Lambda + grad_y
        grad_x = grad_s
        grad_lambda = grad_s * s

        tl.store(grad_x_ptr + t * batch_size * dim + offsets, grad_x, mask=mask)
        tl.store(grad_lambda_ptr + offsets, grad_lambda, mask=mask)

    tl.store(grad_s_ptr + offsets, grad_s, mask=mask)

@triton.jit
def diag_ssm_backward_kernel_complex(
    grad_y_ptr, grad_s_ptr, grad_x_ptr, grad_lambda_ptr, s_ptr, x_ptr, Lambda_ptr,
    batch_size, length, dim,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < batch_size * dim

    grad_s_real = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    grad_s_imag = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    Lambda_real = tl.load(Lambda_ptr + offsets, mask=mask)
    Lambda_imag = tl.load(Lambda_ptr + offsets + batch_size * dim, mask=mask)

    for t in range(length - 1, -1, -1):
        grad_y_real = tl.load(grad_y_ptr + t * batch_size * dim + offsets, mask=mask)
        grad_y_imag = tl.load(grad_y_ptr + t * batch_size * dim + offsets + batch_size * dim, mask=mask)
        x_real = tl.load(x_ptr + t * batch_size * dim + offsets, mask=mask)
        x_imag = tl.load(x_ptr + t * batch_size * dim + offsets + batch_size * dim, mask=mask)
        s_real = tl.load(s_ptr + t * batch_size * dim + offsets, mask=mask)
        s_imag = tl.load(s_ptr + t * batch_size * dim + offsets + batch_size * dim, mask=mask)

        grad_s_real, grad_s_imag = (
            grad_s_real * Lambda_real - grad_s_imag * Lambda_imag + grad_y_real,
            grad_s_real * Lambda_imag + grad_s_imag * Lambda_real + grad_y_imag
        )

        grad_x_real, grad_x_imag = grad_s_real, grad_s_imag
        grad_lambda_real, grad_lambda_imag = (
            grad_s_real * s_real - grad_s_imag * s_imag,
            grad_s_real * s_imag + grad_s_imag * s_real
        )

        tl.store(grad_x_ptr + t * batch_size * dim + offsets, grad_x_real, mask=mask)
        tl.store(grad_x_ptr + t * batch_size * dim + offsets + batch_size * dim, grad_x_imag, mask=mask)
        tl.store(grad_lambda_ptr + offsets, grad_lambda_real, mask=mask)
        tl.store(grad_lambda_ptr + offsets + batch_size * dim, grad_lambda_imag, mask=mask)

    tl.store(grad_s_ptr + offsets, grad_s_real, mask=mask)
    tl.store(grad_s_ptr + offsets + batch_size * dim, grad_s_imag, mask=mask)
