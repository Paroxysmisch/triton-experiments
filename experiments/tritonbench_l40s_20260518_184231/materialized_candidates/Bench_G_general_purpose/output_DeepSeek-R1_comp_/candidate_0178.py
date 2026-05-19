import triton
import triton.language as tl
import torch

@triton.jit
def diag_ssm_forward_kernel(
    x_ptr, s_ptr, lambda_ptr, y_ptr,
    batch_size, dim, length,
    x_bs_stride, x_d_stride, x_l_stride,
    s_bs_stride, s_d_stride,
    lambda_d_stride,
    y_bs_stride, y_d_stride, y_l_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pairs = batch_size * dim
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_pairs

    batch_idx = offsets // dim
    dim_idx = offsets % dim

    # Load initial state
    s_off = batch_idx * s_bs_stride + dim_idx * s_d_stride
    s = tl.load(s_ptr + s_off, mask=mask)

    # Load lambda
    lambda_ = tl.load(lambda_ptr + dim_idx * lambda_d_stride, mask=mask)

    for t in range(length):
        x_off = batch_idx * x_bs_stride + dim_idx * x_d_stride + t * x_l_stride
        x = tl.load(x_ptr + x_off, mask=mask)
        s = s * lambda_ + x
        y_off = batch_idx * y_bs_stride + dim_idx * y_d_stride + t * y_l_stride
        tl.store(y_ptr + y_off, s, mask=mask)

@triton.jit
def diag_ssm_forward_kernel_complex(
    x_real_ptr, x_imag_ptr,
    s_real_ptr, s_imag_ptr,
    lambda_real_ptr, lambda_imag_ptr,
    y_real_ptr, y_imag_ptr,
    batch_size, dim, length,
    x_bs_stride, x_d_stride, x_l_stride,
    s_bs_stride, s_d_stride,
    lambda_d_stride,
    y_bs_stride, y_d_stride, y_l_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pairs = batch_size * dim
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_pairs

    batch_idx = offsets // dim
    dim_idx = offsets % dim

    # Load initial state
    s_off = batch_idx * s_bs_stride + dim_idx * s_d_stride
    s_real = tl.load(s_real_ptr + s_off, mask=mask)
    s_imag = tl.load(s_imag_ptr + s_off, mask=mask)

    # Load lambda components
    lambda_real = tl.load(lambda_real_ptr + dim_idx * lambda_d_stride, mask=mask)
    lambda_imag = tl.load(lambda_imag_ptr + dim_idx * lambda_d_stride, mask=mask)

    for t in range(length):
        x_off = batch_idx * x_bs_stride + dim_idx * x_d_stride + t * x_l_stride
        x_real = tl.load(x_real_ptr + x_off, mask=mask)
        x_imag = tl.load(x_imag_ptr + x_off, mask=mask)

        # Complex multiplication: s * lambda
        new_real = s_real * lambda_real - s_imag * lambda_imag
        new_imag = s_real * lambda_imag + s_imag * lambda_real

        # Add x
        new_real += x_real
        new_imag += x_imag

        # Store result and update state
        y_off = batch_idx * y_bs_stride + dim_idx * y_d_stride + t * y_l_stride
        tl.store(y_real_ptr + y_off, new_real, mask=mask)
        tl.store(y_imag_ptr + y_off, new_imag, mask=mask)
        s_real, s_imag = new_real, new_imag

@triton.jit
def diag_ssm_backward_kernel(
    grad_y_ptr, y_ptr, s_ptr, lambda_ptr,
    grad_s_ptr, grad_x_ptr, grad_lambda_ptr,
    batch_size, dim, length,
    gy_bs_stride, gy_d_stride, gy_l_stride,
    y_bs_stride, y_d_stride, y_l_stride,
    s_bs_stride, s_d_stride,
    lambda_d_stride,
    gs_bs_stride, gs_d_stride,
    gx_bs_stride, gx_d_stride, gx_l_stride,
    glambda_d_stride,
    BLOCK_SIZE: tl.constexpr,
):
    pid = tl.program_id(0)
    num_pairs = batch_size * dim
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < num_pairs

    batch_idx = offsets // dim
    dim_idx = offsets % dim

    # Load lambda and initial state
    lambda_ = tl.load(lambda_ptr + dim_idx * lambda_d_stride, mask=mask)
    s_initial = tl.load(s_ptr + batch_idx * s_bs_stride + dim_idx * s_d_stride, mask=mask)

    grad_s = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)
    grad_lambda_acc = tl.zeros((BLOCK_SIZE,), dtype=tl.float32)

    for t in range(length-1, -1, -1):
        gy_off = batch_idx * gy_bs_stride + dim_idx * gy_d_stride + t * gy_l_stride
        grad_y = tl.load(grad_y_ptr + gy_off, mask=mask)
        grad_s += grad_y

        # Store grad_x
        gx_off = batch_idx * gx_bs_stride + dim_idx * gx_d_stride + t * gx_l_stride
        tl.store(grad_x_ptr + gx_off, grad_s, mask=mask)

        # Get previous state
        if t == 0:
            s_prev = s_initial
        else:
            y_prev_off = batch_idx * y_bs_stride + dim_idx * y_d_stride + (t-1) * y_l_stride
            s_prev = tl.load(y_ptr + y_prev_off, mask=mask)

        # Accumulate lambda gradient
        grad_lambda_acc += s_prev * grad_s

        # Update grad_s for previous timestep
        grad_s = grad_s * lambda_

    # Store final grad_s and atomically update grad_lambda
    tl.store(grad_s_ptr + batch_idx * gs_bs_stride + dim_idx * gs_d_stride, grad_s, mask=mask)
    tl.atomic_add(grad_lambda_ptr + dim_idx * glambda_d_stride, grad_lambda_acc, mask=mask)

class _ssm_forward(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, s, lambda_):
        # Validate inputs
        assert x.dim() == 3 and s.dim() == 2
        batch_size, dim, length = x.shape
        assert s.shape == (batch_size, dim)
        assert lambda_.shape == (dim,)

        y = torch.empty_like(x)
        
        # Strides calculation
        def get_strides(tensor):
            return (tensor.stride(i) for i in range(tensor.dim()))
        
        # Launch appropriate kernel
        if tensor.is_complex():
            # Complex kernel setup (omitted for brevity)
            pass
        else:
            grid = lambda meta: ((batch_size * dim + meta['BLOCK_SIZE'] -1) // meta['BLOCK_SIZE'],)
            diag_ssm_forward_kernel[grid](
                x, s, lambda_, y,
                batch_size, dim, length,
                *get_strides(x), *get_strides(s), *get_strides(lambda_), *get_strides(y),
                BLOCK_SIZE=128
            )

        ctx.save_for_backward(x, s, lambda_, y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        x, s, lambda_, y = ctx.saved_tensors
        batch_size, dim, length = x.shape

        # Initialize gradients
        grad_s = torch.zeros_like(s)
        grad_x = torch.zeros_like(x)
        grad_lambda = torch.zeros_like(lambda_)

        # Launch backward kernel
        grid = lambda meta: ((batch_size * dim + meta['BLOCK_SIZE'] -1) // meta['BLOCK_SIZE'],)
        diag_ssm_backward_kernel[grid](
            grad_y, y, s, lambda_,
            grad_s, grad_x, grad_lambda,
            batch_size, dim, length,
            *grad_y.stride(), *y.stride(), *s.stride(), *lambda_.stride(),
            *grad_s.stride(), *grad_x.stride(), *grad_lambda.stride(),
            BLOCK_SIZE=128
        )

        return grad_x, grad_s, grad_lambda

# Example usage
def ssm_forward(x, s, lambda_):
    return _ssm_forward.apply(x, s, lambda_)
