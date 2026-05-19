_s, mask=mask)
    tl.store(grad_lambda_ptr + col_offsets,
             grad_Lambda, mask=mask)

@triton.jit
def diag_ssm_forward_kernel_complex(s_ptr, x_ptr, lambda_ptr, y_ptr,
                                    length, batch_size, dim,
                                    BLOCK_SIZE: tl.constexpr):
    """
    Args:
        s_ptr: [batch_size, dim, 2]
        x_ptr: [length, batch_size, dim, 2]
        lambda_ptr: [dim, 2]
        y_ptr: [length, batch_size, dim, 2]
    """
    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    s_real = tl.load(s_ptr + col_offsets * 2, mask=mask, other=0)
    s_imag = tl.load(s_ptr + col_offsets * 2 + 1, mask=mask, other=0)
    lambda_real = tl.load(
        lambda_ptr + (col_offsets % dim) * 2, mask=mask, other=0)
    lambda_imag = tl.load(
        lambda_ptr + (col_offsets % dim) * 2 + 1, mask=mask, other=0)

    for t in range(length):
        offsets = (t * batch_size * dim + col_offsets) * 2
        x_real = tl.load(x_ptr + offsets, mask=mask, other=0)
        x_imag = tl.load(x_ptr + offsets + 1, mask=mask, other=0)

        s_real_new = s_real * lambda_real - s_imag * lambda_imag + x_real
        s_imag_new = s_real * lambda_imag + s_imag * lambda_real + x_imag

        s_real = s_real_new
        s_imag = s_imag_new

        tl.store(y_ptr + offsets, s_real, mask=mask)
        tl.store(y_ptr + offsets + 1, s_imag, mask=mask)

@triton.jit
def diag_ssm_backward_kernel_complex(
        s_ptr, lambda_ptr, y_ptr, grad_s_ptr, grad_x_ptr, grad_lambda_ptr,
        grad_y_ptr, length, batch_size, dim, BLOCK_SIZE: tl.constexpr):
    """
    Args:
        s_ptr: [batch_size, dim, 2]
        lambda_ptr: [dim, 2]
        y_ptr: [length, batch_size, dim, 2]
        grad_s_ptr: [batch_size, dim, 2]
        grad_x_ptr: [length, batch_size, dim, 2]
        grad_lambda_ptr: [batch_size, dim, 2]. The shape is different from ``grad_s_ptr``
            because we need the caller to sum the gradients after the kernel finish.
            It's more complicated to sum the gradients inside the kernel.
        grad_y_ptr: [length, batch_size, dim, 2]
    """

    col_idx = tl.program_id(0) * BLOCK_SIZE
    col_offsets = col_idx + tl.arange(0, BLOCK_SIZE)
    mask = col_offsets < batch_size * dim

    lambda_real = tl.load(
        lambda_ptr + (col_offsets % dim) * 2, mask=mask, other=0)
    lambda_imag = tl.load(
        lambda_ptr + (col_offsets % dim) * 2 + 1, mask=mask, other=0)

    # Initialize gradients to zero
    grad_s_real = tl.zeros_like(lambda_real)
    grad_s_imag = tl.zeros_like(lambda_real)
    grad_lambda_real = tl.zeros_like(lambda_real)
    grad_lambda_imag = tl.zeros_like(lambda_real)

    for i in range(length):
        # range(length - 1, -1, -1) is not correctly implemented by Triton
        t = length - 1 - i
        offsets = (t * batch_size * dim + col_offsets) * 2

        grad_y_real = tl.load(grad_y_ptr + offsets, mask=mask, other=0)
        grad_y_imag = tl.load(grad_y_ptr + offsets + 1, mask=mask, other=0)

        if t > 0:
            s_real = tl.load(
                y_ptr + offsets - 2 * batch_size * dim, mask=mask, other=0)
            s_imag = tl.load(
                y_ptr + offsets - 2 * batch_size * dim + 1,
                mask=mask, other=0)
        else:
            s_real = tl.load(s_ptr + 2 * col_offsets, mask=mask, other=0)
            s_imag = tl.load(
                s_ptr + 2 * col_offsets + 1, mask=mask, other=0)

        grad_s_real = grad_y_real + grad_s_real
        grad_s_imag = grad_y_imag + grad_s_imag
        grad_x_real = grad_s_real
        grad_x_imag = grad_s_imag
        grad_lambda_real += grad_s_real * s_real - grad_s_imag * s_imag
        grad_lambda_imag += grad_s_real * s_imag + grad_s_imag * s_real
        grad_s_real = grad_s_real * lambda_real - grad_s_imag * lambda_imag
        grad_s_imag = grad_s_real * lambda_imag + grad_s_imag * lambda_real

        tl.store(grad_x_ptr + offsets, grad_x_real, mask=mask)
        tl.store(grad_x_ptr + offsets + 1, grad_x_imag, mask=mask)

    tl.store(
        grad_s_ptr + 2 * col_offsets, grad_s_real, mask=mask)
    tl.store(
        grad_s_ptr + 2 * col_offsets + 1, grad_s_imag, mask=mask)
    tl.store(
        grad_lambda_ptr + 2 * col_offsets,
        grad_lambda_real, mask=mask)
    tl.store(
        grad_lambda_ptr + 2 * col_offsets + 1,
        grad_lambda_imag, mask=mask)

class _ssm_forward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, s, x, lambda_):
        """
        Args:
            s: [batch_size, dim]
            x: [length, batch_size, dim]
            lambda: [dim]
        """
        length, batch_size, dim = x.shape
        assert s.shape == (batch_size, dim)
        assert lambda_.shape == (dim, )

        y = torch.empty_like(x)
        if s.is_floating_point():
            diag_ssm_forward_kernel[(batch_size * dim, )](
                s, x, lambda_, y, length, batch_size, dim)
        else:
            assert x.is_floating_point() and lambda_.is_floating_point()
            diag_ssm_forward_kernel_complex[(batch_size * dim, )](
                s, x, lambda_, y, length, batch_size, dim)

        ctx.save_for_backward(s, lambda_, y)
        return y

    @staticmethod
    def backward(ctx, grad_y):
        s, lambda_, y = ctx.saved_tensors
        length, batch_size, dim = grad_y.shape
        assert y.shape == grad_y.shape
        grad_s = torch.empty_like(s)
        grad_x = torch.empty_like(grad_y)
        if s.is_floating_point():
            grad_lambda = torch.empty_like(s)
            diag_ssm_backward_kernel[(batch_size * dim, )](
                s, lambda_, y, grad_s, grad_x, grad_lambda, grad_y,
                length, batch_size, dim)
            grad_lambda_sum = torch.sum(grad_lambda, dim=0)
            ctx.backward_cache.save(grad_s, grad_x, grad_lambda_sum)
        else:
            assert grad_y.is_floating_point() and lambda_.is_floating_point()
            grad_lambda = torch.empty_like(s)
            diag_ssm_backward_kernel_complex[(batch_size * dim, )](
                s, lambda_, y, grad_s, grad_x, grad_lambda, grad_y,
                length, batch_size, dim)
            grad_lambda_sum_real = torch.sum(grad_lambda[:, 0], dim=0)
            grad_lambda_sum_imag = torch.sum(grad_lambda[:, 1], dim=0)
            ctx.backward_cache.save(
                grad_s[:, 0], grad_s[:, 1], grad_lambda_sum_real,
                grad_lambda_sum_imag)

        return grad_s, grad_x, grad_lambda
