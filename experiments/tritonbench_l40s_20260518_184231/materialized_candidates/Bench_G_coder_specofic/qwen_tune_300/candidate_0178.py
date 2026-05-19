import torch
import triton
import triton.language as tl

@triton.jit
def diag_ssm_forward_kernel(
    s_ptr, x_ptr, y_ptr, Lambda_ptr,
    length: tl.constexpr, batch_size: tl.constexpr, dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Triton kernel for SSM forward pass, real-valued.
    """
    # position of elements processed by this program
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1)

    # pointers to elements processed by this program
    s_ptr = s_ptr + batch_idx * dim + dim_idx
    x_ptr = x_ptr + (batch_idx * length + length - 1) * dim + dim_idx
    y_ptr = y_ptr + (batch_idx * length + length - 1) * dim + dim_idx
    Lambda_ptr = Lambda_ptr + dim_idx

    # temporary variables
    s = tl.load(s_ptr)
    Lambda = tl.load(Lambda_ptr)
    x = tl.load(x_ptr)

    # computation
    s = s * Lambda + x
    tl.store(y_ptr, s)

def _get_grid_real(batch_size, dim):
    return (batch_size * dim, )

@triton.jit
def diag_ssm_forward_kernel_complex(
    s_ptr, x_ptr, y_ptr, Lambda_ptr,
    length: tl.constexpr, batch_size: tl.constexpr, dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Triton kernel for SSM forward pass, complex-valued.
    """
    # position of elements processed by this program
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1)

    # pointers to elements processed by this program
    s_ptr = s_ptr + batch_idx * 2 * dim + 2 * dim_idx
    x_ptr = x_ptr + (batch_idx * length + length - 1) * 2 * dim + 2 * dim_idx
    y_ptr = y_ptr + (batch_idx * length + length - 1) * 2 * dim + 2 * dim_idx
    Lambda_ptr = Lambda_ptr + 2 * dim_idx

    # temporary variables
    s_real = tl.load(s_ptr)
    s_imag = tl.load(s_ptr + 1)
    Lambda_real = tl.load(Lambda_ptr)
    Lambda_imag = tl.load(Lambda_ptr + 1)
    x_real = tl.load(x_ptr)
    x_imag = tl.load(x_ptr + 1)

    # computation
    s_real = s_real * Lambda_real - s_imag * Lambda_imag + x_real
    s_imag = s_real * Lambda_imag + s_imag * Lambda_real + x_imag
    tl.store(y_ptr, s_real)
    tl.store(y_ptr + 1, s_imag)

def _get_grid_complex(batch_size, dim):
    return (batch_size * dim, )

class _ssm_forward(torch.autograd.Function):

    @staticmethod
    def forward(ctx, init_state, inputs, trans_mat, trans_vec=None):
        """
        init_state: (batch_size, dim)
        inputs: (batch_size, seq_len, dim)
        trans_mat: (dim, )
        trans_vec: (dim, )
        """
        batch_size, seq_len, dim = inputs.shape
        assert init_state.shape == (batch_size, dim)
        assert trans_mat.shape == (dim, )
        if trans_vec is not None:
            assert trans_vec.shape == (dim, )

        outputs = torch.empty_like(init_state)

        if trans_vec is None:
            # real-valued
            grid = _get_grid_real(batch_size, dim)
            diag_ssm_forward_kernel[grid](
                init_state, inputs, outputs, trans_mat,
                seq_len, batch_size, dim,
                trans_vec=trans_vec,
                BLOCK_SIZE=min(65536 // 2, triton.next_power_of_2(dim)),
                )
        else:
            # complex-valued
            grid = _get_grid_complex(batch_size, dim)
            diag_ssm_forward_kernel_complex[grid](
                init_state, inputs, outputs, trans_mat, trans_vec,
                seq_len, batch_size, dim,
                BLOCK_SIZE=min(65536 // 4, triton.next_power_of_2(dim)),
                )

        outputs[:, -1] = init_state[:, -1]
        ctx.save_for_backward(init_state, trans_mat, trans_vec)
        return outputs

def ssm_forward(init_state, inputs, trans_mat, trans_vec=None):
    return _ssm_forward.apply(init_state, inputs, trans_mat, trans_vec)
@triton.jit
def diag_ssm_backward_kernel(
    grad_y_ptr, s_ptr, x_ptr, Lambda_ptr, grad_s_ptr, grad_x_ptr, grad_Lambda_ptr,
    length: tl.constexpr, batch_size: tl.constexpr, dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Triton kernel for SSM backward pass, real-valued.
    """
    # position of elements processed by this program
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1)

    # pointers to elements processed by this program
    grad_y_ptr = grad_y_ptr + (batch_idx * length + length - 1) * dim + dim_idx
    s_ptr = s_ptr + batch_idx * dim + dim_idx
    x_ptr = x_ptr + (batch_idx * length + length - 1) * dim + dim_idx
    Lambda_ptr = Lambda_ptr + dim_idx
    grad_s_ptr = grad_s_ptr + batch_idx * dim + dim_idx
    grad_x_ptr = grad_x_ptr + (batch_idx * length + length - 1) * dim + dim_idx
    grad_Lambda_ptr = grad_Lambda_ptr + batch_idx * dim + dim_idx

    # temporary variables
    grad_y = tl.load(grad_y_ptr)
    s = tl.load(s_ptr)
    Lambda = tl.load(Lambda_ptr)
    x = tl.load(x_ptr)

    # computation
    grad_s = grad_y
    grad_x = grad_s
    grad_Lambda = grad_s * s
    tl.store(grad_s_ptr, grad_s)
    tl.store(grad_x_ptr, grad_x)
    tl.store(grad_Lambda_ptr, grad_Lambda)

@triton.jit
def diag_ssm_backward_kernel_complex(
    grad_y_ptr, s_ptr, x_ptr, Lambda_ptr, trans_vec_ptr,
    grad_s_ptr, grad_x_ptr, grad_Lambda_ptr, grad_trans_vec_ptr,
    length: tl.constexpr, batch_size: tl.constexpr, dim: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
    ):
    """
    Triton kernel for SSM backward pass, complex-valued.
    """
    # position of elements processed by this program
    batch_idx = tl.program_id(0)
    dim_idx = tl.program_id(1)

    # pointers to elements processed by this program
    grad_y_ptr = grad_y_ptr + (batch_idx * length + length - 1) * 2 * dim + 2 * dim_idx
    s_ptr = s_ptr + batch_idx * 2 * dim + 2 * dim_idx
    x_ptr = x_ptr + (batch_idx * length + length - 1) * 2 * dim + 2 * dim_idx
    Lambda_ptr = Lambda_ptr + 2 * dim_idx
    trans_vec_ptr = trans_vec_ptr + dim_idx * 2
    grad_s_ptr = grad_s_ptr + batch_idx * 2 * dim + 2 * dim_idx
    grad_x_ptr = grad_x_ptr + (batch_idx * length + length - 1) * 2 * dim + 2 * dim_idx
    grad_Lambda_ptr = grad_Lambda_ptr + batch_idx * 2 * dim + 2 * dim_idx
    grad_trans_vec_ptr = grad_trans_vec_ptr + batch_idx * 2 * dim + 2 * dim_idx

    # temporary variables
    grad_y_real = tl.load(grad_y_ptr)
    grad_y_imag = tl.load(grad_y_ptr + 1)
    s_real = tl.load(s_ptr)
    s_imag = tl.load(s_ptr + 1)
    Lambda_real = tl.load(Lambda_ptr)
    Lambda_imag = tl.load(Lambda_ptr + 1)
    x_real = tl.load(x_ptr)
    x_imag = tl.load(x_ptr + 1)
    trans_vec_real = tl.load(trans_vec_ptr)
    trans_vec_imag = tl.load(trans_vec_ptr + 1)

    # computation
    grad_y_conj = grad_y_real - 1j * grad_y_imag
    s_conj = s_real - 1j * s_imag
    Lambda_conj = Lambda_real - 1j * Lambda_imag
    x_conj = x_real - 1j * x_imag
    trans_vec_conj = trans_vec_real - 1j * trans_vec_imag

    grad_s = grad_y_conj
    grad_x = grad_s * trans_vec_conj
    grad_trans_vec = grad_s * x_conj
    grad_Lambda = grad_s * s
    grad_s_real = tl.real(grad_s)
    grad_s_imag = tl.imag(grad_s)
    tl.store(grad_s_ptr, grad_s_real)
    tl.store(grad_s_ptr + 1, grad_s_imag)
    tl.store(grad_x_ptr, grad_x)
    tl.store(grad_Lambda_ptr, grad_Lambda)
    tl.store(grad_trans_vec_ptr, grad_trans_vec)

class _ssm_backward
