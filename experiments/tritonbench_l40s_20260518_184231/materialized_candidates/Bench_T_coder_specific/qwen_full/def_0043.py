import torch
import triton
import triton.language as tl

@triton.jit
def _symmetric_matrix_vector_norm_kernel(A, x, y, alpha, beta, p, n: tl.constexpr):
    idx = tl.arange(0, n)
    x_ptr = x + idx
    A_x = tl.mv(A, x)
    y = alpha * A_x + beta * y
    return y

def test_symmetric_matrix_vector_norm_wrapper(func_inputs):
    A, x, alpha, beta, p = func_inputs
    wrapper_inputs = (A, x, alpha, beta, p)
    wrapper_output = symmetric_matrix_vector_norm(*wrapper_inputs)
    torch_output = torch.norm(alpha * torch.mv(A, x) + beta * x, p)
    return torch.allclose(wrapper_output, torch_output)
