import triton
import triton.language as tl

@triton.jit
def polygamma_kernel(
    n: tl.int32,
    x_ptr: tl.tensor,
    y_ptr: tl.tensor,
    n_elements: tl.int32,
):
    pid = tl.program_id(axis=0)
    stride = tl.cdiv(n_elements, tl.num_programs())
    offsets = pid * stride + tl.arange(0, stride)
    x = tl.load(x_ptr + offsets)
    
    # Approximation for psi(x)
    psi_x = tl.log(x) - 1 / (2 * x)
    for k in range(1, n + 1):
        psi_x -= 1 / (x + k) + 1 / (x - k)
    
    # Compute the n-th derivative of psi(x)
    if n == 0:
        y = psi_x
    elif n == 1:
        y = 1 / x - 1 / (x**2)
    elif n == 2:
        y = -1 / (x**2) + 2 / (x**3)
    else:
        # Higher-order derivatives can be computed similarly
        y = 0.0
    
    tl.store(y_ptr + offsets, y)

@triton.jit
def polygamma(n: int, input: triton.Tensor, *, out: triton.Tensor = None) -> triton.Tensor:
    if n < 0:
        raise ValueError("n must be a nonnegative integer")
    
    if out is None:
        out = triton.empty_like(input)
    
    n_elements = input.shape[0]
    polygamma_kernel[(n_elements // 256 + 1)](n, input.data_ptr(), out.data_ptr(), n_elements)
    
    return out
