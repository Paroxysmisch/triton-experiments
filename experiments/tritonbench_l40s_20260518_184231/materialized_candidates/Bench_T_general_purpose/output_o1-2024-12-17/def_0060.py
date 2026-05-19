import triton
import triton.language as tl

@triton.jit
def _exp_sqrt_kernel(in_ptr, out_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = idx < n_elements
    # Load
    x = tl.load(in_ptr + idx, mask=mask)
    # exp + sqrt
    y = tl.exp(x)
    y = tl.sqrt(y)
    # Store
    tl.store(out_ptr + idx, y, mask=mask)

def exp_sqrt(input, out=None):
    """
    def exp_sqrt(input, out=None) -> Tensor
        input (Tensor): The input tensor.
        out (Tensor, optional): The output tensor.
    
    Computes the exponential of each element in the input tensor, followed by
    calculating the square root of the result. Returns a tensor where each
    element is the result of applying exponential followed by square root to
    each element of input.
    
    This function will return NaN for input elements that result in negative
    values after `exp` and `sqrt` due to overflow.

    Math:
        out_i = sqrt(exp(input_i))
    """
    n_elements = input.numel()
    if out is None:
        out = input.empty_like()
    BLOCK_SIZE = 1024
    grid = ((n_elements + BLOCK_SIZE - 1) // BLOCK_SIZE,)
    _exp_sqrt_kernel[grid](input, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out
