import triton
import triton.language as tl

@triton.jit
def _cos_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.cos(x)
    tl.store(y_ptr + offsets, y, mask=mask)

def cos(input, *, out=None):
    """
    cos(input, *, out=None) -> Tensor
    Returns a new tensor with the cosine of the elements of the input tensor.
    
    Args:
        input (Tensor): the input tensor
        out (Tensor, optional): the output tensor
    """
    if out is None:
        # Assume input.shape is available and create out with same shape on device
        out = input.clone()  # or appropriate allocation on device
    
    n_elements = input.numel()
    BLOCK_SIZE = 1024
    grid = lambda meta: ((n_elements + meta['BLOCK_SIZE'] - 1) // meta['BLOCK_SIZE'],)
    
    _cos_kernel[grid](input, out, n_elements, BLOCK_SIZE)
    return out
