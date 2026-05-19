import triton
import triton.language as tl

@triton.jit
def cosine_kernel(input_ptr, output_ptr, N: tl.constexpr):
    """
    Applies the cosine function element-wise to the input tensor.

    Parameters:
    -----------
    input_ptr : tl.tensor
        Pointer to the input tensor in global memory.
    output_ptr : tl.tensor
        Pointer to the output tensor where the cosine values will be stored.
    N : int
        Number of elements in the input tensor.
    """
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = tl.arange(0, BLOCK_SIZE)
    idx = block_start + offsets
    mask = idx < N

    input_val = tl.load(input_ptr + idx, mask=mask)
    cos_val = tl.cos(input_val)
    tl.store(output_ptr + idx, cos_val, mask=mask)
