import triton
import triton.language as tl

@triton.jit
def hurwitz_zeta_kernel(
    x_ptr,  # Pointer to the input tensor x
    q_ptr,  # Pointer to the input tensor q
    out_ptr,  # Pointer to the output tensor
    n_terms: tl.int32,  # Number of terms to sum in the series
    stride_x: tl.int32,  # Stride of the input tensor x
    stride_q: tl.int32,  # Stride of the input tensor q
    stride_out: tl.int32,  # Stride of the output tensor
):
    # Get the index of the current block
    pid = tl.program_id(axis=0)
    grid_size = tl.cdiv(tl.numel(x_ptr), tl.block_dim())

    # Each block processes one element from the input tensors
    if pid >= grid_size:
        return

    # Load x and q values
    x = tl.load(x_ptr + pid * stride_x)
    q = tl.load(q_ptr + pid * stride_q)

    # Initialize the result
    result = tl.zeros([], dtype=tl.float32)

    # Compute the series approximation
    k = tl.arange(n_terms)
    term = 1 / (k + q) ** x
    result += term

    # Store the result in the output tensor
    tl.store(out_ptr + pid * stride_out, result)
