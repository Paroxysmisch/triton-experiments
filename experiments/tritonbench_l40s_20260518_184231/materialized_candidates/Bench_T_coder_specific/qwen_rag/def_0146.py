import triton
import triton.language as tl

@triton.jit
def elu_linear_kernel(
    x_ptr,        # Pointer to input tensor
    w_ptr,        # Pointer to weight tensor
    b_ptr,        # Pointer to bias tensor (optional)
    output_ptr,   # Pointer to output tensor
    n_elements,   # Number of elements in the input tensor
    alpha,        # Alpha value for ELU
    BLOCK_SIZE: tl.constexpr  # Block size for parallelism
):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(x_ptr + offsets, mask=mask)
    w = tl.load(w_ptr + offsets, mask=mask)
    if b_ptr is not None:
        b = tl.load(b_ptr + offsets, mask=mask)
    else:
        b = 0.0

    linear_output = x * w + b
    elu_output = tl.where(linear_output > 0, linear_output, alpha * (tl.exp(linear_output) - 1))

    tl.store(output_ptr + offsets, elu_output, mask=mask)
