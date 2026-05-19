import triton
import triton.language as tl

@triton.jit
def softmax_kernel(
    input_ptr, output_ptr,
    input_row_stride, output_row_stride,
    n_cols,
    BLOCK_SIZE: tl.constexpr):

    row = tl.program_id(axis=0)
    block_start = row * output_row_stride
    offsets = block_start + tl.arange(0, BLOCK_SIZE)

    mask = offsets < n_cols
    x = tl.load(input_ptr + offsets, mask=mask, other=-float('inf'))

    max_val = tl.max(x, axis=0)
    x = tl.exp(x - max_val)

    sum_val = tl.sum(x, axis=0)
    x = x / sum_val

    tl.store(output_ptr + offsets, x, mask=mask)

def triton_softmax(x):
    n_rows, n_cols = x.shape
    output = tl.empty_like(x)

    BLOCK_SIZE = 2**tl.ceil(tl.log2(n_cols))
    BLOCK_SIZE = min(BLOCK_SIZE, 1024)

    grid = (n_rows, )
    softmax_kernel[grid](
        x, output,
        x.stride(0), output.stride(0),
        n_cols,
        BLOCK_SIZE)

    return output
