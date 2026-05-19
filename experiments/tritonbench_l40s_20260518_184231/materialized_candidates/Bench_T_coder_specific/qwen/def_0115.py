import triton
import triton.language as tl

@triton.jit
def tanh_kernel(
    input_ptr,
    weight_ptr,
    bias_ptr,
    output_ptr,
    input_rows: int,
    input_cols: int,
    out_rows: int,
    out_cols: int,
    block_size_x: int = 32,
    block_size_y: int = 8
):
    pid = triton.program_id(0)
    grid = triton.cdiv(out_rows * out_cols, block_size_x * block_size_y)

    if pid >= grid:
        return

    row = pid // block_size_y
    col = pid % block_size_y
    i = row * block_size_y + col

    # Load input
    x = input_ptr[i]

    # Linear transformation
    y = 0.0
    for j in range(input_cols):
        w = weight_ptr[j]
        y += x * w

    # Add bias if present
    if bias_ptr is not None:
        y += bias_ptr

    # Apply Tanh activation
    e_x = triton.math.exp(y)
    e_neg_x = triton.math.exp(-y)
    out = (e_x - e_neg_x) / (e_x + e_neg_x)

    # Store output
    output_ptr[i] = out

def tanh_linear(input, weight, bias=None) -> Tensor:
    assert input.shape[-1] == weight.shape[1], "Input features must match weight columns"
    if bias is not None:
        assert weight.shape[0] == bias.shape[0], "Output features must match bias size"

    out_shape = list(input.shape[:-1]) + [weight.shape[0]]
    output = tl.zeros(out_shape, dtype=input.dtype)

    N = input.numel()
    num_blocks = triton.cdiv(N, 32)

    tanh_kernel[input.device](
        input.data_ptr(),
        weight.data_ptr(),
        bias.data_ptr() if bias is not None else tl.zeros(1, dtype=weight.dtype).data_ptr(),
        output.data_ptr(),
        input.shape[-1],
        len(input),
        out_shape[-1],
        num_blocks,
        num_warps=4
    )

    return output
