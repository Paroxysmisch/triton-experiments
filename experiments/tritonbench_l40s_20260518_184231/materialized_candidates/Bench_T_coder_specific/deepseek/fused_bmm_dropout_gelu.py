import triton
import triton.language as tl

@triton.jit
def fused_bmm_dropout_gelu_kernel(
    input1_ptr, input2_ptr, output_ptr,
    B, N, M, P,
    p,
    training,
    inplace,
    out_opt_layout,
    out_opt_dtype,
    **kwargs
):
    # Define the basic indices
    b = tl.program_id(0)
    n = tl.program_id(1)
    m = tl.program_id(2)
    p = tl.program_id(3)

    # Load inputs
    X = tl.load(input1_ptr + b * N * M + n * M + m)
    Y = tl.load(input2_ptr + b * M * P + m * P + p)

    # Perform batch matrix multiplication
    Z = X * Y

    # Apply dropout
    if training:
        Z = tl.dropout(Z, p)

    # Apply GELU activation
    O = tl.gelu(Z)

    # Store output
    if inplace:
        tl.store(input1_ptr + b * N * M + n * M + m, O)
    else:
        tl.store(output_ptr + b * N * P + n * P + p, O)

def fused_bmm_dropout_gelu(
    input1, input2, p=0.5, training=True, inplace=False, approximate='none', *, out=None
):
    # Validate input shapes
    assert input1.shape[0] == input2.shape[0]
    assert input1.shape[2] == input2.shape[1]

    # Compute output shape
    B, N, M = input1.shape
    _, _, P = input2.shape
    output_shape = (B, N, P)

    # Allocate output tensor
    if out is None:
        output = triton.testing.make_tensor(output_shape, dtype=input1.dtype)
    else:
        assert out.shape == output_shape
        output = out

    # Call the Triton kernel
    fused_bmm_dropout_gelu_kernel[B, N, M, P](
        input1.ptr, input2.ptr, output.ptr,
        B, N, M, P,
        p,
        training,
        inplace,
        output.layout,
        output.dtype
    )

    return output
