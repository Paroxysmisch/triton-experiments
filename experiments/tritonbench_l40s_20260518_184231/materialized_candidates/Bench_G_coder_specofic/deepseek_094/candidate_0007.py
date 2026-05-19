import triton
import triton.language as tl

@triton.jit
def ff_llama(
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, y_ptr,
    x_shape, w1_shape, w3_shape, rms_w_shape, y_shape,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    eps,
    n_elements,
    out_type,
    use_fp8
):
    # Define the grid
    grid = lambda meta: (
        tl.serial_tensor_product(meta.serial_chunks, BLOCK_SIZE_M),
        tl.serial_tensor_product(meta.serial_chunks, BLOCK_SIZE_N),
        tl.serial_tensor_product(meta.serial_chunks, BLOCK_SIZE_K),
    )

    # Define the program
    def program(x_ptr, w1_ptr, w3_ptr, rms_w_ptr, y_ptr,
                x_shape, w1_shape, w3_shape, rms_w_shape, y_shape,
                eps,
                n_elements,
                out_type,
                use_fp8):
        # Load inputs
        x = tl.load(x_ptr, out_type)
        w1 = tl.load(w1_ptr, out_type)
        w3 = tl.load(w3_ptr, out_type)
        rms_w = tl.load(rms_w_ptr, out_type)

        # Compute the accumulated sums
        acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K), out_type)
        acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K), out_type)

        # Compute the matrix multiplications
        for i in range(x_shape[0]):
            for j in range(x_shape[1]):
                for k in range(x_shape[2]):
                    acc1 += tl.dot(x[i, j, k], w1[i, j, k])
                    acc2 += tl.dot(x[i, j, k], w3[i, j, k])

        # Apply RMS scaling
        acc1 = tl.sqrt(acc1 / n_elements + eps) * rms_w
        acc2 = tl.sqrt(acc2 / n_elements + eps) * rms_w

        # Apply the activation function
        y = tl.sigmoid(acc1 * acc2)

        # Store the output
        tl.store(y_ptr, y)

    # Invoke the program
    tl.program_group(program)(
        x_ptr, w1_ptr, w3_ptr, rms_w_ptr, y_ptr,
        x_shape, w1_shape, w3_shape, rms_w_shape, y_shape,
        eps,
        n_elements,
        out_type,
        use_fp8
    )

def kernel_ff(
    x, w1, w3, rms_w, y,
    BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
    eps,
    out_type,
    use_fp8
):
    # Assert correct types and shapes
    assert x.dtype == out_type
    assert w1.dtype == out_type
    assert w3.dtype == out_type
    assert rms_w.dtype == out_type

    # Transpose weight matrices
    w1 = w1.T
    w3 = w3.T

    # Set up grid dimensions
    grid_dims = (x.shape[0], x.shape[1], x.shape[2])

    # Invoke the Triton kernel
    ff_llama[grid_dims, (BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K)](
        x.ctypes.data.ptr, w1.ctypes.data.ptr, w3.ctypes.data.ptr, rms_w.ctypes.data.ptr, y.ctypes.data.ptr,
        x.shape, w1.shape, w3.shape, rms_w.shape, y.shape,
        BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
        eps,
        x.size,
        out_type,
        use_fp8
    )
