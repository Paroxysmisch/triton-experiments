import triton.language as tl

@triton.jit
def _swiglu_fwd_kernel(
    X_ptr,
    Y_ptr,
    OUT_ptr,
    nrows: tl.constexpr,
    ncols: tl.constexpr,
    batch_size: tl.constexpr,
):
    # get local ids
    row_id = tl.program_id(axis=1)
    col_id = tl.program_id(axis=0)

    # load X value
    X = tl.load(X_ptr + row_id * ncols + col_id)

    # load Y value
    Y = tl.load(Y_ptr + row_id * ncols + col_id)

    # perform sigmoid activation
    sigmoid_val = tl.sigmoid(X)

    # perform swiglu operation
    swiglu_val = sigmoid_val * Y

    # store result
    tl.store(OUT_ptr + row_id * ncols + col_id, swiglu_val)


def _swiglu_fwd(X, Y, OUT, nrows, ncols, grid):
    # ensure inputs are contiguous
    X = X.contiguous()
    Y = Y.contiguous()

    # prepare inputs for Triton function
    X_ptr = X.data_ptr()
    Y_ptr = Y.data_ptr()
    OUT_ptr = OUT.data_ptr()

    # define grid and launch kernel
    _swiglu_fwd_kernel[grid](
        X_ptr,
        Y_ptr,
        OUT_ptr,
        nrows=nrows,
        ncols=ncols,
        batch_size=1,
    )
