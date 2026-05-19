[0]

    max_fused_size = 4096 // a.dtype.itemsize
    BLOCK_SIZE, num_warps = calculate_settings(n_cols, max_fused_size=max_fused_size)

    _swiglu_forward_kernel[(n_rows,)](
        a,
        b,
        c,
        c.strides[-2],
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a, b, c.reshape(ori_shape)

def swiglu_backward(a, b, dc):
    ori_shape = dc.shape
    n_cols = ori_shape[-1]
    dc = dc.reshape([-1, n_cols])
    n_rows = dc.shape[0]

    max_fused_size = 4096 // a.dtype.itemsize
    BLOCK_SIZE, num_warps = calculate_settings(n_cols, max_fused_size=max_fused_size)

    _swiglu_backward_kernel[(n_rows,)](
        dc,
        a,
        b,
        dc.strides[-2],
        n_cols=n_cols,
        BLOCK_SIZE=BLOCK_SIZE,
        num_warps=num_warps,
    )
    return a.reshape(ori_shape), b.reshape(ori_shape)
