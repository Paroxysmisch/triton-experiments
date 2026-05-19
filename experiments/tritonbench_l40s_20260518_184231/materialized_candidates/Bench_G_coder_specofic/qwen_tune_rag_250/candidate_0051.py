:
            out = torch.empty([], dtype=dtype, device=inp.device)
        else:
            shape = list(inp.shape)
            for i in range(0, inp.dim()):
                shape[i] = 1
            out = torch.empty(shape, dtype=dtype, device=inp.device)
        with torch.cuda.device(inp.device):
            amax_kernel_1[(mid_size, 1, 1)](
                inp, mid, M, block_size, INT64_INDEX=use_int64_index
            )
            amax_kernel_2[(1, 1, 1)](mid, out, mid_size, block_mid)
        return out
    else:
        return max_dim(inp, dim, keepdim)


# Function to call the kernels for max_dim
def max_dim(inp, dim, keepdim):
    logging.debug("GEMS MAX DIM")

    inp_shape = inp.shape
    dim = dim % inp.ndim
    M = 1
    N = inp.shape[dim]
    for i in range(0, inp.ndim):
        if i == dim:
            continue
        else:
            M *= inp.shape[i]
    inp = inp.contiguous()

    use_int64_index = not can_use_int32_index(inp)
    if not keepdim:
        out_shape = list()
        for i in range(0, inp.ndim):
            if i == dim:
                continue
            else:
                out_shape.append(inp.shape[i])
        out = torch.empty(out_shape, dtype=inp.dtype, device=inp.device)
    else:
        out = torch.empty(inp.shape, dtype=inp.dtype, device=inp.device)

    if M >= 65536:
        BLOCK_SIZE = 1024
    elif M >= 4096:
        BLOCK_SIZE = 2048
    else:
        BLOCK_SIZE = 4096

    grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]), 1, 1)
    with torch.cuda.device(inp.device):
        amax_kernel[grid](
            inp,
            out,
            M,
            N,
            BLOCK_M=BLOCK_SIZE,
            INT64_INDEX=use_int64_index,
        )
    return out
