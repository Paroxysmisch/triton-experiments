1)[:, None]
    tl.store(out, all, row_mask)


def can_use_int32_index(inp):
    return inp.numel() < (1 << 32)


def max(inp, dim=None, keepdim=False):
    if dim is None or keepdim:
        shape = inp.shape
        inp = inp.reshape(-1)
        mid_size = triton.cdiv(inp.numel(), TRITON_MAX_NUMEL)
        mid = torch.empty((mid_size,), dtype=inp.dtype, device=inp.device)
        BLOCK_MID = triton.cdiv(mid_size, TRITON_MAX_NUMEL_2)
        mid2 = torch.empty((BLOCK_MID,), dtype=mid.dtype, device=mid.device) if mid_size >= TRITON_MAX_NUMEL_2 else None
        amax_kernel_1[(mid_size, 1, 1)](inp, mid, inp.numel(), BLOCK_SIZE=TRITON_MAX_NUMEL, INT64_INDEX=not can_use_int32_index(inp))
        if mid2 is not None:
            amax_kernel_2[(BLOCK_MID, 1, 1)](mid, mid2, mid_size, BLOCK_MID=TRITON_MAX_NUMEL_2)
            mid = mid2
            mid_size = BLOCK_MID
        out = torch.empty((1,), dtype=mid.dtype, device=mid.device)
        amax_kernel_2[(1, 1, 1)](mid, out, mid_size, BLOCK_MID=TRITON_MAX_NUMEL_2)
        return out.reshape(shape) if keepdim else out
    else:
        shape = inp.shape
        dim = dim % inp.ndim
        N = shape[dim]
        M = math.prod(shape[:dim])
        inp = inp.contiguous()
        out = torch.empty((M,), dtype=inp.dtype, device=inp.device)
        new_shape = list(shape)
        new_shape[dim] = 1
        mid = torch.empty(new_shape, dtype=inp.dtype, device=inp.device)
        K = triton.cdiv(N, TRITON_MAX_NUMEL)
        BLOCK_K = triton.cdiv(K, TRITON_MAX_NUMEL_2) if K >= TRITON_MAX_NUMEL_2 else 1
        grid = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]) * BLOCK_K,)
        amax_kernel[grid](inp, mid, M, N, BLOCK_M=TRITON_MAX_NUMEL, BLOCK_N=TRITON_MAX_NUMEL, INT64_INDEX=not can_use_int32_index(inp))
        if BLOCK_K > 1:
            mid2 = torch.empty((triton.cdiv(M, TRITON_MAX_NUMEL) * BLOCK_K,), dtype=mid.dtype, device=mid.device)
            grid_2 = lambda meta: (triton.cdiv(triton.cdiv(M, TRITON_MAX_NUMEL), meta["BLOCK_M"]) * BLOCK_K,)
            amax_kernel_2[grid_2](mid, mid2, triton.cdiv(M, TRITON_MAX_NUMEL), BLOCK_MID=TRITON_MAX_NUMEL_2)
            mid = mid2
        grid_3 = lambda meta: (triton.cdiv(M, meta["BLOCK_M"]),)
        amax_kernel_2[grid_3](mid, out, mid.numel(), BLOCK_MID=TRITON_MAX_NUMEL_2)
        return out.reshape(shape) if keepdim else out
