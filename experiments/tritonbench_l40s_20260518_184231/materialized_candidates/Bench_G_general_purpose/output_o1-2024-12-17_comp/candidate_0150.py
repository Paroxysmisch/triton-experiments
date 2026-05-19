import triton
import triton.language as tl
import torch


def can_use_int32_index(numel):
    return numel < 2**31


@triton.jit
def argmax_kernel_1(
    inp_ptr,
    mid_value_ptr,
    mid_index_ptr,
    M,
    BLOCK_SIZE,
    INT64_INDEX,
    **meta
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < M
    # Load input
    values = tl.where(mask, tl.load(inp_ptr + offsets, mask=mask), float('-inf'))
    # Keep track of max value and index
    max_val = values
    max_idx = tl.astype(offsets, tl.int64) if INT64_INDEX else tl.astype(offsets, tl.int32)
    # Perform pairwise reduction
    for offset in [BLOCK_SIZE // 2**i for i in range(1, 9)]:
        cond = tl.arange(0, BLOCK_SIZE) < offset
        left_val = tl.broadcast_to(max_val, [BLOCK_SIZE])
        right_val = tl.broadcast_to(tl.shift(max_val, offset), [BLOCK_SIZE])
        left_idx = tl.broadcast_to(max_idx, [BLOCK_SIZE])
        right_idx = tl.broadcast_to(tl.shift(max_idx, offset), [BLOCK_SIZE])
        better = right_val > left_val
        max_val = tl.where(cond & better, right_val, left_val)
        max_idx = tl.where(cond & better, right_idx, left_idx)
    if tl.thread_id_x() == 0:
        tl.store(mid_value_ptr + pid, max_val[0], mask=True)
        if INT64_INDEX:
            tl.store(mid_index_ptr + pid, max_idx[0], mask=True)
        else:
            tl.store(mid_index_ptr + pid, tl.astype(max_idx[0], tl.int32), mask=True)


@triton.jit
def argmax_kernel_2(
    mid_value_ptr,
    mid_index_ptr,
    out_ptr,
    mid_size,
    BLOCK_MID,
    **meta
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_MID
    offsets = block_start + tl.arange(0, BLOCK_MID)
    mask = offsets < mid_size
    values = tl.where(mask, tl.load(mid_value_ptr + offsets, mask=mask), float('-inf'))
    indices = tl.load(mid_index_ptr + offsets, mask=mask)
    max_val = values
    max_idx = indices
    for offset in [BLOCK_MID // 2**i for i in range(1, 9)]:
        cond = tl.arange(0, BLOCK_MID) < offset
        left_val = tl.broadcast_to(max_val, [BLOCK_MID])
        right_val = tl.broadcast_to(tl.shift(max_val, offset), [BLOCK_MID])
        left_idx = tl.broadcast_to(max_idx, [BLOCK_MID])
        right_idx = tl.broadcast_to(tl.shift(max_idx, offset), [BLOCK_MID])
        better = right_val > left_val
        max_val = tl.where(cond & better, right_val, left_val)
        max_idx = tl.where(cond & better, right_idx, left_idx)
    if tl.thread_id_x() == 0:
        tl.store(out_ptr, max_idx[0], mask=True)


@triton.jit
def argmax_kernel(
    inp_ptr,
    out_ptr,
    M,  # product of dims before 'dim'
    N,  # dimension size to reduce
    K,  # product of dims after 'dim'
    BLOCK_M,
    BLOCK_N,
    INT64_INDEX,
    **meta
):
    pid_m = tl.program_id(0)
    pid_k = tl.program_id(1)
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = pid_k * BLOCK_M + tl.arange(0, BLOCK_M)
    # Flatten index in dimension order: M, N, K
    # We'll handle the K dimension as well if needed
    # For simplicity, we flatten M and K in this example
    mask_m = offs_m < M
    out_offset = pid_m * K + pid_k  # index in output
    max_val = tl.full([BLOCK_M], float('-inf'), tl.float32)
    max_idx = tl.zeros([BLOCK_M], dtype=tl.int64) if INT64_INDEX else tl.zeros([BLOCK_M], dtype=tl.int32)
    for n_block_id in range(0, N, BLOCK_N):
        mask_n = offs_n + n_block_id < N
        cols = n_block_id + offs_n
        idx = (offs_m * N + cols) * K + pid_k if (K == 1) else (offs_m * N + cols) * K + pid_k
        mask = mask_m & mask_n
        vals = tl.where(
            mask,
            tl.load(inp_ptr + idx, mask=mask, other=float('-inf')),
            float('-inf')
        )
        better = vals > max_val
        max_val = tl.where(better, vals, max_val)
        if INT64_INDEX:
            new_index = tl.astype(cols, tl.int64)
            max_idx = tl.where(better, new_index, max_idx)
        else:
            new_index = tl.astype(cols, tl.int32)
            max_idx = tl.where(better, new_index, max_idx)
    if tl.thread_id_x() == 0:
        if INT64_INDEX:
            tl.store(out_ptr + out_offset, max_idx[0], mask=mask_m[0])
        else:
            tl.store(out_ptr + out_offset, tl.astype(max_idx[0], tl.int32), mask=mask_m[0])


def argmax(inp: torch.Tensor, dim=None):
    device = inp.device
    dtype = inp.dtype
    N = inp.numel()
    INT64_INDEX = not can_use_int32_index(N)
    if dim is None:
        BLOCK_SIZE = 1024
        mid_size = (N + BLOCK_SIZE - 1) // BLOCK_SIZE
        mid_value = torch.empty(mid_size, dtype=inp.dtype, device=device)
        if INT64_INDEX:
            mid_index = torch.empty(mid_size, dtype=torch.int64, device=device)
            out = torch.empty([], dtype=torch.int64, device=device)
        else:
            mid_index = torch.empty(mid_size, dtype=torch.int32, device=device)
            out = torch.empty([], dtype=torch.int32, device=device)

        grid = lambda meta: (mid_size,)
        argmax_kernel_1[grid](inp, mid_value, mid_index, N, BLOCK_SIZE, INT64_INDEX)
        BLOCK_MID = 1024
        final_size = (mid_size + BLOCK_MID - 1) // BLOCK_MID
        grid2 = lambda meta: (final_size,)
        argmax_kernel_2[grid2](mid_value, mid_index, out, mid_size, BLOCK_MID)
        return out.item()

    # Reduction along specified dimension
    shape = list(inp.shape)
    dim_size = shape[dim]
    M = 1
    for i in range(dim):
        M *= shape[i]
    K = 1
    for i in range(dim + 1, len(shape)):
        K *= shape[i]
    BLOCK_M = 64
    BLOCK_N = 64
    out_shape = shape[:dim] + shape[dim+1:]
    if INT64_INDEX:
        out = torch.empty(out_shape, dtype=torch.int64, device=device)
    else:
        out = torch.empty(out_shape, dtype=torch.int32, device=device)

    grid = (
        ( (M + BLOCK_M - 1) // BLOCK_M, (K + BLOCK_M - 1) // BLOCK_M ),
    )
    argmax_kernel[grid](
        inp, out, M, dim_size, K, BLOCK_M, BLOCK_N, INT64_INDEX
    )
    return out
