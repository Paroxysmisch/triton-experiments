import triton
import triton.language as tl
import torch

@triton.jit
def argmax_kernel_1(
    inp,
    mid_value,
    mid_index,
    M,
    BLOCK_SIZE: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    pid = tl.program_id(0)
    if INT64_INDEX:
        pid = pid.to(tl.int64)
    offset = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    inp_ptrs = inp + offset
    mask = offset < M
    inp_val = tl.load(inp_ptrs, mask=mask, other=float("-inf"))
    index = tl.argmax(inp_val)
    max_val = tl.max(inp_val)
    mid_value_ptr = mid_value + pid
    mid_index_ptr = mid_index + pid
    tl.store(mid_value_ptr, max_val)
    tl.store(mid_index_ptr, index)

@triton.jit
def argmax_kernel_2(mid_value, mid_index, out, mid_size, BLOCK_MID: tl.constexpr):
    offset = tl.arange(0, BLOCK_MID)
    mid_val_ptrs = mid_value + offset
    mid_index_ptrs = mid_index + offset
    mask = offset < mid_size
    mid_val = tl.load(mid_val_ptrs, mask=mask, other=float("-inf"))
    mid_index = tl.load(mid_index_ptrs, mask=mask, other=-1)
    final_index = tl.max(mid_val)
    argmax_index = tl.where(mid_val == final_index, mid_index, -1)
    final_index_ptr = out
    tl.store(final_index_ptr, argmax_index)

@triton.autotune(configs=argmax_configs(), key=["M", "N", "K"])
@triton.jit
def argmax_kernel(
    inp,
    out,
    M,
    N,
    K,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STRIDED: tl.constexpr,
    dim: tl.constexpr,
    INT64_INDEX: tl.constexpr = False,
):
    # Map the program id to the row of inp it should compute.
    if dim == 0:
        # Not implemented for dim=0.
        assert (False)
    elif dim == 1:
        pid = tl.program_id(0)
        if INT64_INDEX:
            pid = pid.to(tl.int64)
        rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)[:, None]
        inp = inp + rows * N
        out = out + rows
        row_mask = rows < M

        _max = tl.full([BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32)
        _index = tl.full([BLOCK_M, BLOCK_N], value=-1, dtype=tl.int32)
        for off in range(0, N, BLOCK_N):
            cols = off + tl.arange(0, BLOCK_N)[None, :]
            col_mask = cols < N
            if STRIDED:
                mask = row_mask and col_mask
                z = tl.load(inp + stride_add(cols, (0, K)), mask).to(tl.float32)
                ind = tl.load(
                    inp + stride_add(cols, (0, K)), mask
                ).to(tl.int32)
                c = cols + K
            else:
                mask = row_mask and col_mask
                a = tl.load(inp + cols, mask).to(tl.float32)
                b = tl.load(inp + cols, mask).to(tl.int32)
                z = a
                ind = b
                c = cols
            any_ = tl.where(_max < z, z, float("-inf"))
            new_max = tl.maximum(_max, any_)
            new_index = tl.where(_max < z, ind, -1)
            new_max = tl.where(new_max == _max, float("-inf"), new_max)
            new_index = tl.where(new_index == -1 and new_max != float("-inf"), _index, new_index)
            _max = new_max
            _index = new_index
        max_ = tl.max(_max, axis=1)[:, None]
        index_ = tl.max(_index, axis=1)[:, None]
        max_ = tl.where(max_ == float("-inf"), 0, max_)
        tl.store(out, index_, row_mask)
    elif dim == 2:
        pid = tl.program_id(0)
        if INT64_INDEX:
            pid = pid.to(tl.int64)
        cols = pid * BLOCK_N + tl.arange(0, BLOCK_N)[None, :]
        inp = inp + (K * M + cols)
        out = out + cols
        col_mask = cols < N

        _max = tl.full([BLOCK_M, BLOCK_N], value=float("-inf"), dtype=tl.float32)
        _index = tl.full([BLOCK_M, BLOCK_N], value=-1, dtype=tl.int32)
        for off in range(0, K, BLOCK_M):
            rows = off + tl.arange(0, BLOCK_M)[:, None]
            row_mask = rows < K
            mask = row_mask and col_mask
            masked_inp_val = tl.load(inp + stride_add(rows, (M, 0)), mask).to(tl.float32)
            masked_index = tl.load(inp + stride_add(rows, (M, 0)), mask).to(tl.int32)
            any_ = tl.where(_max < masked_inp_val, masked_inp_val, float("-inf"))
            new_max = tl.maximum(_max, any_)
            new_index = tl.where(_max < masked_inp_val, masked_index, -1)
            new_max = tl.where(new_max == _max, float("-inf"), new_max)
            new_index = tl.where(new_index == -1 and new_max != float("-inf"), _index, new_index)
            _max = new_max
            _index = _index
        max_ = tl.max(_max, axis=0)[None, :]
        index_ = tl.max(_index, axis=0)[None, :]
        mask = col_mask
        max_ = tl.where(mask, max_, 0)
        index_ = tl.where(mask, index_, -1)
        tl.store(out, index_, mask)

def argmax(inp, dim: int = None, keepdim: bool = False):
    logging.debug("GEMS ARGMAX")
    if dim is None:
        M = inp.numel()
        block_size = triton.next_power_of_2(int(math.sqrt(M)))
        mid_size = triton.cdiv(M, block_size)
        block_mid = triton.next_power_of_2(mid_size)
        dtype = inp.dtype

        mid_value = torch.empty((mid_size,), dtype=dtype, device=inp.device)
        mid_index = torch.empty((mid_size,), dtype=torch.int32, device=inp
