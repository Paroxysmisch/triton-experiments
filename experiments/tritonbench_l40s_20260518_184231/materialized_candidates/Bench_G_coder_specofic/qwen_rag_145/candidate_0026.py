import torch
import triton
import triton.language as tl

@triton.jit
def q_kernel_per_block_int8(
        q_ptr, out_ptr, scale_ptr,
        MAX_VAL: tl.constexpr, BLKQ: tl.constexpr, BLKQ2: tl.constexpr, BLKQ3: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset_q = pid * BLKQ * BLKQ3
    block = tl.load(q_ptr + offset_q * tl.stride(BLKQ, BLKQ3), mask=one_mask(BLKQ, BLKQ3))
    scale = tl.max(tl.abs(block)) / MAX_VAL
    block = block / scale
    block = block.round() * scale
    tl.store(out_ptr + offset_q * tl.stride(4, 4), block.to(tl.int8))
    tl.store(scale_ptr + offset_q, scale)

@triton.jit
def k_kernel_per_block_int8(
        k_ptr, out_ptr, scale_ptr,
        MAX_VAL: tl.constexpr, BLKK: tl.constexpr, BLKK2: tl.constexpr, BLKK3: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    offset_k = pid * BLKK * BLKK3
    block = tl.load(k_ptr + offset_k * tl.stride(BLKK, BLKK3), mask=one_mask(BLKK, BLKK3))
    scale = tl.max(tl.abs(block)) / MAX_VAL
    block = block / scale
    block = block.round() * scale
    tl.store(out_ptr + offset_k * tl.stride(4, 4), block.to(tl.int8))
    tl.store(scale_ptr + offset_k, scale)

def per_block_int8(q, k, BLKQ, BLKK, q_int8, k_int8, q_scale, k_scale):
    with torch.cuda.device(q.device):
        q_int8.fill_(0)
        k_int8.fill_(0)
        q_scale.fill_(0)
        k_scale.fill_(0)
        grid = lambda META: ((q.shape[0] - 1) // META['BLKQ'] + 1,)
        q_kernel_per_block_int8[grid](
            q.contiguous().view(-1).data_ptr(),
            q_int8.contiguous().view(-1).data_ptr(),
            q_scale.data_ptr(),
            tl.float32(127), BLKQ, BLKQ // 4,
        )
        grid = lambda META: ((k.shape[0] - 1) // META['BLKK'] + 1,)
        k_kernel_per_block_int8[grid](
            k.contiguous().view(-1).data_ptr(),
            k_int8.contiguous().view(-1).data_ptr(),
            k_scale.data_ptr(),
            tl.float32(127), BLKK, BLKK // 4,
        )
