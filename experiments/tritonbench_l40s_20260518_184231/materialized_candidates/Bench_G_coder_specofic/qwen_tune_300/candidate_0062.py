import torch
import triton
import triton.language as tl

@triton.jit
def embedding_kernel(
    a_ptr,  # shape: [vocab_size, d_model]
    idx_ptr,  # shape: [max_seqlen]
    out_ptr,  # shape: [max_seqlen, d_model]
    seq_len,
    BLOCK_N: tl.constexpr,
    BLOCK_NN: tl.constexpr,
    BLOCK_DMODEL: tl.constexpr,
):
    cur_index = tl.program_id(0)
    cur_batch = tl.program_id(1)

    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_DMODEL)
    cur_batch_seq_len = tl.load(seq_len + cur_batch)

    a_ptrs = a_ptr + cur_index * BLOCK_NN + offs_n[:, None] * BLOCK_N + offs_d[None, :]
    a_mask = (cur_batch_seq_len > offs_n)[:, None]
    a = tl.load(a_ptrs, mask=a_mask, other=0.0)

    idx = tl.load(idx_ptr + cur_batch * seq_len + cur_index * BLOCK_N + offs_n)
    idx_ptrs = out_ptr + cur_batch * seq_len * BLOCK_DMODEL + idx * BLOCK_DMODEL + offs_d
    tl.store(idx_ptrs, a, mask=(cur_batch_seq_len > offs_n))


def embedding(weight: torch.Tensor, idx: torch.Tensor, max_seqlen: int):
    assert idx.ndim == 1
    assert weight.ndim == 2

    seq_len = torch.full((idx.shape[0],), fill_value=idx.shape[0], dtype=torch.int32, device=idx.device)
    out = torch.empty((idx.shape[0], max_seqlen, weight.shape[1]), dtype=weight.dtype, device=weight.device)

    BLOCK_NN = triton.next_power_of_2(weight.shape[0])
    grid = (triton.cdiv(idx.shape[0], 32), idx.shape[1])

    embedding_kernel[grid](
        weight,
        idx,
        out,
        seq_len,
        BLOCK_N=32,
        BLOCK_NN=BLOCK_NN,
        BLOCK_DMODEL=weight.shape[1],
        num_warps=4,
        num_stages=1,
    )
    return out
