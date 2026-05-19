import torch
import triton
import triton.language as tl
from torch import Tensor

@triton.jit
def embedding_kernel(
    weight,
    token_ids,
    out,
    stride_z,
    stride_x,
    num_tokens_per_seq,
    num_rows,
    BLOCK_N: tl.constexpr,
    BLOCK_NN: tl.constexpr,
    DMODEL: tl.constexpr,
):
    seq_num = tl.program_id(0)
    i_nn = tl.program_id(1)
    i_m = tl.arange(0, BLOCK_N)
    i_n = i_nn * BLOCK_NN + tl.arange(0, BLOCK_NN)
    token_id_offsets = seq_num * num_tokens_per_seq + i_m
    token_ids_mask = token_id_offsets < num_tokens_per_seq
    token_ids_2d = token_ids + token_id_offsets * stride_x
    weight_ptrs = weight + i_n[None, :] * stride_z + i_m[:, None] * stride_x
    tl.static_print(f"{weight_ptrs=}")
    tl.static_print(f"{token_ids_2d=}")
    tl.static_print(f"{token_ids_mask=}")
    z = tl.load(
        weight_ptrs,
        mask=i_n[None, :] < num_rows,
        other=0.0,
    )
    token_ids_3d = tl.load(
        token_ids_2d,
        mask=token_ids_mask[:, None],
        other=0.0,
    ).to(tl.int32)
    out_ptrs = out + token_id_offsets[:, None] * stride_z + i_n[None, :] * stride_x
    tl.store(
        out_ptrs,
        z,
        mask=token_ids_mask[:, None] & (i_n[None, :] < DMODEL),
    )


class Embedding(torch.autograd.Function):
    @staticmethod
    def forward(ctx, weight, token_ids, padding_idx, num_tokens_per_seq, out):
        if padding_idx is not None:
            # If padding_idx is specified, we replace token_ids == padding_idx with 0
            padding_mask = token_ids == padding_idx
            token_ids = torch.where(padding_mask, 0, token_ids)
        num_seqs = token_ids.shape[0]
        num_rows, dim = weight.shape
        BLOCK_N = triton.next_power_of_2(num_tokens_per_seq)
        BLOCK_NN = triton.next_power_of_2(dim)
        DMODEL = triton.next_power_of_2(dim)
        grid = (num_seqs, triton.cdiv(dim, BLOCK_NN))
        with torch.cuda.device(weight.device.index):
            embedding_kernel[grid](
                weight,
                token_ids,
                out,
                out.stride(0),
                out.stride(1),
                num_tokens_per_seq,
                num_rows,
                BLOCK_N=BLOCK_N,
                BLOCK_NN=BLOCK_NN,
                DMODEL=DMODEL,
                num_warps=4,
                num_stages=2,
                constant_cache={
                    "LOAD_WEIGHT_CACHE_SIZE": 16 * 1024,
                    "STORE_WEIGHT_CACHE_SIZE": 16 * 1024,
                },
            )
        return out


def embedding(weight, token_ids, padding_idx=None, num_tokens_per_seq=None, out=None):
    if num_tokens_per_seq is None:
        num_tokens_per_seq = token_ids.shape[-1]
    if out is None:
        batch_shape = token_ids.shape[:-1]
        out = torch.empty(
            *batch_shape,
            num_tokens_per_seq,
            weight.shape[-1],
            device=weight.device,
            dtype=weight.dtype,
        )
    else:
        assert out.shape[0] == token_ids.shape[0]
        assert out.shape[2] == weight.shape[-1]
    return Embedding.apply(weight, token_ids, padding_idx, num_tokens_per_seq, out)
