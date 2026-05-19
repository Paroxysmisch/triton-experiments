import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    N,
    K,
    lora_indices,
    input_batch_stride,
    input_n_stride,
    input_k_stride,
    lora_batch_stride,
    lora_n_stride,
    lora_r_stride,
    out_batch_stride,
    out_n_stride,
    out_k_stride,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    ADD_INPUT: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    batch_idx = tl.program_id(2)
    n_blk_idx = tl.program_id(0)
    k_blk_idx = tl.program_id(1)

    n_range = n_blk_idx * BLOCK_N + tl.arange(0, BLOCK_N)
    k_range = k_blk_idx * BLOCK_K + tl.arange(0, BLOCK_K)

    lora_index = tl.load(lora_indices + batch_idx)
    if lora_index == -1:
        return

    input_block_ptr = (
        input_ptr
        + batch_idx * input_batch_stride
        + n_range[:, None] * input_n_stride
        + k_range[None, :] * input_k_stride
    )
    lora_block_ptr = (
        lora_ptr
        + batch_idx * lora_batch_stride
        + n_range[:, None] * lora_n_stride
        + k_range[None, :] * lora_r_stride
    )

    if CAST_TYPE:
        accumulator = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    else:
        accumulator = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float16)
    for _ in range(0, tl.cdiv(K, BLOCK_K)):
        input_block = tl.load(
            input_block_ptr,
            mask=(n_range[:, None] < N) & (k_range[None, :] < K - k_blk_idx * BLOCK_K),
            other=0.0,
        )
        lora_block = tl.load(
            lora_block_ptr,
            mask=(n_range[:, None] < N) & (k_range[None, :] < K - k_blk_idx * BLOCK_K),
            other=0.0,
        )

        accumulator += tl.dot(input_block, lora_block, allow_tf32=False)

        input_block_ptr += BLOCK_K * k_blk_idx * input_k_stride
        lora_block_ptr += BLOCK_K * k_blk_idx * lora_r_stride
        k_blk_idx += 1

    accumulator = tl.where((n_range[:, None] < N) & (k_range[None, :] < K), accumulator, 0.0)

    if ADD_INPUT:
        out_block_ptr = (
            out_ptr
            + batch_idx * out_batch_stride
            + n_range[:, None] * out_n_stride
            + k_range[None, :] * out_k_stride
        )
        out_block = tl.load(
            out_block_ptr,
            mask=(n_range[:, None] < N) & (k_range[None, :] < K),
            other=0.0,
        )
        accumulator += out_block

    tl.store(
        out_ptr
        + batch_idx * out_batch_stride
        + n_range[:, None] * out_n_stride
        + k_range[None, :] * out_k_stride,
        accumulator,
        mask=(n_range[:, None] < N) & (k_range[None, :] < K),
    )


def _bgmv_expand(
    input: torch.Tensor,
    lora_A: torch.Tensor,
    out: torch.Tensor | None,
    lora_indices: torch.Tensor,
    add_input: bool,
) -> torch.Tensor:
    if out is None:
        out = torch.empty_like(input)
    else:
        assert out.shape == input.shape and out.stride() == input.stride()

    assert (
        input.ndim == 3
        and lora_A.ndim == 3
        and lora_indices.ndim == 1
        and input.shape[1] == lora_A.shape[0]
    )
    input, lora_A = input.contiguous(), lora_A.contiguous()
    N, K, batch_size = input.shape[1], input.shape[2], lora_indices.shape[0]
    BLOCK_K, BLOCK_N = 32, 64
    if N <= 128:
        BLOCK_N = 128
    if N <= 64:
        BLOCK_N = 64
    if N <= 32:
        BLOCK_N = 32

    CAST_TYPE = False
    if input.dtype == torch.float32 and lora_A.dtype == torch.float32:
        CAST_TYPE = True

    def grid(meta):
        return (
            triton.cdiv(N, meta["BLOCK_N"]),
            triton.cdiv(K, meta["BLOCK_K"]),
            batch_size,
        )

    _bgmv_expand_kernel[grid](
        input,
        lora_A,
        out,
        N,
        K,
        lora_indices,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        lora_A.stride(0),
        lora_A.stride(1),
        lora_A.stride(2),
        lora_indices.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        ADD_INPUT=add_input,
        CAST_TYPE=CAST_TYPE,
    )
    return out
