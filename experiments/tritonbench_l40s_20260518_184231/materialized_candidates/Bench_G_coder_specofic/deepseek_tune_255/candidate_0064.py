import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_slice_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    lora_indices,
    batch,
    index,
    N: tl.constexpr,
    K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_N: tl.constexpr,
    EVEN_K: tl.constexpr,
    ADD_INPUTS: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    pid = tl.program_id(0)
    bid = tl.program_id(1)
    split_n = tl.num_programs(0)

    if EVEN_K:
        k_offsets = tl.arange(0, BLOCK_K)
    else:
        k_offsets = tl.where(tl.arange(0, BLOCK_K) < K - (bid + 1) * BLOCK_K, tl.arange(0, BLOCK_K), K - (bid + 1) * BLOCK_K + tl.arange(0, BLOCK_K))

    n_offsets = tl.arange(0, BLOCK_N) + bid * BLOCK_N + pid * SPLIT_N
    if ADD_INPUTS:
        x_ptrs = input_ptr + n_offsets[:, None] * K + k_offsets[None, :]
        lora_ptrs = lora_ptr + n_offsets[:, None] * K + k_offsets[None, :]
    else:
        x_ptrs = input_ptr + n_offsets[:, None] * K
        lora_ptrs = lora_ptr + n_offsets[:, None] * K + k_offsets[None, :]

    if CAST_TYPE:
        lobat = tl.load(lora_ptrs, mask=k_offsets[None, :] < K, other=0.0)
    else:
        lobat = tl.load(lora_ptrs, mask=k_offsets[None, :] < K, other=0)

    if ADD_INPUTS:
        if CAST_TYPE:
            x_batch = tl.load(x_ptrs, mask=k_offsets[:, None] < K, other=0.0)
        else:
            x_batch = tl.load(x_ptrs, mask=k_offsets[:, None] < K, other=0)
        lobat = lobat + x_batch

    lobat = tl.sum(lobat, axis=1)
    lobat = tl.where(n_offsets < N, lobat, 0)

    if CAST_TYPE:
        out_ptrs = out_ptr + n_offsets + index * N + batch * N * index
        tl.store(out_ptrs, lobat.to(out_ptr.dtype.element_ty), mask=n_offsets < N)
    else:
        out_ptrs = out_ptr + n_offsets + index * N + batch * N * index
        tl.store(out_ptrs, lobat, mask=n_offsets < N)


@torch.inference_mode()
def _bgmv_expand_slice(
    input: torch.Tensor,
    lora: torch.Tensor,
    lora_indices: torch.Tensor,
    batched_gemm: bool,
    add_inputs: bool,
    cast_type: bool,
):
    assert input.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert lora.dtype in [torch.float16, torch.bfloat16, torch.float32]
    assert input.shape[0] == lora.shape[0]
    assert lora_indices.shape[0] == 2

    M, N = input.shape
    K = lora.shape[1]
    batch = lora_indices.shape[1] if batched_gemm else 1
    out = torch.empty((batch * M, N), dtype=input.dtype, device=input.device) if batched_gemm else torch.empty((M, N), dtype=input.dtype, device=input.device)

    if K <= 1024:
        BLOCK_K = K
    else:
        BLOCK_K = 1024

    if N <= 1024:
        BLOCK_N = N
    else:
        BLOCK_N = 1024

    SPLIT_N = triton.next_power_of_2(N // BLOCK_N)
    if SPLIT_N > 2:
        num_warps = 8
    else:
        num_warps = 4

    if K <= 256:
        EVEN_K = True
    else:
        EVEN_K = False

    for i in range(batch):
        grid = lambda meta: (
            triton.cdiv(meta["N"], meta["BLOCK_N"] * SPLIT_N),
            triton.cdiv(meta["K"], meta["BLOCK_K"]),
        )
        _bgmv_expand_slice_kernel[grid](
            input,
            lora,
            out,
            lora_indices,
            i,
            i if batched_gemm else 0,
            N,
            K,
            BLOCK_N,
            BLOCK_K,
            SPLIT_N,
            EVEN_K,
            add_inputs,
            cast_type,
            num_warps=num_warps,
            num_stages=1,
        )

    if batched_gemm:
        return out
    else:
        return out.sum(0)[None, :]
