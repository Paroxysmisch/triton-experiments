import torch
import triton
import triton.language as tl
from torch import Tensor
from packaging import version

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    scaling,
    n,
    k,
    b,
    lora_indices,
    stride_input_b,
    stride_input_n,
    stride_input_k,
    stride_lora_b,
    stride_lora_n,
    stride_lora_k,
    stride_lora_out_b,
    stride_lora_out_n,
    stride_lora_out_k,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    # batch idx
    pid_b = tl.program_id(axis=0)
    # lora idx
    pid_l = tl.program_id(axis=1)
    # handle -1, skip compute
    if pid_l == -1:
        return
    # compute offset
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    # input: [b, n, k]
    input_offset = (
        pid_b * stride_input_b
        + (offs_n[:, None] * stride_input_n + offs_k[None, :] * stride_input_k)
    )
    # lora: [b, n, k]
    lora_offset = (
        pid_l * stride_lora_b
        + (offs_n[:, None] * stride_lora_n + offs_k[None, :] * stride_lora_k)
    )
    # compute
    input_ptrs = input_ptr + input_offset
    lora_ptrs = lora_ptr + lora_offset
    acc = tl.zeros((BLOCK_N, BLOCK_K), dtype=tl.float32)
    for i in range(0, tl.cdiv(k, BLOCK_K * SPLIT_K)):
        # ---------------------
        # navie version
        # ---------------------
        # input_block = tl.load(
        #     input_ptrs,
        #     mask=(offs_n[:, None] < n) & (offs_k[None, :] < k - i * BLOCK_K * SPLIT_K),
        #     other=0.0,
        # ).to(tl.float32)
        # lora_block = tl.load(
        #     lora_ptrs,
        #     mask=(offs_n[:, None] < n)
        #     & (offs_k[None, :] < k - i * BLOCK_K * SPLIT_K),
        #     other=0.0,
        # )
        # acc = tl.dot(input_block, lora_block.to(input_block), acc)
        # ---------------------
        # fused version
        # ---------------------
        mask_k = offs_k[None, :] < k - i * BLOCK_K * SPLIT_K
        mask_n = offs_n[:, None] < n
        mask = mask_k & mask_n
        input_block = tl.load(
            input_ptrs, mask=(mask_k & mask_n), other=0.0
        ).to(tl.float32)
        lora_block = tl.load(lora_ptrs, mask=mask, other=0.0)
        acc += tl.dot(input_block, lora_block.to(input_block))
        # update ptr
        input_ptrs += BLOCK_K * SPLIT_K
        lora_ptrs += BLOCK_K * SPLIT_K
    acc = tl.sum(acc, axis=1)
    acc = acc.to(out_ptr.dtype.element_ty)
    # output: [b, n, k]
    lora_out_offset = pid_l * stride_lora_out_b + offs_n * stride_lora_out_n
    lora_out_ptrs = out_ptr + lora_out_offset
    tl.atomic_add(lora_out_ptrs, acc[:, None], mask=offs_n[None, :] < n)

def _bgmv_shrink(
    input: Tensor,
    lora_weight: Tensor,
    out: Tensor,
    scaling: float,
    lora_indices: Tensor,
) -> None:
    # check
    assert input.is_contiguous()
    assert lora_weight.is_contiguous()
    assert out.is_contiguous()
    # param
    [b, n, k] = lora_weight.shape
    # lora batch
    lora_b = lora_weight.shape[0]
    # block
    BLOCK_N = triton.next_power_of_2(n)
    # grid
    grid = (b, lora_b)
    # launch kernel
    _bgmv_shrink_kernel[grid](
        input,
        lora_weight,
        out,
        scaling,
        n,
        k,
        b,
        lora_indices,
        input.stride(0),
        input.stride(1),
        input.stride(2),
        lora_weight.stride(0),
        lora_weight.stride(1),
        lora_weight.stride(2),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        BLOCK_N=BLOCK_N,
        # NOTE: this is hack, triton does not support dynamic constexpr value
        # BLOCK_K=cfg["BLOCK_K"],
        BLOCK_K=32,
        # NOTE: this is hack, triton does not support dynamic constexpr value
        # SPLIT_K=cfg["SPLIT_K"],
        SPLIT_K=1,
    )
    return
