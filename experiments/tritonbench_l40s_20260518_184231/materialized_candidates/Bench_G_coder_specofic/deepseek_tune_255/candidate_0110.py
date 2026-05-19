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
    stride_input_n,
    stride_input_k,
    stride_lora_n,
    stride_lora_k,
    stride_out_n,
    stride_out_k,
    stride_lora_indices_n,
    split_n: tl.constexpr,
    ADD_INPUT: tl.constexpr,
    CAST_TYPE: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_split = tl.program_id(1)
    lora_index = tl.load(lora_indices + pid_batch * stride_lora_indices_n)
    if lora_index == -1:
        return
    n_offset = pid_split * split_n + tl.arange(0, BLOCK_N)
    k_offset = tl.arange(0, BLOCK_K)
    input_ptrs = input_ptr + (n_offset[:, None] * stride_input_n + k_offset[None, :] * stride_input_k)
    lora_ptrs = lora_ptr + (lora_index * stride_lora_n + n_offset[:, None] * stride_lora_k + k_offset[None, :])
    out_ptrs = out_ptr + (n_offset[:, None] * stride_out_n + k_offset[None, :] * stride_out_k)
    mask = (n_offset < N) & (k_offset < K)

    if CAST_TYPE:
        input_val = tl.load(input_ptrs, mask=mask, other=0.0).to(lora_ptr.dtype.element_ty)
        lora_val = tl.load(lora_ptrs, mask=mask)
    else:
        input_val = tl.load(input_ptrs, mask=mask, other=0.0)
        lora_val = tl.load(lora_ptrs, mask=mask).to(input_ptr.dtype.element_ty)

    if ADD_INPUT:
        out_val = tl.dot(input_val, lora_val, allow_tf32=True) + tl.load(out_ptrs, mask=mask, other=0.0)
    else:
        out_val = tl.dot(input_val, lora_val, allow_tf32=True)

    if BLOCK_N * BLOCK_K * NUM_SMEM_BANKS(K) < 65536:
        tl.store(out_ptrs, out_val, mask=mask)
    else:
        tl.atomic_add(out_ptrs, out_val, mask=mask)

def _bgmv_expand(input, lora, out, lora_indices, split_n, add_input=False):
    assert input.is_contiguous(), "input tensor must be contiguous"
    assert lora.is_contiguous(), "lora weight tensor must be contiguous"
    assert out.is_contiguous(), "out tensor must be contiguous"
    assert input.shape[0] == out.shape[0] and input.shape[1] == lora.shape[1] == out.shape[1], \
        "shapes must be consistent across inputs and out"

    N, K = input.shape
    lora_index_device = torch.where(lora_indices != -1)[0].device
    lora_indices = lora_indices.to(lora_index_device)
    CAST_TYPE = input.dtype != lora.dtype

    grid = lambda META: (input.shape[0], triton.cdiv(N, META["split_n"]))
    BLOCK_K = triton.next_power_of_2(max(K, 1))
    BLOCK_N = triton.next_power_of_2(split_n)
    num_warps = 4 if CAST_TYPE else 8

    _bgmv_expand_kernel[grid](
        input,
        lora,
        out,
        N,
        K,
        lora_indices,
        input.stride(0),
        input.stride(1),
        lora.stride(0),
        lora.stride(1),
        out.stride(0),
        out.stride(1),
        lora_indices.stride(0),
        split_n=split_n,
        ADD_INPUT=add_input,
        CAST_TYPE=CAST_TYPE,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_stages=1,
    )
