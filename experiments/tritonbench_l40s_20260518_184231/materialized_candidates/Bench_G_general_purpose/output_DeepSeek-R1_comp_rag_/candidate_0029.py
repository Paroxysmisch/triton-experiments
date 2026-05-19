import triton
import triton.language as tl
import torch

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr,
    lora_ptr,
    out_ptr,
    lora_indices_ptr,
    scaling,
    batch_size,
    n_rank,
    input_batch_stride,
    input_stride_k,
    lora_stride_batch,
    lora_stride_n,
    lora_stride_k,
    out_batch_stride,
    out_stride_n,
    lora_indices_batch_stride,
    BLOCK_K: tl.constexpr,
    BLOCK_N: tl.constexpr,
    SPLIT_K: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_split = tl.program_id(2)

    lora_idx = tl.load(lora_indices_ptr + pid_batch * lora_indices_batch_stride)
    if lora_idx == -1:
        return

    k_start = pid_split * BLOCK_K
    k_offsets = k_start + tl.arange(0, BLOCK_K)
    k_mask = k_offsets < n_rank

    input_ptr += pid_batch * input_batch_stride + k_offsets * input_stride_k
    input_data = tl.load(input_ptr, mask=k_mask, other=0.0)

    lora_ptr += lora_idx * lora_stride_batch + pid_n * lora_stride_n + k_offsets * lora_stride_k
    lora_data = tl.load(lora_ptr, mask=k_mask, other=0.0)

    product = input_data * lora_data
    sum_k = tl.sum(product, axis=0)
    sum_k *= scaling

    out_ptr += pid_batch * out_batch_stride + pid_n * out_stride_n
    if SPLIT_K > 1:
        tl.atomic_add(out_ptr, sum_k.to(tl.float16))
    else:
        tl.store(out_ptr, sum_k.to(tl.float16))

def _bgmv_shrink(
    input: torch.Tensor,
    lora: torch.Tensor,
    out: torch.Tensor,
    lora_indices: torch.Tensor,
    scaling: float,
):
    assert input.is_contiguous(), "Input tensor must be contiguous"
    assert lora.is_contiguous(), "LoRA tensor must be contiguous"
    assert out.is_contiguous(), "Output tensor must be contiguous"
    assert lora_indices.is_contiguous(), "LoRA indices must be contiguous"

    batch_size = lora.size(0)
    N = lora.size(1)
    K = lora.size(2)

    BLOCK_N = 1 << (N - 1).bit_length()
    BLOCK_K = 64  # Tune this based on hardware and performance needs
    SPLIT_K = (K + BLOCK_K - 1) // BLOCK_K

    grid = (batch_size, triton.cdiv(N, BLOCK_N), SPLIT_K)

    _bgmv_shrink_kernel[grid](
        input_ptr=input,
        lora_ptr=lora,
        out_ptr=out,
        lora_indices_ptr=lora_indices,
        scaling=scaling,
        batch_size=batch_size,
        n_rank=K,
        input_batch_stride=input.stride(0),
        input_stride_k=input.stride(1),
        lora_stride_batch=lora.stride(0),
        lora_stride_n=lora.stride(1),
        lora_stride_k=lora.stride(2),
        out_batch_stride=out.stride(0),
        out_stride_n=out.stride(1),
        lora_indices_batch_stride=lora_indices.stride(0),
        BLOCK_K=BLOCK_K,
        BLOCK_N=BLOCK_N,
        SPLIT_K=SPLIT_K,
    )
