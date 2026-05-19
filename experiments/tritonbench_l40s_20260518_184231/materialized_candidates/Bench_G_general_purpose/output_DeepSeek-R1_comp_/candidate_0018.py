import triton
import triton.language as tl

@triton.jit
def _bgmv_shrink_kernel(
    input_ptr, lora_ptr, output_ptr, lora_indices_ptr,
    scaling, input_stride_b, input_stride_k,
    lora_stride_b, lora_stride_n, lora_stride_k,
    output_stride_b, output_stride_n,
    H, N, K,
    BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr, SPLIT_K: tl.constexpr
):
    pid_b = tl.program_id(0)
    pid_split = tl.program_id(1)
    
    lora_idx = tl.load(lora_indices_ptr + pid_b)
    if lora_idx == -1:
        return
    
    offs_n = tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)
    
    input_start = pid_b * input_stride_b
    lora_start = lora_idx * lora_stride_b
    
    k_offset_base = pid_split * BLOCK_K * SPLIT_K
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    for sk in range(SPLIT_K):
        k_offset = k_offset_base + sk * BLOCK_K
        mask_k = k_offset + offs_k < K
        
        input_ptr_sk = input_ptr + input_start + k_offset
        input = tl.load(input_ptr_sk, mask=mask_k, other=0.0)
        
        lora_ptrs_sk = lora_ptr + lora_start + (offs_n[:, None] * lora_stride_n + (k_offset + offs_k[None, :]) * lora_stride_k)
        lora = tl.load(lora_ptrs_sk, mask=mask_k & (offs_n[:, None] < N), other=0.0)
        
        acc += tl.sum(input[None, :] * lora, axis=1)
    
    acc = acc * scaling
    out_offs = pid_b * output_stride_b + offs_n
    tl.atomic_add(output_ptr + out_offs, acc, mask=offs_n < N)

def _bgmv_shrink(
    y: torch.Tensor,
    x: torch.Tensor,
    wa: torch.Tensor,
    lora_indices: torch.Tensor,
    scaling: float,
    BLOCK_K: int = 32,
    SPLIT_K: int = 1,
):
    assert wa.dim() == 3, "LORA weights must be 3D (L, N, K)"
    assert x.dim() == 2, "Input must be 2D (B, H)"
    B, H = x.shape
    L, N, K = wa.shape

    x = x.contiguous()
    wa = wa.contiguous()
    y = y.contiguous()
    lora_indices = lora_indices.contiguous()

    BLOCK_N = triton.next_power_of_2(N)
    grid = (B, triton.cdiv(K, BLOCK_K * SPLIT_K))

    _bgmv_shrink_kernel[grid](
        x, wa, y, lora_indices,
        scaling,
        x.stride(0), x.stride(1),
        wa.stride(0), wa.stride(1), wa.stride(2),
        y.stride(0), y.stride(1),
        H, N, K,
        BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, SPLIT_K=SPLIT_K
    )
