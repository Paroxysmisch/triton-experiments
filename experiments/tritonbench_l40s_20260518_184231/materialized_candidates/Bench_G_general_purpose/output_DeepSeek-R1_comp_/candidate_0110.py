import torch
import triton
import triton.language as tl

@triton.jit
def _bgmv_expand_kernel(
    input_ptr, lora_ptr, out_ptr,
    N, K,
    stride_ik, stride_in,
    stride_lk, stride_ln,
    stride_ok, stride_on,
    lora_indices_ptr,
    split_n: tl.constexpr,
    add_input: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
    CAST_TYPE: tl.constexpr,
):
    pid_batch = tl.program_id(0)
    pid_split = tl.program_id(1)
    
    lora_index = tl.load(lora_indices_ptr + pid_batch)
    if lora_index == -1:
        return
    
    n_start = pid_split * BLOCK_N
    n_offsets = n_start + tl.arange(0, BLOCK_N)
    mask_n = n_offsets < N
    
    acc = tl.zeros((BLOCK_N,), dtype=tl.float32)
    
    lora_base = lora_ptr + lora_index * (stride_ln * N)
    
    for k in range(0, K, BLOCK_K):
        k_offsets = k + tl.arange(0, BLOCK_K)
        mask_k = k_offsets < K
        
        input_vals = tl.load(input_ptr + pid_batch * stride_in + k_offsets * stride_ik, mask=mask_k, other=0.0)
        
        lora_ptrs = lora_base + (n_offsets[:, None] * stride_ln + k_offsets[None, :] * stride_lk)
        lora_vals = tl.load(lora_ptrs, mask=mask_n[:, None] & mask_k[None, :], other=0.0)
        
        if CAST_TYPE:
            input_vals = input_vals.to(tl.float32)
            lora_vals = lora_vals.to(tl.float32)
        
        acc += tl.sum(input_vals[None, :] * lora_vals, axis=1)
    
    if add_input:
        k_offsets_expand = n_offsets % K
        input_add_vals = tl.load(input_ptr + pid_batch * stride_in + k_offsets_expand * stride_ik, mask=mask_n, other=0.0)
        if CAST_TYPE:
            input_add_vals = input_add_vals.to(tl.float32)
        acc += input_add_vals
    
    output_dtype = tl.float16 if CAST_TYPE else input_vals.dtype
    acc = acc.to(output_dtype)
    
    tl.store(out_ptr + pid_batch * stride_on + n_offsets * stride_ok, acc, mask=mask_n)

def _bgmv_expand(input: torch.Tensor, lora: torch.Tensor, out: torch.Tensor, lora_indices: torch.Tensor, split_n: int, add_input: bool = False):
    assert input.is_contiguous(), "Input tensor must be contiguous"
    assert lora.is_contiguous(), "LoRA tensor must be contiguous"
    assert out.is_contiguous(), "Output tensor must be contiguous"
    assert lora_indices.is_contiguous(), "LoRA indices must be contiguous"
    
    batch_size, K = input.shape
    L, N, lora_K = lora.shape
    assert K == lora_K, f"Input feature size {K} does not match LoRA K dimension {lora_K}"
    assert lora_indices.shape == (batch_size,), "LoRA indices must have shape (batch_size,)"
    
    expected_output_shape = (batch_size, N)
    assert out.shape == expected_output_shape, f"Output tensor must have shape {expected_output_shape}"
    
    cast_type = input.dtype != lora.dtype
    if cast_type:
        assert input.dtype in [torch.float16, torch.float32] and lora.dtype in [torch.float16, torch.float32], "Casting only supported between float16 and float32"
    
    BLOCK_N = 128
    while BLOCK_N > 1 and (N % BLOCK_N != 0 and split_n * BLOCK_N < N):
        BLOCK_N //= 2
    
    BLOCK_K = 64 if K % 64 == 0 else 32
    
    grid = (batch_size, (N + BLOCK_N - 1) // BLOCK_N)
    
    _bgmv_expand_kernel[grid](
        input, lora, out,
        N, K,
        input.stride(1), input.stride(0),
        lora.stride(2), lora.stride(1),
        out.stride(1), out.stride(0),
        lora_indices,
        split_n=split_n,
        add_input=add_input,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        CAST_TYPE=cast_type,
    )
