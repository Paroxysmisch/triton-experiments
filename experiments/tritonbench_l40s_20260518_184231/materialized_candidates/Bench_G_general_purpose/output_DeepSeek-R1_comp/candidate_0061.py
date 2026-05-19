import torch
import triton
import triton.language as tl

@triton.jit
def _sgmv_expand_slice_kernel(
    Y, X, W, lora_indices,
    stride_yb, stride_ys, stride_yn,
    stride_xb, stride_xs, stride_xk,
    stride_wl, stride_wk, stride_wn,
    max_seqs, max_seq_length, rank, n_problems,
    num_loras, n,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    INPUT_PRECISION: tl.constexpr
):
    pid_problem = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)
    
    if pid_problem >= n_problems:
        return
    
    lora_idx = tl.load(lora_indices + pid_problem)
    if lora_idx < 0 or lora_idx >= num_loras:
        return
    
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    offs_k = tl.arange(0, BLOCK_K)

    y_ptrs = (
        Y + pid_problem * stride_yb + 
        offs_m[:, None] * stride_ys + 
        offs_n[None, :] * stride_yn
    )
    
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    for k in range(0, rank, BLOCK_K):
        k_mask = k + offs_k < rank
        
        x_ptrs = (
            X + pid_problem * stride_xb + 
            offs_m[:, None] * stride_xs + 
            (k + offs_k)[None, :] * stride_xk
        )
        
        w_ptrs = (
            W + lora_idx * stride_wl + 
            (k + offs_k)[:, None] * stride_wk + 
            offs_n[None, :] * stride_wn
        )
        
        x = tl.load(x_ptrs, mask=k_mask[None, :], other=0.0)
        w = tl.load(w_ptrs, mask=k_mask[:, None], other=0.0)
        
        if INPUT_PRECISION == 'fp32':
            x = x.to(tl.float16)
            
        acc += tl.dot(x, w, allow_tf32=True)
    
    seq_mask = offs_m[:, None] < max_seq_length
    feat_mask = offs_n[None, :] < n
    tl.store(y_ptrs, acc.to(Y.dtype.element_ty), mask=seq_mask & feat_mask)

def _sgmv_expand_slice(
    Y: torch.Tensor, 
    X: torch.Tensor, 
    W: torch.Tensor, 
    lora_indices: torch.Tensor, 
    max_seq_length: int
):
    assert X.dim() == 3 and W.dim() == 3 and lora_indices.dim() == 1
    assert Y.dim() == 3 and Y.is_contiguous()
    
    batch_size, seq_len, rank = X.shape
    num_loras, w_rank, n = W.shape
    assert rank == w_rank and batch_size == lora_indices.size(0)
    
    X = X if X.is_contiguous() else X.contiguous()
    W = W if W.is_contiguous() else W.contiguous()
    lora_indices = lora_indices.contiguous()

    if W.dtype != X.dtype:
        W = W.to(X.dtype)
        
    input_precision = 'fp32' if X.dtype == torch.float32 else 'fp16'
    
    BLOCK_M, BLOCK_N, BLOCK_K = 16, 64, 32
    grid = (
        batch_size, 
        triton.cdiv(max_seq_length, BLOCK_M), 
        triton.cdiv(n, BLOCK_N)
    )
    
    _sgmv_expand_slice_kernel[grid](
        Y, X, W, lora_indices,
        Y.stride(0), Y.stride(1), Y.stride(2),
        X.stride(0), X.stride(1), X.stride(2),
        W.stride(0), W.stride(1), W.stride(2),
        max_seqs=batch_size,
        max_seq_length=max_seq_length,
        rank=rank,
        n_problems=batch_size,
        num_loras=num_loras,
        n=n,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        BLOCK_K=BLOCK_K,
        INPUT_PRECISION=input_precision
    )
