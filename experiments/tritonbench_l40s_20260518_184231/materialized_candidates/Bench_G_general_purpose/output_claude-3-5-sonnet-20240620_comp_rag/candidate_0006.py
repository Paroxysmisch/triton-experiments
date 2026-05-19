import triton
import triton.language as tl
import torch

@triton.jit
def ff_llama(
    x_ptr, w1_ptr, w3_ptr, rms_w_ptr, out_ptr,
    M, N, K,
    stride_xm, stride_xn,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_rm, stride_rn,
    stride_om, stride_on,
    EPS: tl.float32,
    USE_FP8: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_in_group
    group_size_m = min(num_pid_m - first_pid_m, num_pid_in_group)
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m

    offs_am = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_bn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_am[:, None] * stride_xm + offs_k[None, :] * stride_xn)
    w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k + offs_bn[None, :] * stride_w1n)
    w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k + offs_bn[None, :] * stride_w3n)
    rms_w_ptrs = rms_w_ptr + offs_am * stride_rm
    
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc2 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    rms_sum = tl.zeros((BLOCK_SIZE_M,), dtype=tl.float32)

    for k in range(0, K, BLOCK_SIZE_K):
        x = tl.load(x_ptrs)
        w1 = tl.load(w1_ptrs) if not USE_FP8 else tl.load(w1_ptrs).to(tl.float16)
        w3 = tl.load(w3_ptrs) if not USE_FP8 else tl.load(w3_ptrs).to(tl.float16)
        
        acc1 += tl.dot(x, w1)
        acc2 += tl.dot(x, w3)
        rms_sum += tl.sum(x * x, axis=1)
        
        x_ptrs += BLOCK_SIZE_K * stride_xn
        w1_ptrs += BLOCK_SIZE_K * stride_w1k
        w3_ptrs += BLOCK_SIZE_K * stride_w3k

    rms_w = tl.load(rms_w_ptrs)
    rms = tl.sqrt(rms_sum / K + EPS) * rms_w
    acc1 = acc1 / rms[:, None]
    acc2 = acc2 / rms[:, None]
    
    output = tl.sigmoid(acc1) * acc2

    offs_om = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    offs_on = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    out_ptrs = out_ptr + (offs_om[:, None] * stride_om + offs_on[None, :] * stride_on)
    out_mask = (offs_om[:, None] < M) & (offs_on[None, :] < N)
    tl.store(out_ptrs, output, mask=out_mask)

def kernel_ff(x, w1, w3, rms_w, EPS=1e-6, USE_FP8=False):
    assert x.is_contiguous(), "Input tensor must be contiguous"
    assert w1.is_contiguous(), "Weight tensor w1 must be contiguous"
    assert w3.is_contiguous(), "Weight tensor w3 must be contiguous"
    assert rms_w.is_contiguous(), "RMS weight tensor must be contiguous"
    
    M, K = x.shape
    N, K = w1.shape
    assert w3.shape == (N, K), "Weight tensors w1 and w3 must have the same shape"
    assert rms_w.shape == (M,), "RMS weight tensor must have shape (M,)"
    
    w1 = w1.transpose(0, 1).contiguous()
    w3 = w3.transpose(0, 1).contiguous()
    
    output = torch.empty((M, N), device=x.device, dtype=x.dtype)
    
    BLOCK_SIZE_M = 16
    BLOCK_SIZE_N = 16
    BLOCK_SIZE_K = 32
    
    def grid(META):
        return (
            triton.cdiv(M, BLOCK_SIZE_M) * triton.cdiv(N, BLOCK_SIZE_N),
        )
    
    ff_llama[grid](
        x, w1, w3, rms_w, output,
        M, N, K,
        x.stride(0), x.stride(1),
        w1.stride(0), w1.stride(1),
        w3.stride(0), w3.stride(1),
        rms_w.stride(0), 0,
        output.stride(0), output.stride(1),
        EPS,
        USE_FP8,
        BLOCK_SIZE_M,
        BLOCK_SIZE_N,
        BLOCK_SIZE_K,
    )
    
    return output

# Example usage
if __name__ == "__main__":
    M, N, K = 1024, 1024, 1024
    x = torch.randn(M, K, device='cuda')
    w1 = torch.randn(N, K, device='cuda')
    w3 = torch.randn(N, K, device='cuda')
    rms_w = torch.randn(M, device='cuda')
    
    output = kernel_ff(x, w1, w3, rms_w)
    print(output.shape)  # Should be (M, N)
