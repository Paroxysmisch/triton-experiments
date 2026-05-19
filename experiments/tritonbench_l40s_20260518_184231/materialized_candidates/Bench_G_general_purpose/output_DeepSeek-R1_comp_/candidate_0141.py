import torch
import triton
import triton.language as tl

@triton.jit
def matmul_kernel_persistent(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Stride information
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Tile sizes
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    total_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    total_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    total_tiles = total_pid_m * total_pid_n
    grid_size = tl.num_programs(axis=0)
    
    for tile_idx in range(pid, total_tiles, grid_size):
        pid_m = tile_idx // total_pid_n
        pid_n = tile_idx % total_pid_n
        
        offs_m = pid_m * BLOCK_SIZE_M
        offs_n = pid_n * BLOCK_SIZE_N
        
        accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
        
        for k in range(0, K, BLOCK_SIZE_K):
            k_offs = k + tl.arange(0, BLOCK_SIZE_K)
            
            a_ptrs = a_ptr + (offs_m * stride_am)[:, None] + (k_offs * stride_ak)[None, :]
            b_ptrs = b_ptr + (k_offs * stride_bk)[:, None] + (offs_n * stride_bn)[None, :]
            
            a_mask = (offs_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (k_offs[None, :] < K)
            b_mask = (k_offs[:, None] < K) & (offs_n + tl.arange(0, BLOCK_SIZE_N)[None, :] < N)
            
            a = tl.load(a_ptrs, mask=a_mask, other=0.0)
            b = tl.load(b_ptrs, mask=b_mask, other=0.0)
            
            accumulator += tl.dot(a, b, out_dtype=tl.float32)
        
        c = accumulator.to(tl.float16)  # Adjust based on output type
        
        c_ptrs = c_ptr + (offs_m * stride_cm)[:, None] + (offs_n * stride_cn)[None, :]
        c_mask = (offs_m + tl.arange(0, BLOCK_SIZE_M)[:, None] < M) & (offs_n + tl.arange(0, BLOCK_SIZE_N)[None, :] < N)
        tl.store(c_ptrs, c, mask=c_mask)

def matmul_persistent(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.dim() == 2 and b.dim() == 2, "Inputs must be 2D matrices"
    assert a.shape[1] == b.shape[0], "Incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    
    device = a.device
    if device.type != 'cuda':
        raise RuntimeError("Device must be CUDA")
    c = torch.empty((M, N), device=device, dtype=a.dtype)
    
    num_sms = torch.cuda.get_device_properties(device).multi_processor_count
    BLOCK_SIZE_M = 64
    BLOCK_SIZE_N = 64
    BLOCK_SIZE_K = 32 if a.dtype == torch.float16 else 16
    num_warps = 4 if a.dtype == torch.float16 else 8
    num_stages = 3
    
    grid = (num_sms * 4,)
    
    matmul_kernel_persistent[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        num_warps=num_warps,
        num_stages=num_stages
    )
    return c
