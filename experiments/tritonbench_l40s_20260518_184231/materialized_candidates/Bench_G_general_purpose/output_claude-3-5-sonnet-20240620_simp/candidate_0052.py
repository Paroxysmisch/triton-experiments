import triton
import triton.language as tl

@triton.jit
def matmul_kernel(
    # Pointers to matrices
    a_ptr, b_ptr, c_ptr,
    # Matrix dimensions
    M, N, K,
    # Matrix strides
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    # Block sizes
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    # Pipeline stages and warps
    STAGES: tl.constexpr,
    NUM_WARPS: tl.constexpr,
    # Whether to use swizzling
    USE_SWIZZLE: tl.constexpr
):
    # Calculate tile coordinates
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    num_pid_total = num_pid_m * num_pid_n
    
    # Swizzled or linear tile mapping
    if USE_SWIZZLE:
        width = num_pid_n
        group_size = 8  # Size of swizzle group
        row = pid // width
        col = pid % width
        group_id = row // group_size
        group_row = row % group_size
        swizzled_col = (col + group_id) % width
        pid_m = row
        pid_n = swizzled_col
    else:
        pid_m = pid // num_pid_n
        pid_n = pid % num_pid_n

    # Block pointers
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    
    # Initialize accumulator
    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    
    # Iterate through K dimension
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        k_idx = k * BLOCK_K
        # Load blocks from A and B
        a = tl.load(a_ptr + offs_am[:, None] * stride_am + (k_idx + offs_k[None, :]) * stride_ak,
                   mask=(k_idx + offs_k[None, :]) < K, other=0.0)
        b = tl.load(b_ptr + (k_idx + offs_k[:, None]) * stride_bk + offs_bn[None, :] * stride_bn,
                   mask=(k_idx + offs_k[:, None]) < K, other=0.0)
        # Compute matrix multiplication
        acc += tl.dot(a, b)
    
    # Store result
    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn,
             acc, mask=mask)

# Wrapper class for matrix multiplication
class Matmul:
    def __init__(self, BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, STAGES=3, NUM_WARPS=8, USE_SWIZZLE=True):
        self.BLOCK_M = BLOCK_M
        self.BLOCK_N = BLOCK_N
        self.BLOCK_K = BLOCK_K
        self.STAGES = STAGES
        self.NUM_WARPS = NUM_WARPS
        self.USE_SWIZZLE = USE_SWIZZLE
        
    def __call__(self, a, b, c=None):
        # Extract matrix dimensions
        M, K = a.shape
        K, N = b.shape
        
        # Allocate output if not provided
        if c is None:
            c = torch.empty((M, N), device=a.device, dtype=a.dtype)
            
        # Get strides
        stride_am, stride_ak = a.stride()
        stride_bk, stride_bn = b.stride()
        stride_cm, stride_cn = c.stride()
        
        # Launch kernel
        grid = lambda META: (triton.cdiv(M, META['BLOCK_M']) * triton.cdiv(N, META['BLOCK_N']),)
        matmul_kernel[grid](
            a, b, c,
            M, N, K,
            stride_am, stride_ak,
            stride_bk, stride_bn,
            stride_cm, stride_cn,
            self.BLOCK_M, self.BLOCK_N, self.BLOCK_K,
            self.STAGES,
            self.NUM_WARPS,
            self.USE_SWIZZLE
        )
        return c
