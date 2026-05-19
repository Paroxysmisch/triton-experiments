import torch
import triton
import triton.language as tl
from triton import next_power_of_2

@triton.jit
def matmul_tma_load_store(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    OUTPUT_F16: tl.constexpr,
):
    """Kernel for computing the matmul C = A x B.
    A has shape (M, K), B has shape (K, N) and C has shape (M, N)
    """
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr, shape=(M, K), strides=(stride_am, stride_ak),
        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_K), order=(1, 0)
    )
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr, shape=(K, N), strides=(stride_bk, stride_bn),
        offsets=(0, 0), block_shape=(BLOCK_K, BLOCK_N), order=(0, 1)
    )
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr, shape=(M, N), strides=(stride_cm, stride_cn),
        offsets=(0, 0), block_shape=(BLOCK_M, BLOCK_N), order=(1, 0)
    )
    a = tl.load(a_block_ptr)
    b = tl.load(b_block_ptr)
    c = tl.dot(a, b)
    if OUTPUT_F16:
        c = c.to(tl.float16)
    tl.store(c_block_ptr, c)

def warpper_tma_load_store(M, N, K, num_warps, num_ctas,
                           trans_a, trans_b,
                           out_dtype_str):
    a = torch.randn((M, K), device='cuda', dtype=torch.float16)
    b = torch.randn((K, N), device='cuda', dtype=torch.float16)
    if trans_a:
        a = a.t()
    if trans_b:
        b = b.t()
    c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    if out_dtype_str == 'float16':
        c = torch.empty((M, N), device='cuda', dtype=torch.float16)
    elif out_dtype_str == 'bf16':
        c = torch.empty((M, N), device='cuda', dtype=torch.bfloat16)

    BLOCK_M = next_power_of_2(M)
    BLOCK_N = next_power_of_2(N)
    BLOCK_K = next_power_of_2(K)
    matmul_tma_load_store[(1, 1)](
        a_ptr=a, b_ptr=b, c_ptr=c,
        M=M, N=N, K=K,
        stride_am=a.stride(0), stride_ak=a.stride(1),
        stride_bk=b.stride(0), stride_bn=b.stride(1),
        stride_cm=c.stride(0), stride_cn=c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
        num_warps=num_warps,
        num_ctas=num_ctas,
        OUTPUT_F16=(out_dtype_str == 'float16'),
    )
    return a, b, c
