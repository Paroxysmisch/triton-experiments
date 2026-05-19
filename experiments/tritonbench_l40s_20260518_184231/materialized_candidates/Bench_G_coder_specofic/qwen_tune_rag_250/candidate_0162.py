import torch
import triton
import triton.language as tl
from torch._inductor.triton_heuristics import grid
from torch._inductor.utils import instance_descriptor
from torch._inductor import triton_helpers

@triton.jit
def matmul_kernel(
        a_ptr, b_ptr, c_ptr,
        M, N, K,
        stride_am, stride_ak,
        stride_bk, stride_bn,
        stride_cm, stride_cn,
        m_tile_size: tl.constexpr, n_tile_size: tl.constexpr, k_tile_size: tl.constexpr,
        NUM_SM: tl.constexpr,
):
    # Get program ID
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, m_tile_size)
    num_pid_n = tl.cdiv(N, n_tile_size)

    # Determine block indices and offsets
    block_idx = pid // num_pid_m
    y_offset = block_idx * n_tile_size
    block_idx_in_group = block_idx % NUM_SM
    x_offset = block_idx_in_group * m_tile_size
    x_index = x_offset + (pid % m_tile_size)
    y_index = y_offset + (pid % n_tile_size)

    # Initialize pointers and compute memory offsets
    stride_am = a_ptr.stride(0)
    stride_ak = a_ptr.stride(1)
    a_tile_ptr = a_ptr + x_index * stride_am
    b_tile_ptr = b_ptr + y_index * stride_bn

    offs_k = tl.arange(0, k_tile_size)
    offs_am = x_index + tl.arange(0, m_tile_size) * stride_am
    offs_bn = y_index + tl.arange(0, n_tile_size) * stride_bn

    # Load tiles of A and B matrices
    a = tl.load(a_tile_ptr + offs_k[None, :] * stride_ak, mask=(offs_k[None, :] < K), other=0.0)
    b = tl.load(b_tile_ptr + offs_k[:, None] * stride_bk, mask=(offs_k[:, None] < K), other=0.0)

    # Compute matrix multiplication
    c = tl.dot(a, b.to(tl.float16))

    # Store result in output matrix
    c_ptr[x_index, y_index] = c.to(c_ptr.dtype.element_ty)

def matmul(a, b):
    # Ensure input tensors are contiguous
    assert a.is_contiguous(), "Matrix A must be contiguous"
    assert b.is_contiguous(), "Matrix B must be contiguous"
    
    # Get matrix dimensions
    M, K = a.shape
    K, N = b.shape

    # Initialize output tensor
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    # Define grid and stream for kernel execution
    stream = torch.cuda.current_stream(a.device)
    triton_helpers.run(matmul_kernel, grid(M, N), stream=stream,
                       a_ptr=a, b_ptr=b, c_ptr=c,
                       M=M, N=N, K=K,
                       stride_am=a.stride(0), stride_ak=a.stride(1),
                       stride_bk=b.stride(0), stride_bn=b.stride(1),
                       stride_cm=c.stride(0), stride_cn=c.stride(1),
                       NUM_SM=48,
                       m_tile_size=128,
                       n_tile_size=128,
                       k_tile_size=32,
                       pre_hook=init_to_zero(c),
                       constants={},
                       inductor_heuristics=heuristics,
                       inductor_meta={'signature': {0: '*i64', 1: '*i64', 2: '*i64', 3: 'i32', 4: 'i32', 5: 'i32', 6: 'i32', 7: 'i32', 8: 'i32', 9: 'i32', 10: 'i32', 11: 'i32'}, 'device': 0, 'device_type': 'cuda', 'constants': {}, 'mutated_arg_names': [], 'autotune_hints': set(), 'kernel_name': 'matmul_kernel'})
    return c
