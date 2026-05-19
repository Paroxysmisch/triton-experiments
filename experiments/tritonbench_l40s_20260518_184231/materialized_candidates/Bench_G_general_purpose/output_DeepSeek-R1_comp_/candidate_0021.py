import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=4),
        triton.Config({'BLOCK_M': 128, 'BLOCK_N': 128, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'BLOCK_M': 256, 'BLOCK_N': 64, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8),
        triton.Config({'BLOCK_M': 64, 'BLOCK_N': 256, 'BLOCK_K': 32, 'GROUP_M': 8}, num_warps=8),
    ],
    key=['M', 'N', 'K'],
)
@triton.jit
def bmm_kernel(
    a_ptr, b_ptr, o_ptr,
    B, M, N, K,
    stride_ab, stride_am, stride_ak,
    stride_bb, stride_bk, stride_bn,
    stride_ob, stride_om, stride_on,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid_batch = tl.program_id(2)
    pid = tl.program_id(0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n

    if GROUP_M != 0:
        group_size = GROUP_M
        group_id = pid_m // group_size
        num_groups_m = tl.cdiv(tl.cdiv(M, BLOCK_M), group_size)
        pid_m = pid_m % group_size
        pid = (group_id * group_size + pid_m) * num_pid_n + pid_n

    off_om = pid_m * BLOCK_M
    off_on = pid_n * BLOCK_N

    a_ptr += pid_batch * stride_ab
    b_ptr += pid_batch * stride_bb
    o_ptr += pid_batch * stride_ob

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    a_offsets = off_om + tl.arange(0, BLOCK_M)[:, None]
    b_offsets = tl.arange(0, BLOCK_N)[None, :]
    k_range = tl.arange(0, BLOCK_K)[None, :]

    for k in range(0, K, BLOCK_K):
        a_mask = (a_offsets < M) & (k + k_range < K)
        a = tl.load(a_ptr + a_offsets * stride_am + (k + k_range) * stride_ak, mask=a_mask, other=0.0)
        b_mask = (k + k_range < K) & (b_offsets < N)
        b = tl.load(b_ptr + (k + k_range) * stride_bk + off_on * stride_bn + b_offsets, mask=b_mask, other=0.0)
        acc += tl.dot(a, b, out_dtype=tl.float32)

    o_offsets = off_om + tl.arange(0, BLOCK_M)[:, None]
    o_mask = (o_offsets < M) & (off_on + tl.arange(0, BLOCK_N)[None, :] < N)
    tl.store(o_ptr + o_offsets * stride_om + (off_on + tl.arange(0, BLOCK_N)) * stride_on, acc, mask=o_mask)

def bmm(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    assert a.is_cuda and b.is_cuda, "Inputs must be on GPU"
    assert a.dtype == b.dtype == torch.float16, "Inputs must be float16"
    assert a.dim() == 3 and b.dim() == 3, "Inputs must be 3D tensors"
    batch, M, K = a.shape
    batch_b, K_, N = b.shape
    assert batch == batch_b and K == K_, "Dimension mismatch"
    
    o = torch.empty((batch, M, N), device=a.device, dtype=a.dtype)
    
    def grid(meta):
        grid_m = triton.cdiv(M, meta['BLOCK_M'])
        grid_n = triton.cdiv(N, meta['BLOCK_N'])
        return (grid_m * grid_n, 1, batch)
    
    bmm_kernel[grid](
        a, b, o,
        batch, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        o.stride(0), o.stride(1), o.stride(2),
        GROUP_M=8
    )
    return o
