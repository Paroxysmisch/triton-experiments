import math
import torch
import triton
import triton.language as tl

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_K": 64}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_K": 128}, num_stages=3, num_warps=4),
    ],
    key=["M", "K"],
)
@triton.jit
def quantize_int8_perrow_kernel(
    fpa_ptr,       # *Fp32 input matrix
    a_ptr,         # *Int8 output quantized matrix
    as_ptr,        # *Fp32 scaling factors per row
    M,             # total rows
    K,             # total columns
    stride_fpa,    # row stride for fpa_ptr
    stride_a,      # row stride for a_ptr
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
):
    row_id = tl.program_id(0)
    row_offs = row_id * BLOCK_SIZE_M
    if row_offs >= M:
        return

    col_offsets = tl.arange(0, BLOCK_SIZE_K)
    row_mask = col_offsets < K
    ptr_fpa = fpa_ptr + row_offs * stride_fpa
    ptr_a = a_ptr + row_offs * stride_a

    fpa_vals = tl.load(ptr_fpa + col_offsets, mask=row_mask, other=0.0)
    abs_vals = tl.abs(fpa_vals)
    max_val = tl.reduce_max(abs_vals, axis=0)
    scale = 127.0 / max_val
    quantized = tl.libdevice.llrint(fpa_vals * scale)
    quantized_i8 = quantized.to(tl.int8)

    tl.store(ptr_a + col_offsets, quantized_i8, mask=row_mask)
    tl.store(as_ptr + row_id, max_val)

def quantize_int8_perrow(fpa: torch.Tensor):
    M, K = fpa.shape
    a = torch.empty_like(fpa, dtype=torch.int8)
    ascale = torch.empty(M, dtype=torch.float32, device=fpa.device)

    grid = lambda meta: (math.ceil(M / meta["BLOCK_SIZE_M"]),)
    quantize_int8_perrow_kernel[grid](
        fpa,
        a,
        ascale,
        M, K,
        fpa.stride(0),
        a.stride(0),
    )
    return a, ascale

@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE_M": 64, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 32}, num_stages=2, num_warps=2),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 64, "BLOCK_SIZE_K": 64}, num_stages=2, num_warps=4),
        triton.Config({"BLOCK_SIZE_M": 128, "BLOCK_SIZE_N": 128, "BLOCK_SIZE_K": 64}, num_stages=3, num_warps=4),
    ],
    key=["M", "N", "K"],
)
@triton.jit
def matmul_kernel(
    A_ptr, B_ptr, C_ptr,
    AS_ptr, BS_ptr, # scaling factors
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    SPLIT_K: tl.constexpr,
    BLOCK_SIZE_M: tl.constexpr, BLOCK_SIZE_N: tl.constexpr, BLOCK_SIZE_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    pid_k = tl.program_id(2)

    rm = pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)
    rn = pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)
    rk = pid_k * BLOCK_SIZE_K + tl.arange(0, BLOCK_SIZE_K)

    mask_m = rm < M
    mask_n = rn < N
    a_ptrs = A_ptr + rm[:, None] * stride_am + rk[None, :] * stride_ak
    b_ptrs = B_ptr + rk[:, None] * stride_bk + rn[None, :] * stride_bn

    a_scale = tl.load(AS_ptr + rm, mask=mask_m, other=0.0)
    b_scale = tl.load(BS_ptr + rk, mask=rk < K, other=1.0)  # if per-row scale for B, adjust accordingly

    a_vals = tl.load(a_ptrs, mask=(mask_m[:, None] & (rk[None, :] < K)), other=0).to(tl.int32)
    b_vals = tl.load(b_ptrs, mask=(rk[:, None] < K) & mask_n[None, :], other=0).to(tl.int32)

    # apply scale
    a_sc = a_scale[:, None]
    b_sc = b_scale[None, :]
    a_fp32 = a_vals.to(tl.float32) / 127.0 * a_sc
    b_fp32 = b_vals.to(tl.float32) / 127.0 * b_sc

    acc = tl.dot(a_fp32, b_fp32)
    offs_c = (rm[:, None] * stride_cm + rn[None, :] * stride_cn)
    mask_c = mask_m[:, None] & mask_n[None, :]
    tl.store(C_ptr + offs_c, acc, mask=mask_c)

def matmul_int8(A, B, AS, BS, M, N, K, SPLIT_K=1, BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32, out=None):
    if out is None:
        out = torch.empty((M, N), device=A.device, dtype=torch.float32)
    grid = lambda meta: (
        (M + meta["BLOCK_SIZE_M"] - 1) // meta["BLOCK_SIZE_M"],
        (N + meta["BLOCK_SIZE_N"] - 1) // meta["BLOCK_SIZE_N"],
        SPLIT_K,
    )
    matmul_kernel[grid](
        A, B, out,
        AS, BS,
        M, N, K,
        A.stride(0), A.stride(1),
        B.stride(0), B.stride(1),
        out.stride(0), out.stride(1),
        SPLIT_K,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_N=BLOCK_SIZE_N,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
    )
    return out

def matmul_quantize_int8(fpa, Bq, BS, M, N, K):
    Aq, AS = quantize_int8_perrow(fpa)
    return matmul_int8(Aq, Bq, AS, BS, M, N, K)

def quantize_int8(tensor: torch.Tensor, axis: int = 0):
    abs_tensor = tensor.abs()
    max_vals, _ = abs_tensor.max(dim=axis, keepdim=True)
    scale = (127.0 / max_vals).clamp_min(1e-6)
    q_tensor = torch.round(tensor * scale).clamp(-127, 127).to(torch.int8)
    return q_tensor, max_vals.squeeze(axis)
