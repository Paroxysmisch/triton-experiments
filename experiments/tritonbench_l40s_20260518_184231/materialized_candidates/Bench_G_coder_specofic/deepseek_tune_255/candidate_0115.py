import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
    x_ptr,
    rms_w_ptr,
    out_ptr,
    M,
    K,
    eps,
    stride_x_batch,
    stride_x_M,
    stride_x_K,
    stride_rms_w_M,
    stride_out_batch,
    stride_out_M,
    stride_out_K,
    NUM_WARPS: tl.constexpr,
    BLOCK_N_SIZE: tl.constexpr,
):
    program_id = tl.program_id(0)
    batch_id = tl.program_id(1)

    M_offset = tl.arange(0, NUM_WARPS) * BLOCK_N_SIZE
    K_offset = tl.arange(0, BLOCK_N_SIZE)

    rms_w_ptrs = rms_w_ptr + M_offset * stride_rms_w_M

    rms = tl.zeros([NUM_WARPS, ], dtype=tl.float32)
    for i in range(M // BLOCK_N_SIZE):
        x_ptrs = x_ptr + batch_id * stride_x_batch + (M_offset + i * BLOCK_N_SIZE) * stride_x_M + K_offset * stride_x_K
        x = tl.load(x_ptrs, mask=M_offset + i * BLOCK_N_SIZE < M, other=0.0).to(tl.float32)
        rms += x * x

    rms = tl.sum(rms, axis=0) / (M * K)
    rms = tl.sqrt(rms)
    rms = rms + 1e-5

    for i in range(M // BLOCK_N_SIZE):
        x_ptrs = x_ptr + batch_id * stride_x_batch + (M_offset + i * BLOCK_N_SIZE) * stride_x_M + K_offset * stride_x_K
        x = tl.load(x_ptrs, mask=M_offset + i * BLOCK_N_SIZE < M, other=0.0).to(tl.float32)
        rms_w = tl.load(rms_w_ptrs, mask=M_offset + i * BLOCK_N_SIZE < M, other=0.0).to(tl.float32)

        x_hat = x / rms
        out = x_hat * rms_w

        out_ptrs = out_ptr + batch_id * stride_out_batch + (M_offset + i * BLOCK_N_SIZE) * stride_out_M + K_offset * stride_out_K
        tl.store(out_ptrs, out.to(out_ptr.dtype.element_ty), mask=M_offset + i * BLOCK_N_SIZE < M)

def rmsnorm_wrapper(x, rms_w, eps):
    M, K = x.shape[-2:]
    out = torch.empty_like(x)
    BLOCK_N_SIZE = 32
    num_warps = 8
    if K <= 128:
        num_warps = 4
    elif K <= 256:
        num_warps = 8
    rmsnorm_triton[(x.shape[0], M // BLOCK_N_SIZE)](
        x,
        rms_w,
        out,
        M,
        K,
        eps,
        x.stride(0),
        x.stride(1),
        x.stride(2),
        rms_w.stride(0),
        out.stride(0),
        out.stride(1),
        out.stride(2),
        NUM_WARPS=num_warps,
        BLOCK_N_SIZE=BLOCK_N_SIZE,
        num_warps=num_warps,
    )
    return out
