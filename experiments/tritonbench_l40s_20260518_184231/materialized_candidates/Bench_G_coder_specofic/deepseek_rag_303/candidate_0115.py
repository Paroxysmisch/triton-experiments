import torch
import triton
import triton.language as tl

@triton.jit
def rmsnorm_triton(
        x_ptr,
        rms_w_ptr,
        out_ptr,
        M, K,
        x_batch_stride, x_m_stride, x_k_stride,
        rms_w_k_stride,
        out_batch_stride, out_m_stride, out_k_stride,
        N_SIZE,
        eps,
        BLOCK_N_SIZE: tl.constexpr,
        num_warps: tl.constexpr,
):
    # Assumes N is power of 2 for now
    batch_idx = tl.program_id(0)
    m_idx = tl.program_id(1)

    offsets = tl.arange(0, BLOCK_N_SIZE)

    x_ptr_offset = x_ptr + batch_idx * x_batch_stride + m_idx * x_m_stride
    x_ptrs = x_ptr_offset + offsets * x_k_stride
    mask = (offsets < K)

    cumsum_x = 0.0
    cumsum_x_squared = 0.0

    for off_in_mask in range(0, tl.cdiv(N_SIZE, BLOCK_N_SIZE)):
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        cumsum_x += tl.sum(x)
        cumsum_x_squared += tl.sum(x * x)
        # Advance the ptrs for the mask and the for loop
        x_ptrs += BLOCK_N_SIZE * x_k_stride
        mask = (offsets + off_in_mask * BLOCK_N_SIZE < K)

    var = tl.maximum((cumsum_x_squared - cumsum_x * cumsum_x / N_SIZE), 0.0)

    rms = tl.math.sqrt(var / N_SIZE + eps)
    inv_rms = tl.math.rsqrt(rms)

    # Reset to first block
    cumsum_x = 0.0
    cumsum_x_squared = 0.0

    x_ptrs = tl.math.exp2(tl.math.ctz(offsets)) + x_ptr_offset
    mask = (offsets < K)

    for log2_stride in range(tl.math.log2(BLOCK_N_SIZE).to(int), tl.math.log2(N_SIZE).to(int) + 1):
        stride = tl.math.exp2(log2_stride)
        x = tl.load(x_ptrs, mask=mask, other=0.0)
        x_normalized = x * inv_rms
        cumsum_x += tl.sum(x_normalized)
        cumsum_x_squared += tl.sum(x_normalized * x_normalized)

        # Write to output
        out_ptr_offset = out_ptr + batch_idx * out_batch_stride + m_idx * out_m_stride
        out_ptrs = out_ptr_offset + (offsets * stride)
        normalized_mask = (offsets % stride == 0)
        tl.store(out_ptrs, x_normalized, mask=(normalized_mask & mask))
        tl.store(out_ptr_offset + rms_w_k_stride * m_idx + offsets, rms, mask=(normalized_mask & mask))
        # Advance the ptrs for the mask
        x_ptrs += stride * x_k_stride


def rmsnorm_wrapper(x, rms_w):
    M, K = x.shape[-2:]
    x = x.unsqueeze(dim=-1)
    out = torch.empty_like(x)
    rmsnorm_triton[(x.shape[0], M)](
        x, rms_w, out,
        M, K,
        x.stride(0), x.stride(1), x.stride(2),
        rms_w.stride(0),
        out.stride(0), out.stride(1), out.stride(2),
        N_SIZE=K, eps=1e-32,
        BLOCK_N_SIZE=16,
        num_warps=4,
        num_stages=2,
        num_ctas=8
    )
    # [batch, M, K]
    return out.squeeze(dim=-1)
