import triton
import triton.language as tl
import torch

# Triton kernel for RMS normalization
@triton.jit
def rmsnorm_triton(x_ptr: tl.pointer_type,
                   rms_w_ptr: tl.pointer_type,
                   out_ptr: tl.pointer_type,
                   batch: tl.uint32,
                   M: tl.uint32,
                   K: tl.uint32,
                   eps: tl.float32,
                   BLOCK_N_SIZE: tl.constexpr,
                   num_warps: tl.constexpr):
    pid = tl.program_id(0)
    batch_idx = pid // M
    m_idx = pid % M

    block_start = batch_idx * M * K + m_idx * K
    offsets = block_start + tl.arange(0, BLOCK_N_SIZE)

    mask = offsets < (batch_idx * M * K + (m_idx + 1) * K)

    x_block = tl.load(x_ptr + offsets, mask=mask, other=0)
    rms_w_block = tl.load(rms_w_ptr + offsets % K, mask=mask, other=1)

    # Compute the squared sum
    squared_sum = tl.sum(x_block * x_block, axis=0)
    rms = tl.sqrt(squared_sum / K + eps)

    # Normalize and apply RMS weights
    normalized_block = x_block / rms
    out_block = normalized_block * rms_w_block

    # Store the result
    tl.store(out_ptr + offsets, out_block, mask=mask)

# Wrapper function to facilitate the execution of the kernel
def rmsnorm_wrapper(x: torch.Tensor, rms_w: torch.Tensor, eps: float = 1e-6):
    batch, M, K = x.shape

    assert rms_w.shape == (K,), "RMS weights must have the same size as the last dimension of the input tensor"
    assert x.is_cuda and rms_w.is_cuda, "Input tensors must be on the same CUDA device"
    assert x.is_contiguous() and rms_w.is_contiguous(), "Input tensors must be contiguous"

    out = torch.empty_like(x)

    # Determine the block size
    BLOCK_N_SIZE = triton.next_power_of_2(K)

    # Launch the kernel
    grid = (batch * M, )
    rmsnorm_triton[grid](
        x, rms_w, out,
        batch, M, K, eps,
        BLOCK_N_SIZE, num_warps=4
    )

    return out

# Example usage
if __name__ == "__main__":
    batch, M, K = 2, 3, 4
    x = torch.randn(batch, M, K, device="cuda")
    rms_w = torch.randn(K, device="cuda")

    out = rmsnorm_wrapper(x, rms_w)
    print(out)
