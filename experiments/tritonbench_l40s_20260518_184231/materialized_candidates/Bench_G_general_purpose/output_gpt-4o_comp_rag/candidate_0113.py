import triton
import triton.language as tl
import torch

@triton.jit
def rmsnorm_triton(x_ptr: tl.pointer_type,
                   rms_w_ptr: tl.pointer_type,
                   out_ptr: tl.pointer_type,
                   batch_stride: tl.uint32,
                   M_stride: tl.uint32,
                   K: tl.uint32,
                   eps: tl.float32,
                   BLOCK_N_SIZE: tl.constexpr):
    batch_id = tl.program_id(0)
    M_id = tl.program_id(1)

    # Compute the starting pointer for this instance
    x_start_ptr = x_ptr + batch_id * batch_stride + M_id * M_stride
    out_start_ptr = out_ptr + batch_id * batch_stride + M_id * M_stride

    # Compute offsets within the K dimension
    offsets = tl.arange(0, BLOCK_N_SIZE)
    mask = offsets < K

    # Load input and RMS weights
    x = tl.load(x_start_ptr + offsets, mask=mask, other=0.0)
    rms_weights = tl.load(rms_w_ptr + offsets, mask=mask, other=1.0)

    # Compute RMS normalization
    squared_x = x * x
    squared_mean = tl.sum(squared_x, axis=0) / K
    rms = tl.sqrt(squared_mean + eps)

    # Normalize and apply RMS weights
    normalized_x = x / rms
    scaled_x = normalized_x * rms_weights

    # Store the result
    tl.store(out_start_ptr + offsets, scaled_x, mask=mask)

def rmsnorm_wrapper(x, rms_weights, eps=1e-6, BLOCK_N_SIZE=128, num_warps=4):
    batch, M, K = x.shape
    assert x.is_cuda and rms_weights.is_cuda, "Expected CUDA tensors"
    assert x.is_contiguous(), "Input tensor must be contiguous"

    # Allocate output tensor
    y = torch.empty_like(x)

    # Launch the kernel
    grid = (batch, M)
    rmsnorm_triton[grid](
        x, rms_weights, y,
        x.stride(0), x.stride(1), K,
        eps, BLOCK_N_SIZE=BLOCK_N_SIZE,
        num_warps=num_warps
    )

    return y

# Example usage
x = torch.randn(32, 128, 256, device='cuda')  # Example tensor
rms_weights = torch.ones(256, device='cuda')  # Example RMS weights
y = rmsnorm_wrapper(x, rms_weights)
