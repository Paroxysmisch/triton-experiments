import triton
import triton.language as tl
import torch

@triton.jit
def rms_matmul_rbe(x_ptr: tl.pointer_type,
                   w_ptr: tl.pointer_type,
                   rms_w_ptr: tl.pointer_type,
                   output_ptr: tl.pointer_type,
                   M: tl.uint32,
                   N: tl.uint32,
                   K: tl.uint32,
                   BLOCK_SIZE_M: tl.constexpr,
                   BLOCK_SIZE_N: tl.constexpr,
                   BLOCK_SIZE_K: tl.constexpr,
                   THETA: tl.float32 = 0.0,
                   apply_rotary: tl.constexpr = False):
    # Program IDs
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Block pointers
    x_block_ptr = x_ptr + pid_m * BLOCK_SIZE_M * K
    w_block_ptr = w_ptr + pid_n * BLOCK_SIZE_N * K

    # Shared memory for storing intermediate values
    acc = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)

    # Iterate over K dimension in blocks
    for k in range(0, K, BLOCK_SIZE_K):
        # Load blocks of x and w
        x = tl.load(x_block_ptr + k, mask=True)
        w = tl.load(w_block_ptr + k, mask=True)

        # Apply RMS normalization to x
        x_sq = x * x
        rms = tl.sqrt(tl.sum(x_sq, axis=1) / K)
        x_norm = x / rms[:, None]

        # Perform matrix multiplication
        acc += tl.dot(x_norm, w)

    # Apply rotary embeddings if specified
    if apply_rotary:
        pos = tl.arange(0, BLOCK_SIZE_M)[:, None]
        angle = THETA * pos
        cos, sin = tl.cos(angle), tl.sin(angle)
        acc_rot = acc * cos - acc.flip(axis=1) * sin
        acc = acc_rot

    # Store the result
    tl.store(output_ptr + pid_m * BLOCK_SIZE_M * N + pid_n * BLOCK_SIZE_N, acc)

def rms_matmul_rbe_wrapper(x, w, rms_w, THETA=0.0, apply_rotary=False):
    # Ensure inputs are on CUDA
    assert x.is_cuda and w.is_cuda and rms_w.is_cuda, "Expected CUDA tensors"

    # Dimensions
    B, M, K = x.shape
    _, N, _ = w.shape

    # Output tensor
    output = torch.empty((B, M, N), device=x.device, dtype=torch.float32)

    # Launch the Triton kernel
    grid = (M // BLOCK_SIZE_M, N // BLOCK_SIZE_N)
    rms_matmul_rbe[grid](
        x, w, rms_w, output,
        M, N, K,
        BLOCK_SIZE_M=128, BLOCK_SIZE_N=128, BLOCK_SIZE_K=32,
        THETA=THETA, apply_rotary=apply_rotary
    )

    return output
