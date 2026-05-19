import triton
import triton.language as tl
import torch

@triton.jit
def ff_llama(x_ptr: tl.pointer_type,
             w1_ptr: tl.pointer_type,
             w3_ptr: tl.pointer_type,
             rms_w_ptr: tl.pointer_type,
             output_ptr: tl.pointer_type,
             x_row_stride: tl.uint32,
             w1_row_stride: tl.uint32,
             w3_row_stride: tl.uint32,
             rms_w_row_stride: tl.uint32,
             M: tl.uint32,
             N: tl.uint32,
             K: tl.uint32,
             eps: tl.float32,
             BLOCK_SIZE_M: tl.constexpr,
             BLOCK_SIZE_N: tl.constexpr,
             BLOCK_SIZE_K: tl.constexpr):
    pid = tl.program_id(0)
    num_pid_m = tl.cdiv(M, BLOCK_SIZE_M)
    num_pid_n = tl.cdiv(N, BLOCK_SIZE_N)
    num_pid_in_group = num_pid_m * num_pid_n
    group_id = pid // num_pid_in_group
    first_pid_m = group_id * num_pid_m
    group_size_m = num_pid_m
    pid_m = first_pid_m + (pid % group_size_m)
    pid_n = (pid % num_pid_n)
    offs_am = (pid_m * BLOCK_SIZE_M + tl.arange(0, BLOCK_SIZE_M)) % M
    offs_bn = (pid_n * BLOCK_SIZE_N + tl.arange(0, BLOCK_SIZE_N)) % N
    offs_k = tl.arange(0, BLOCK_SIZE_K)
    x_ptrs = x_ptr + (offs_am[:, None] * x_row_stride + offs_k[None, :])
    w1_ptrs = w1_ptr + (offs_k[:, None] * w1_row_stride + offs_bn[None, :])
    w3_ptrs = w3_ptr + (offs_k[:, None] * w3_row_stride + offs_bn[None, :])
    rms_w_ptrs = rms_w_ptr + (offs_am * rms_w_row_stride)

    # Load x and weights
    x = tl.load(x_ptrs, mask=offs_k[None, :] < K, other=0.0)
    w1 = tl.load(w1_ptrs, mask=offs_k[:, None] < K, other=0.0)
    w3 = tl.load(w3_ptrs, mask=offs_k[:, None] < K, other=0.0)
    rms_w = tl.load(rms_w_ptrs, mask=offs_am < M, other=1.0)

    # Compute RMS normalization
    squared_x = x * x
    squared_mean = tl.sum(squared_x, axis=1) / K
    rms = tl.sqrt(squared_mean + eps)
    normalized_x = x / rms[:, None]

    # Apply RMS weight
    scaled_x = normalized_x * rms_w[:, None]

    # Matrix multiplications
    acc1 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    acc3 = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=tl.float32)
    for k in range(0, K, BLOCK_SIZE_K):
        a = scaled_x[:, k:k + BLOCK_SIZE_K]
        b1 = w1[k:k + BLOCK_SIZE_K, :]
        b3 = w3[k:k + BLOCK_SIZE_K, :]
        acc1 += tl.dot(a, b1)
        acc3 += tl.dot(a, b3)

    # Apply SILU activation to the result of w1
    silu_acc1 = tl.sigmoid(acc1) * acc1

    # Combine results
    output = silu_acc1 * acc3

    # Store the result
    output_ptrs = output_ptr + (offs_am[:, None] * x_row_stride + offs_bn[None, :])
    tl.store(output_ptrs, output, mask=offs_bn[None, :] < N)

class KernelFF_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w3, rms_w):
        # Save inputs for backward pass
        ctx.save_for_backward(x, w1, w3, rms_w)

        M, K = x.shape
        N = w1.shape[1]
        assert w3.shape == w1.shape, "Dimension mismatch for w1 and w3"
        assert rms_w.shape[0] == M, "Dimension mismatch for rms_w"

        assert x.is_cuda and w1.is_cuda and w3.is_cuda and rms_w.is_cuda, "Expected CUDA tensors"
        assert x.is_contiguous() and w1.is_contiguous() and w3.is_contiguous() and rms_w.is_contiguous(), "Expected contiguous tensors"

        # Allocate output tensor
        output = torch.empty((M, N), device=x.device, dtype=x.dtype)

        # Define block sizes
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 32

        # Launch the kernel
        grid = (M * N // (BLOCK_SIZE_M * BLOCK_SIZE_N),)
        ff_llama[grid](
            x, w1, w3, rms_w, output,
            x.stride(0), w1.stride(0), w3.stride(0), rms_w.stride(0),
            M, N, K, 1e-5,
            BLOCK_SIZE_M, BLOCK_SIZE_N, BLOCK_SIZE_K,
            num_warps=4
        )

        return output

    @staticmethod
    def backward(ctx, grad_output):
        x, w1, w3, rms_w = ctx.saved_tensors
        M, K = x.shape
        N = w1.shape[1]

        # Allocate gradient tensors
        grad_x = torch.empty_like(x)
        grad_w1 = torch.empty_like(w1)
        grad_w3 = torch.empty_like(w3)
        grad_rms_w = torch.empty_like(rms_w)

        # Define block sizes
        BLOCK_SIZE_M = 16
        BLOCK_SIZE_N = 16
        BLOCK_SIZE_K = 32

        # Launch the kernel for backward pass
        grid = (M * N // (BLOCK_SIZE_M * BLOCK_SIZE_N),)
        # Note: Backward pass kernel implementation is not provided here
        # but should be similar to the forward pass, computing gradients
        # and storing them in the allocated gradient tensors.

        return grad_x, grad_w1, grad_w3, grad_rms_w

# Example usage
if __name__ == "__main__":
    M, K, N = 128, 256, 128
    x = torch.randn((M, K), device="cuda", dtype=torch.float16)
    w1 = torch.randn((K, N), device="cuda", dtype=torch.float16)
    w3 = torch.randn((K, N), device="cuda", dtype=torch.float16)
    rms_w = torch.randn((M,), device="cuda", dtype=torch.float16)

    output = KernelFF_Triton.apply(x, w1, w3, rms_w)
    print(output.shape)  # Should be (M, N)
