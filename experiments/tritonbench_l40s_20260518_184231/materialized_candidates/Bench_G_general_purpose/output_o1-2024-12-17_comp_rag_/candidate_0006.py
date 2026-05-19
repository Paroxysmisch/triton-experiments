import triton
import triton.language as tl
import torch

# -----------------------------------------------------------------------------------
# Fused Feed-Forward Kernel for LLAMA: ff_llama_kernel
# -----------------------------------------------------------------------------------
@triton.jit
def ff_llama_kernel(
    x_ptr,         # [M, K]
    w1_ptr,        # [K, N]
    w3_ptr,        # [K, N]
    rms_w_ptr,     # [N], RMS scaling weights
    out_ptr,       # [M, N]
    M, N, K,
    stride_xm, stride_xk,
    stride_w1k, stride_w1n,
    stride_w3k, stride_w3n,
    stride_outm, stride_outn,
    EPS,           # Epsilon for RMS stability
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    USE_FP8: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)
    row_off = pid_m * BLOCK_SIZE_M
    col_off = pid_n * BLOCK_SIZE_N

    # Create 2D pointers for the output tile
    offs_m = row_off + tl.arange(0, BLOCK_SIZE_M)
    offs_n = col_off + tl.arange(0, BLOCK_SIZE_N)
    # Create an accumulator for the two matrix products
    acc1 = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)
    acc2 = tl.zeros([BLOCK_SIZE_M, BLOCK_SIZE_N], dtype=tl.float32)

    # k-range in steps of BLOCK_SIZE_K
    for k_block in range(0, K, BLOCK_SIZE_K):
        # Compute offset for partial chunk
        offs_k = k_block + tl.arange(0, BLOCK_SIZE_K)
        # Load a block of x
        x_mask = (offs_m[:, None] < M) & (offs_k[None, :] < K)
        x_ptrs = x_ptr + (offs_m[:, None] * stride_xm) + (offs_k[None, :] * stride_xk)
        x_block = tl.load(x_ptrs, mask=x_mask, other=0.0).to(tl.float32)

        # Load a block of w1
        w1_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w1_ptrs = w1_ptr + (offs_k[:, None] * stride_w1k) + (offs_n[None, :] * stride_w1n)
        if USE_FP8:
            w1_block = tl.load(w1_ptrs, mask=w1_mask, other=0.0).to(tl.float16).to(tl.float32)
        else:
            w1_block = tl.load(w1_ptrs, mask=w1_mask, other=0.0).to(tl.float32)

        # Compute partial product for acc1
        acc1 += tl.dot(x_block, w1_block)

        # Load a block of w3
        w3_mask = (offs_k[:, None] < K) & (offs_n[None, :] < N)
        w3_ptrs = w3_ptr + (offs_k[:, None] * stride_w3k) + (offs_n[None, :] * stride_w3n)
        if USE_FP8:
            w3_block = tl.load(w3_ptrs, mask=w3_mask, other=0.0).to(tl.float16).to(tl.float32)
        else:
            w3_block = tl.load(w3_ptrs, mask=w3_mask, other=0.0).to(tl.float32)

        # Compute partial product for acc2
        acc2 += tl.dot(x_block, w3_block)

    # Now acc1 = w1(x), acc2 = w3(x)
    # RMS normalizing each row in the tile
    # We compute L2-norm across the N dimension (per row). The tile might partially cover M dimension.
    # For numerical correctness, we'll do a per-element approach inside this tile

    # Each thread is responsible for a sub-block of size [BLOCK_SIZE_M, BLOCK_SIZE_N].
    # Compute the L2 norm for each row
    # (acc1 is [BLOCK_SIZE_M, BLOCK_SIZE_N], so we sum squares across N dimension)
    sq_acc1 = acc1 * acc1
    l2_norm_row = tl.sum(sq_acc1, 1)  # shape [BLOCK_SIZE_M]
    l2_norm_row = l2_norm_row / tl.float32(BLOCK_SIZE_N)
    l2_norm_row = tl.sqrt(l2_norm_row + EPS)

    # Expand dims to match [BLOCK_SIZE_M, BLOCK_SIZE_N]
    l2_norm_tile = l2_norm_row[:, None]

    # Multiply by RMS weight
    offs_n_2d = col_off + tl.arange(0, BLOCK_SIZE_N)
    w_rms_mask = offs_n_2d < N
    w_rms_vals = tl.load(rms_w_ptr + offs_n_2d, mask=w_rms_mask, other=1.0)

    # Normalized + scaled
    acc1 = (acc1 / l2_norm_tile) * w_rms_vals[None, :]

    # Apply siLU activation
    # siLU(x) = x * sigmoid(x)
    sig = 1 / (1 + tl.exp(-acc1))  # sigmoid
    acc1 = acc1 * sig

    # Multiply by acc2
    out_val = acc1 * acc2

    # Store output (if within bounds)
    out_mask_m = offs_m < M
    out_mask_n = offs_n < N
    out_ptrs = out_ptr + (offs_m[:, None] * stride_outm) + (offs_n[None, :] * stride_outn)
    tl.store(out_ptrs, out_val, mask=out_mask_m[:, None] & out_mask_n[None, :])


# -----------------------------------------------------------------------------------
# Python wrapper: kernel_ff
# -----------------------------------------------------------------------------------
class FF_LLaMA_Triton(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, w1, w3, rms_w, use_fp8=False):
        """
        x: [M, K]
        w1, w3: [K, N]
        rms_w: [N]
        """
        assert x.is_cuda and w1.is_cuda and w3.is_cuda and rms_w.is_cuda, "All inputs must be CUDA tensors"
        assert x.dim() == 2 and w1.dim() == 2 and w3.dim() == 2, "x, w1, and w3 must be 2D"
        assert rms_w.dim() == 1, "rms_w must be 1D"
        assert x.size(1) == w1.size(0) == w3.size(0), "Shapes must match for matrix multiplication"
        assert w1.size(1) == w3.size(1) == rms_w.size(0), "Output dimension must match"

        M, K = x.shape
        _, N = w1.shape

        # Save for backward
        ctx.save_for_backward(x, w1, w3, rms_w)
        ctx.use_fp8 = use_fp8

        # Create output
        out = torch.empty((M, N), device=x.device, dtype=torch.float32)

        # Compute block sizes
        BLOCK_SIZE_M = 64
        BLOCK_SIZE_N = 64
        BLOCK_SIZE_K = 32

        grid_m = (M + BLOCK_SIZE_M - 1) // BLOCK_SIZE_M
        grid_n = (N + BLOCK_SIZE_N - 1) // BLOCK_SIZE_N

        ff_llama_kernel[grid_m, grid_n](
            x, w1, w3, rms_w, out,
            M, N, K,
            x.stride(0), x.stride(1),
            w1.stride(0), w1.stride(1),
            w3.stride(0), w3.stride(1),
            out.stride(0), out.stride(1),
            1e-5,  # EPS
            BLOCK_SIZE_M=BLOCK_SIZE_M,
            BLOCK_SIZE_N=BLOCK_SIZE_N,
            BLOCK_SIZE_K=BLOCK_SIZE_K,
            USE_FP8=use_fp8
        )
        return out

    @staticmethod
    def backward(ctx, grad_out):
        """
        No backward pass implemented here for demonstration.
        Return None for all since we are not computing gradients.
        """
        return None, None, None, None, None


def kernel_ff(x, w1, w3, rms_w, use_fp8=False):
    return FF_LLaMA_Triton.apply(x, w1, w3, rms_w, use_fp8)
