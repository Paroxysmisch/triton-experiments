import torch
import triton
import triton.language as tl

@triton.jit
def cosine_similarity_kernel(
    x1_ptr,
    x2_ptr,
    output_ptr,
    B,
    C,
    H,
    W,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_h = tl.program_id(1)
    pid_w = tl.program_id(2)

    if pid_b >= B or pid_h >= H or pid_w >= W:
        return

    x1_offset_base = pid_b * C * H * W + pid_h * W + pid_w
    x2_offset_base = pid_b * C * H * W + pid_h * W + pid_w

    dot_product = 0.0
    norm_x1 = 0.0
    norm_x2 = 0.0

    for c in range(0, C, BLOCK_SIZE):
        c_offsets = c + tl.arange(0, BLOCK_SIZE)
        mask = c_offsets < C

        x1_val = tl.load(x1_ptr + x1_offset_base + c_offsets * H * W, mask=mask, other=0.0)
        x2_val = tl.load(x2_ptr + x2_offset_base + c_offsets * H * W, mask=mask, other=0.0)

        dot_product += tl.sum(x1_val * x2_val)
        norm_x1 += tl.sum(x1_val * x1_val)
        norm_x2 += tl.sum(x2_val * x2_val)

    norm = tl.sqrt(norm_x1 * norm_x2) + eps
    cosine_sim = dot_product / norm

    output_offset = pid_b * H * W + pid_h * W + pid_w
    tl.store(output_ptr + output_offset, cosine_sim)

@triton.jit
def avg_pool2d_kernel(
    input_ptr,
    output_ptr,
    B,
    C,
    H,
    W,
    kernel_h,
    kernel_w,
    stride_h,
    stride_w,
    padding_h,
    padding_w,
    eps,
    BLOCK_SIZE: tl.constexpr,
):
    pid_b = tl.program_id(0)
    pid_oh = tl.program_id(1)
    pid_ow = tl.program_id(2)

    OH = (H + 2 * padding_h - kernel_h) // stride_h + 1
    OW = (W + 2 * padding_w - kernel_w) // stride_w + 1

    if pid_b >= B or pid_oh >= OH or pid_ow >= OW:
        return

    h_start = pid_oh * stride_h - padding_h
    w_start = pid_ow * stride_w - padding_w
    h_end = h_start + kernel_h
    w_end = w_start + kernel_w

    h_start_clamped = tl.maximum(h_start, 0)
    w_start_clamped = tl.maximum(w_start, 0)
    h_end_clamped = tl.minimum(h_end, H)
    w_end_clamped = tl.minimum(w_end, W)

    total = 0.0
    count = 0

    for h in range(h_start_clamped, h_end_clamped):
        for w in range(w_start_clamped, w_end_clamped):
            input_idx = pid_b * C * H * W + h * W + w
            val = tl.load(input_ptr + input_idx)
            total += val
            count += 1

    avg = total / (count + eps)
    output_idx = pid_b * OH * OW + pid_oh * OW + pid_ow
    tl.store(output_ptr + output_idx, avg)

def fused_avg_pool2d_cosine_similarity(
    x1: torch.Tensor,
    x2: torch.Tensor,
    kernel_size: int,
    stride: int = None,
    padding: int = 0,
    eps: float = 1e-8
) -> torch.Tensor:
    assert x1.device.type == 'cuda' and x2.device.type == 'cuda', "Inputs must be on CUDA"
    assert x1.shape == x2.shape, "x1 and x2 must have the same shape"

    if stride is None:
        stride = kernel_size

    B, C, H, W = x1.shape
    output_cos = torch.empty((B, H, W), device=x1.device)

    grid = (B, H, W)
    BLOCK_SIZE = 256
    cosine_similarity_kernel[grid](
        x1_ptr=x1,
        x2_ptr=x2,
        output_ptr=output_cos,
        B=B,
        C=C,
        H=H,
        W=W,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    output_cos_unsqueezed = output_cos.unsqueeze(1)

    kernel_h = kernel_w = kernel_size
    stride_h = stride_w = stride
    padding_h = padding_w = padding

    H_pool = (H + 2 * padding_h - kernel_h) // stride_h + 1
    W_pool = (W + 2 * padding_w - kernel_w) // stride_w + 1
    output_pool = torch.empty((B, 1, H_pool, W_pool), device=x1.device)

    avg_pool_grid = (B, H_pool, W_pool)
    avg_pool2d_kernel[avg_pool_grid](
        input_ptr=output_cos_unsqueezed,
        output_ptr=output_pool,
        B=B,
        C=1,
        H=H,
        W=W,
        kernel_h=kernel_h,
        kernel_w=kernel_w,
        stride_h=stride_h,
        stride_w=stride_w,
        padding_h=padding_h,
        padding_w=padding_w,
        eps=eps,
        BLOCK_SIZE=BLOCK_SIZE,
    )

    return output_pool
