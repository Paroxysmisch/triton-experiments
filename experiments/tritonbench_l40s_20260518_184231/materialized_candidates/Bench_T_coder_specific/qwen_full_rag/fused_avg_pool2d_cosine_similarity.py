import torch
import triton
import triton.language as tl
from torch.nn.functional import avg_pool2d

@triton.jit
def cosine_similarity(x1, x2, eps: tl.constexpr = 1e-8):
    # Compute the cosine similarity between two tensors
    dot_product = tl.sum(x1 * x2, axis=-1)
    norm_x1 = tl.sqrt(tl.sum(x1 * x1, axis=-1))
    norm_x2 = tl.sqrt(tl.sum(x2 * x2, axis=-1))
    return dot_product / (norm_x1 * norm_x2 + eps)

@triton.jit
def _fused_avg_pool2d_cosine_similarity(x1, x2, out, kernel_size, stride, padding, eps: tl.constexpr = 1e-8):
    # Fused operation for average pooling and cosine similarity
    batch, channel, height, width = x1.shape
    pad_height = height + 2 * padding
    pad_width = width + 2 * padding
    pool_height = (height + 2 * padding - kernel_size) // stride + 1
    pool_width = (width + 2 * padding - kernel_size) // stride + 1

    y1 = tl.zeros([pool_height, pool_width], dtype=x1.dtype)
    y2 = tl.zeros([pool_height, pool_width], dtype=x2.dtype)
    idx_c = tl.arange(0, channel)
    idx_h = tl.arange(0, kernel_size)
    idx_w = tl.arange(0, kernel_size)
    idx_kh = tl.arange(0, 8)
    idx_kw = tl.arange(0, 8)
    off_h = idx_kh[:, None] * stride - padding
    off_w = idx_kw[None, :] * stride - padding
    pos_h = tl.where(off_h >= 0, off_h, 0)
    pos_w = tl.where(off_w >= 0, off_w, 0)
    
    for c in idx_c:
        offset = batch * channel * height * width + c * height * width
        for kh in idx_kh:
            for kw in idx_kw:
                idx_h = kh * 8 + idx_h
                idx_w = kw * 8 + idx_w
                idx = idx_h[:, None] * width + idx_w[None, :]
                offset_idx = offset + pos_h[:, None] * width + pos_w[None, :]
                x1_vals = tl.load(x1 + offset_idx, mask=(pos_h[:, None] < pad_height) & (pos_w[None, :] < pad_width), other=0.)
                x2_vals = tl.load(x2 + offset_idx, mask=(pos_h[:, None] < pad_height) & (pos_w[None, :] < pad_width), other=0.)
                y1 += x1_vals * x1_vals
                y2 += x2_vals * x2_vals
    
    y1 = y1 / (channel * kernel_size * kernel_size)
    y2 = y2 / (channel * kernel_size * kernel_size)
    y = cosine_similarity(y1, y2, eps=eps)
    y = y.to(out.dtype)
    idx_yh = tl.arange(0, 8)
    idx_yw = tl.arange(0, 8)
    off_yh = idx_yh[:, None] * stride
    off_yw = idx_yw[None, :] * stride
    pos_yh = tl.where(off_yh < pool_height, off_yh, pool_height - stride)
    pos_yw = tl.where(off_yw < pool_width, off_yw, pool_width - stride)
    for yh in idx_yh:
        for yw in idx_yw:
            pos_offset = yh * pool_width + yw
            select_h = pos_yh[yh, yw]
            select_w = pos_yw[yh, yw]
            out_val = tl.load(out + pos_offset, mask=True, other=0.)
            final_val = tl.where((select_h != pos_yh[yh, yw]) & (select_w != pos_yw[yh, yw]), 0., y)
            tl.store(out + pos_offset, final_val)

def fused_avg_pool2d_cosine_similarity(x1: torch.Tensor, x2: torch.Tensor, kernel_size: int, stride: int = None, padding: int = 0, eps: float = 1e-8) -> torch.Tensor:
    # Wrapper function for the Triton kernel
    if stride is None:
        stride = kernel_size
    assert x1.shape[-2:] == x2.shape[-2:], "The spatial dimensions of x1 and x2 must match"
    in_shape = x1.shape
    x1 = x1.reshape(-1, x1.shape[-2], x1.shape[-1])
    x2 = x2.reshape(-1, x2.shape[-2], x2.shape[-1])
    out = avg_pool2d(x1 * x2, kernel_size=kernel_size, stride=stride, padding=padding)
    out = out.unsqueeze(dim=2)
    M = x1.shape[-2] * x1.shape[-1]
    eps = eps * M
    _, C, H, W = out.shape
    out = out.reshape(C, H*W)
    grid = lambda META: (triton.cdiv(C, META["BLOCK_SIZE"]), 1)
    _fused_avg_pool2d_cosine_similarity[grid](x1, x2, out, kernel_size, stride, padding, eps=eps)
    out = out.squeeze(dim=2)
    out = out.reshape(in_shape[:-2] + out.shape[-2:])
    return out
