import torch
import triton
import triton.language as tl

@triton.jit
def _cosine_similarity_kernel(
    x1_ptr, x2_ptr, out_ptr,
    B, C, H, W,
    stride_bxc1, stride_bxc2,  # Bytes strides for batch*x, channel*x in x1
    stride_hx1, stride_wx1,     # Bytes strides for height*x, width*x in x1
    stride_bxc3, stride_bxc4,  # Bytes strides for batch*x, channel*x in x2
    stride_hx2, stride_wx2,     # Bytes strides for height*x, width*x in x2
    stride_bxout, stride_hxout, stride_wxout,  # Bytes strides for out
    eps,
    BLOCK_SIZE: tl.constexpr
):
    """
    For each index [n, h, w], compute:
        dot = sum_{c} x1[n,c,h,w]* x2[n,c,h,w]
        norm1 = sqrt( sum_{c} x1[n,c,h,w]^2 )
        norm2 = sqrt( sum_{c} x2[n,c,h,w]^2 )
        out[n,h,w] = dot / (norm1 * norm2 + eps)
    """
    pid = tl.program_id(0)
    # We map pid to a global index over N*H*W
    # B*H*W total elements in the output. We'll flatten (N, H, W).
    # The block covers BLOCK_SIZE elements in this flattened space.
    start = pid * BLOCK_SIZE
    end = tl.minimum(start + BLOCK_SIZE, B * H * W)

    for idx in range(start, end):
        n = idx // (H * W)
        hw = idx % (H * W)
        hh = hw //
