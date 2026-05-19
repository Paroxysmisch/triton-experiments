import torch
import triton
import triton.language as tl

@triton.jit
def _adaptive_avg_pool2d_kernel(
    x_ptr,  # Pointer to the input tensor
    output_ptr,  # Pointer to the output tensor
    N,  # Number of batches
    C,  # Number of channels
    H,  # Input height
    W,  # Input width
    OH,  # Output height
    OW,  # Output width
    HBLOCK: tl.constexpr,  # Block size for height
    WBLOCK: tl.constexpr  # Block size for width
):
    pid = tl.program_id(0)
    block_start_h = pid // OW
    block_start_w = pid % OW
    block_size_h = H // OH
    block_size_w = W // OW

    for h in range(block_start_h * HBLOCK, (block_start_h + 1) * HBLOCK):
        for w in range(block_start_w * WBLOCK, (block_start_w + 1) * WBLOCK):
            if h < H and w < W:
                block_h = h // block_size_h
                block_w = w // block_size_w
                out_index = block_h * OW + block_w
                for c in range(C):
                    in_index = c * H * W + h * W + w
                    val = tl.load(x_ptr + in_index)
                    tl.atomic_add(output_ptr + c * OH * OW + out_index, val)

@triton.jit
def _pairwise_distance_kernel(
    x1_ptr,  # Pointer to the first input tensor
    x2_ptr,  # Pointer to the second input tensor
    dist_ptr,  # Pointer to the output distance tensor
    N,  # Number of batches
    C,  # Number of channels
    H,  # Input height
    W,  # Input width
    p: tl.float32,  # Norm degree
    eps: tl.float32,  # Small value to avoid division by zero
    XBLOCK: tl.constexpr,  # Block size for X dimension
    YBLOCK: tl.constexpr  # Block size for Y dimension
):
    pid = tl.program_id(0)
    xoffset = pid * XBLOCK
    xindex = xoffset + tl.arange(0, XBLOCK)
    yindex = tl.arange(0, YBLOCK)

    xmask = xindex < N
    ymask = yindex < N

    dist = tl.zeros((XBLOCK, YBLOCK), dtype=tl.float32)
    for c in range(C):
        for h in range(H):
            for w in range(W):
                in1_index = c * H * W + h * W + w
                in2_index = c * H * W + h * W + w
                x1_val = tl.load(x1_ptr + in1_index, xmask, other=0)
                x2_val = tl.load(x2_ptr + in2_index, ymask, other=0)
                diff = x1_val - x2_val
                dist += tl.abs(diff) ** p

    dist = (dist + eps) ** (1 / p)
    tl.store(dist_ptr + xindex * N + yindex, dist, xmask & ymask)

### Python Wrapper Function
